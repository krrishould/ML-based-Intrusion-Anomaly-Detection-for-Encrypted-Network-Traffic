"""Feature-table assembly and preprocessing.

Turns raw per-dataset flow records into the exact matrix the models see:

  1. Drop identifier columns (IPs, ports, timestamps).  Training on these is the
     classic encrypted-traffic leakage trap: the model memorises the capture
     lab's addressing plan, scores 0.99 offline, and is useless on real traffic.
  2. Replace the infinities CICFlowMeter emits for zero-duration flows.
  3. Median-impute genuinely missing values (CTU-13 has no packet-size
     distributions) and record which columns were imputed.
  4. Log1p-compress the heavy-tailed volume/rate columns, then standardise.

The fitted preprocessor is saved and reused verbatim at inference time, so live
traffic is transformed identically to the training data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..utils.logging_utils import get_logger
from . import schema

log = get_logger("features.build")

# Heavy-tailed columns: byte counts and rates span six orders of magnitude, so
# a raw standardisation would leave almost all mass in a single bin.
_LOG_COLUMNS = {
    "duration_ms", "fwd_packets", "bwd_packets", "total_packets",
    "fwd_bytes", "bwd_bytes", "total_bytes",
    "flow_bytes_per_s", "flow_packets_per_s", "fwd_packets_per_s",
    "bwd_packets_per_s", "flow_iat_min", "flow_iat_max", "flow_iat_mean",
    "flow_iat_std", "fwd_iat_mean", "fwd_iat_std", "bwd_iat_mean", "bwd_iat_std",
    "down_up_byte_ratio", "down_up_packet_ratio", "ja3_rarity",
    "fwd_header_bytes", "bwd_header_bytes", "handshake_duration_ms",
}


@dataclass
class FeaturePipeline:
    """Fit-once / apply-everywhere preprocessing for the flow feature matrix."""

    use_flow: bool = True
    use_tls: bool = True
    columns: list[str] = field(default_factory=list)
    log_columns: list[str] = field(default_factory=list)
    imputer: SimpleImputer | None = None
    scaler: StandardScaler | None = None
    imputed_columns: list[str] = field(default_factory=list)
    # JA3 fingerprint frequencies observed in the TRAINING corpus, kept so that
    # `ja3_rarity` can be computed identically at inference time.  See
    # `apply_fingerprints`.
    ja3_frequency_: dict[str, float] = field(default_factory=dict)

    # -- fit ---------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        self.columns = [c for c in schema.feature_columns(self.use_flow, self.use_tls)
                        if c in df.columns]
        missing = set(schema.feature_columns(self.use_flow, self.use_tls)) - set(self.columns)
        if missing:
            log.warning("Absent from the feature table, dropped: %s",
                        ", ".join(sorted(missing)))

        X = self._to_numeric(df)

        # A column with no observed value anywhere is not a feature - it is an
        # empty slot. Imputing it would invent a constant for every row, and
        # sklearn's imputer silently drops it instead, which desynchronises the
        # matrix from `self.columns`. Drop it here, explicitly and loudly.
        empty = [c for c in self.columns if X[c].isna().all()]
        if empty:
            log.warning("Dropping %d feature(s) with no observed values in the "
                        "training data: %s", len(empty), ", ".join(empty))
            self.columns = [c for c in self.columns if c not in set(empty)]
            X = X.drop(columns=empty)
        if not self.columns:
            raise ValueError("Every candidate feature is empty - check the "
                             "feature table")

        self.imputed_columns = [c for c in self.columns if X[c].isna().any()]
        if self.imputed_columns:
            log.info("Median-imputing %d column(s) with missing values",
                     len(self.imputed_columns))

        self.log_columns = [c for c in self.columns if c in _LOG_COLUMNS]
        self.imputer = SimpleImputer(strategy="median").fit(X)
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.columns,
                             index=X.index)
        self.scaler = StandardScaler().fit(self._compress(X_imp))

        if not self.ja3_frequency_:
            self.record_fingerprints(df)

        log.info("Feature pipeline fitted: %d features "
                 "(flow=%s, tls=%s)", len(self.columns), self.use_flow, self.use_tls)
        return self

    # -- fingerprint features ----------------------------------------------
    def record_fingerprints(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Freeze the JA3 frequency distribution of the training split.

        Call this on the TRAINING rows only, before transforming anything.
        Deriving the distribution from the full table would let the held-out
        rows influence a feature the model then uses to score them - a small
        leak, but a real one, and free to avoid.

        Frequency is relative to the flows that actually *have* a fingerprint.
        Including non-TLS flows in the denominator would mean that adding plain
        HTTP traffic to the corpus makes every TLS fingerprint look rarer, which
        is not what rarity should measure.
        """
        if "ja3_hash" not in df.columns:
            return self
        hashes = df["ja3_hash"].fillna("").astype(str)
        counts = hashes[hashes != ""].value_counts()
        total = max(int(counts.sum()), 1)
        self.ja3_frequency_ = {str(k): float(v) / total for k, v in counts.items()}
        log.info("Recorded %d distinct JA3 fingerprints for rarity scoring",
                 len(self.ja3_frequency_))
        return self

    def apply_fingerprints(self, df: pd.DataFrame,
                           n_buckets: int = 256) -> pd.DataFrame:
        """Add ``ja3_bucket`` / ``ja3s_bucket`` / ``ja3_rarity`` to live flows.

        Rarity is looked up in the frequency table frozen at fit time.  It must
        NOT be recomputed from the incoming batch: in a live window of a few
        dozen flows almost every fingerprint appears once, so every flow would
        score as maximally rare and the feature would carry no information at
        all - while still looking perfectly healthy in the output.

        A fingerprint never seen during training gets the maximum rarity of the
        training corpus rather than infinity, which is the honest reading: it is
        at least as rare as the rarest thing we have evidence for.
        """
        from ..features.tls_features import ja3_bucket

        out = df.copy()
        for source_col, target in (("ja3_hash", "ja3_bucket"),
                                   ("ja3s_hash", "ja3s_bucket")):
            series = (out[source_col] if source_col in out.columns
                      else pd.Series([""] * len(out), index=out.index))
            out[target] = series.fillna("").astype(str).map(
                lambda h: ja3_bucket(h, n_buckets))

        if self.ja3_frequency_:
            unseen = 1.0 / min(self.ja3_frequency_.values())
        else:
            unseen = 0.0
        ja3 = (out["ja3_hash"] if "ja3_hash" in out.columns
               else pd.Series([""] * len(out), index=out.index))
        out["ja3_rarity"] = ja3.fillna("").astype(str).map(
            lambda h: (1.0 / self.ja3_frequency_[h]
                       if h in self.ja3_frequency_ else (unseen if h else 0.0))
        ).clip(upper=1e6)
        return out

    # -- apply -------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.scaler is None:
            raise RuntimeError("FeaturePipeline.fit() must be called first")
        X = self._to_numeric(df)
        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.columns,
                             index=X.index)
        return self.scaler.transform(self._compress(X_imp))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    # -- internals ---------------------------------------------------------
    def _to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the model columns and coerce them to clean floats."""
        out = pd.DataFrame(index=df.index)
        for col in self.columns:
            series = df[col] if col in df.columns else pd.Series(np.nan, index=df.index)
            out[col] = pd.to_numeric(series, errors="coerce")
        # CICFlowMeter writes +/-Inf for rates on zero-duration flows.
        return out.replace([np.inf, -np.inf], np.nan)

    def _compress(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.log_columns:
            X[col] = np.log1p(np.clip(X[col].to_numpy(), 0, None))
        return X

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Feature pipeline -> %s", path)

    @staticmethod
    def load(path: str | Path) -> "FeaturePipeline":
        return joblib.load(path)


# ---------------------------------------------------------------------------
def clean_flow_table(df: pd.DataFrame, min_packets: int = 2) -> pd.DataFrame:
    """Drop degenerate rows before any modelling happens."""
    before = len(df)
    df = df.copy()

    if "total_packets" in df.columns:
        total = pd.to_numeric(df["total_packets"], errors="coerce").fillna(0)
        df = df[total >= min_packets]

    # Exact duplicates are usually a merge artefact, and they inflate scores by
    # putting the same flow in both train and test.
    feature_cols = [c for c in schema.FLOW_FEATURES if c in df.columns]
    if feature_cols:
        df = df.drop_duplicates(subset=feature_cols)

    df = df.reset_index(drop=True)
    log.info("Cleaning: %d -> %d flows (removed %d)", before, len(df),
             before - len(df))
    return df


def assemble(frames: list[pd.DataFrame], min_packets: int = 2,
             ja3_buckets: int = 256) -> pd.DataFrame:
    """Concatenate per-dataset frames into one schema-complete feature table."""
    from ..data.pcap_to_flows import add_fingerprint_buckets

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise ValueError("No data to assemble - enable a dataset in config.yaml")

    df = pd.concat(frames, ignore_index=True)
    df = add_fingerprint_buckets(df, n_buckets=ja3_buckets)
    df = clean_flow_table(df, min_packets=min_packets)

    if schema.BINARY_TARGET_COLUMN not in df.columns:
        df[schema.BINARY_TARGET_COLUMN] = (
            df[schema.TARGET_COLUMN] != schema.BENIGN_LABEL).astype(int)
    df[schema.BINARY_TARGET_COLUMN] = pd.to_numeric(
        df[schema.BINARY_TARGET_COLUMN], errors="coerce").fillna(0).astype(int)
    df[schema.TARGET_COLUMN] = df[schema.TARGET_COLUMN].fillna(schema.BENIGN_LABEL)

    log.info("Assembled table: %d flows, %d malicious (%.1f%%), %d classes",
             len(df), int(df[schema.BINARY_TARGET_COLUMN].sum()),
             100.0 * df[schema.BINARY_TARGET_COLUMN].mean(),
             df[schema.TARGET_COLUMN].nunique())
    return df


def summarise(df: pd.DataFrame) -> dict[str, Any]:
    """Small dict of dataset facts, written into the run report."""
    return {
        "n_flows": int(len(df)),
        "n_malicious": int(df[schema.BINARY_TARGET_COLUMN].sum()),
        "malicious_rate": float(df[schema.BINARY_TARGET_COLUMN].mean()),
        "n_classes": int(df[schema.TARGET_COLUMN].nunique()),
        "class_counts": {str(k): int(v) for k, v in
                         df[schema.TARGET_COLUMN].value_counts().items()},
        "sources": {str(k): int(v) for k, v in
                    df.get(schema.SOURCE_COLUMN,
                           pd.Series(dtype=str)).value_counts().items()},
        "tls_flow_rate": float(pd.to_numeric(
            df.get("is_tls", pd.Series(0, index=df.index)),
            errors="coerce").fillna(0).mean()),
    }
