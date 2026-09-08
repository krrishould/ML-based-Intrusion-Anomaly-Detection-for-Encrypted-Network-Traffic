"""Decision fusion - combining Stage 1 and Stage 2 into one verdict.

Per the project methodology, a flow is flagged if **either** stage fires:

    alert = (P_supervised(malicious) > t1)  OR  (anomaly_score > t2)

That OR rule is deliberate.  The two stages have complementary blind spots:
Stage 1 knows attack families it was taught and nothing else; Stage 2 knows
only "unlike normal" and cannot name what it found.  Requiring both to agree
(AND) would throw away exactly the zero-day coverage Stage 2 was added for.

The cost of OR is a higher false-positive rate than either stage alone, which
is why false-positive rate is a reported metric and not a footnote, and why
``weighted`` fusion is provided as a tunable middle ground.

Each alert carries a ``reason`` naming which stage fired, so an analyst can
tell "known Trickbot C2" from "nothing like normal traffic".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger
from .anomaly import AnomalyDetector
from .supervised import SupervisedDetector

log = get_logger("models.fusion")

REASON_NONE = "benign"
REASON_KNOWN = "known-attack"          # Stage 1 only
REASON_ANOMALY = "anomalous"           # Stage 2 only  <- the zero-day path
REASON_BOTH = "known-attack+anomalous"


@dataclass
class TwoStageDetector:
    """The full pipeline: preprocessing -> both stages -> fused verdict."""

    pipeline: Any                       # FeaturePipeline
    supervised: SupervisedDetector
    anomaly: AnomalyDetector
    rule: str = "or"
    supervised_threshold: float = 0.5
    weighted_alpha: float = 0.6
    weighted_threshold: float = 0.5

    # -- scoring -----------------------------------------------------------
    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a raw flow table -> per-flow verdicts and explanatory columns."""
        X = self.pipeline.transform(df)
        return self.score_matrix(X, index=df.index)

    def score_matrix(self, X: np.ndarray, index=None) -> pd.DataFrame:
        p_mal = self.supervised.predict_malicious_proba(X)
        family = self.supervised.predict_labels(X)
        anomaly_raw = self.anomaly.score(X)
        anomaly_norm = self.anomaly.normalised_score(X)

        stage1 = (p_mal > self.supervised_threshold).astype(int)
        stage2 = (anomaly_raw > self.anomaly.threshold_).astype(int)

        if self.rule == "or":
            alert = ((stage1 | stage2)).astype(int)
        elif self.rule == "and":
            alert = ((stage1 & stage2)).astype(int)
        elif self.rule == "weighted":
            a = self.weighted_alpha
            combined = a * p_mal + (1 - a) * _squash(anomaly_norm)
            alert = (combined > self.weighted_threshold).astype(int)
        else:
            raise ValueError(f"Unknown fusion rule: {self.rule!r}")

        reason = np.select(
            [(stage1 == 1) & (stage2 == 1), stage1 == 1, stage2 == 1],
            [REASON_BOTH, REASON_KNOWN, REASON_ANOMALY],
            default=REASON_NONE,
        )
        # Under AND/weighted the fused verdict can differ from the reason codes;
        # keep the reason honest by blanking it where no alert was raised.
        reason = np.where(alert == 1, reason, REASON_NONE)

        return pd.DataFrame({
            "alert": alert,
            "reason": reason,
            "stage1_fired": stage1,
            "stage2_fired": stage2,
            "p_malicious": p_mal,
            "predicted_family": family,
            "anomaly_score": anomaly_raw,
            "anomaly_sigma": anomaly_norm,
            "risk": np.clip(np.maximum(p_mal, _squash(anomaly_norm)), 0, 1),
        }, index=index)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.score_frame(df)["alert"].to_numpy()

    # -- persistence -------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.pipeline.save(directory / "feature_pipeline.joblib")
        self.supervised.save(directory / "stage1_supervised.joblib")
        self.anomaly.save(directory / "stage2_anomaly.joblib")
        joblib.dump({
            "rule": self.rule,
            "supervised_threshold": self.supervised_threshold,
            "weighted_alpha": self.weighted_alpha,
            "weighted_threshold": self.weighted_threshold,
        }, directory / "fusion.joblib")
        log.info("Two-stage detector saved to %s", directory)

    @staticmethod
    def load(directory: str | Path) -> "TwoStageDetector":
        from ..features.build_features import FeaturePipeline

        directory = Path(directory)
        meta = joblib.load(directory / "fusion.joblib")
        return TwoStageDetector(
            pipeline=FeaturePipeline.load(directory / "feature_pipeline.joblib"),
            supervised=SupervisedDetector.load(directory / "stage1_supervised.joblib"),
            anomaly=AnomalyDetector.load(directory / "stage2_anomaly.joblib"),
            **meta,
        )


def _squash(sigma: np.ndarray) -> np.ndarray:
    """Map an unbounded sigma-score onto (0, 1) so it can share a scale with a
    probability.  Logistic centred at 3 sigma."""
    return 1.0 / (1.0 + np.exp(-(np.asarray(sigma, dtype=float) - 3.0)))


def build_from_config(cfg, pipeline, supervised, anomaly) -> TwoStageDetector:
    return TwoStageDetector(
        pipeline=pipeline,
        supervised=supervised,
        anomaly=anomaly,
        rule=cfg.get_path("fusion.rule", "or"),
        supervised_threshold=cfg.get_path("fusion.supervised_threshold", 0.5),
        weighted_alpha=cfg.get_path("fusion.weighted_alpha", 0.6),
    )
