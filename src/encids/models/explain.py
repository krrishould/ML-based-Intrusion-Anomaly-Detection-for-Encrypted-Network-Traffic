"""SHAP-based explanations for flagged flows.

Objective 4 of the proposal: an alert must come with a human-readable reason.
"Flow flagged, risk 0.94" is not actionable; "flagged because the inter-arrival
time is metronomic (60.0s +/- 0.3s) and the JA3 fingerprint is rare" is.

Two explanation paths, matching the two stages:

  * Stage 1 (tree model)  -> exact SHAP values via ``shap.TreeExplainer``.
  * Stage 2 (anomaly)     -> per-feature deviation from the benign profile,
    which for a reconstruction-based detector is literally *what the model got
    wrong about this flow*, and for IsolationForest is a z-score against the
    benign training distribution.  This is a cheaper surrogate than KernelSHAP,
    and unlike KernelSHAP it is fast enough to run inside the live pipeline.

Every explanation is rendered to a short English sentence for the dashboard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger("models.explain")

# Plain-English names, so an operator does not have to read snake_case.
FEATURE_GLOSSARY: dict[str, str] = {
    "duration_ms": "flow duration",
    "total_packets": "packet count",
    "total_bytes": "bytes transferred",
    "fwd_bytes": "bytes sent by client",
    "bwd_bytes": "bytes sent by server",
    "flow_iat_mean": "average gap between packets",
    "flow_iat_std": "regularity of packet timing",
    "fwd_iat_mean": "average client send interval",
    "bwd_iat_mean": "average server send interval",
    "pkt_len_mean": "average packet size",
    "pkt_len_std": "packet size variability",
    "fwd_pkt_len_mean": "average client packet size",
    "bwd_pkt_len_mean": "average server packet size",
    "down_up_byte_ratio": "download/upload byte ratio",
    "down_up_packet_ratio": "download/upload packet ratio",
    "fwd_bytes_fraction": "share of traffic sent by the client",
    "flow_bytes_per_s": "throughput",
    "flow_packets_per_s": "packet rate",
    "avg_packet_size": "average packet size",
    "syn_count": "SYN flag count",
    "rst_count": "RST flag count",
    "fin_count": "FIN flag count",
    "psh_count": "PSH flag count",
    "ja3_bucket": "client TLS fingerprint (JA3)",
    "ja3s_bucket": "server TLS fingerprint (JA3S)",
    "ja3_rarity": "rarity of the TLS fingerprint",
    "ja3_is_known_browser": "TLS fingerprint matches a mainstream browser",
    "n_cipher_suites": "number of offered cipher suites",
    "n_extensions": "number of TLS extensions",
    "has_grease": "GREASE values present in the handshake",
    "sni_entropy": "randomness of the requested hostname",
    "sni_length": "hostname length",
    "sni_digit_ratio": "digits in the hostname",
    "tls_version": "TLS version",
    "cert_chain_len": "certificate chain length",
    "handshake_duration_ms": "TLS handshake duration",
    "is_tls": "TLS in use",
    "alpn_is_h2": "HTTP/2 negotiated",
}


def pretty(name: str) -> str:
    return FEATURE_GLOSSARY.get(name, name.replace("_", " "))


@dataclass
class Explainer:
    """Produces per-flow, per-feature attributions for both stages."""

    detector: Any                      # TwoStageDetector
    top_k: int = 6
    background_samples: int = 200
    _tree_explainer: Any = None
    benign_mean_: np.ndarray | None = None
    benign_std_: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    # -- setup -------------------------------------------------------------
    def fit(self, X_background: np.ndarray) -> "Explainer":
        """Prepare the explainers from a background sample of benign flows."""
        import shap

        self.feature_names = list(self.detector.pipeline.columns)
        n = min(self.background_samples, len(X_background))
        background = X_background[
            np.random.default_rng(42).choice(len(X_background), n, replace=False)
        ]

        try:
            self._tree_explainer = shap.TreeExplainer(self.detector.supervised.model)
            log.info("SHAP TreeExplainer ready (%d background flows)", n)
        except Exception as exc:
            log.warning("TreeExplainer unavailable (%s); Stage-1 explanations "
                        "will fall back to global feature importance", exc)
            self._tree_explainer = None

        self.benign_mean_ = background.mean(axis=0)
        self.benign_std_ = np.where(background.std(axis=0) == 0, 1.0,
                                    background.std(axis=0))
        return self

    # -- persistence -------------------------------------------------------
    # The explainer holds a reference to the detector, and shap.TreeExplainer
    # holds its own copy of the Stage-1 model. Pickling those alongside the
    # detector files duplicated the whole model twice and produced a 231 MB
    # artefact for what is really two small arrays. Both are dropped on save
    # and rebuilt by `rebind` on load.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["detector"] = None
        state["_tree_explainer"] = None
        return state

    def rebind(self, detector: Any) -> "Explainer":
        """Reattach a loaded detector and rebuild the SHAP explainer."""
        self.detector = detector
        if detector is None:
            return self
        self.feature_names = list(detector.pipeline.columns)
        try:
            import shap

            self._tree_explainer = shap.TreeExplainer(detector.supervised.model)
        except Exception as exc:
            log.warning("Could not rebuild TreeExplainer (%s); explanations "
                        "will use benign-profile deviation only", exc)
            self._tree_explainer = None
        return self

    # -- Stage 1 -----------------------------------------------------------
    def shap_values(self, X: np.ndarray) -> np.ndarray | None:
        """SHAP values for the malicious outcome, shape (n_samples, n_features)."""
        if self._tree_explainer is None:
            return None
        values = self._tree_explainer.shap_values(X, check_additivity=False)
        values = np.asarray(values)
        if values.ndim == 3:
            # (samples, features, classes) - aggregate the malicious classes.
            idx = [i for i in self.detector.supervised.malicious_class_indices
                   if i < values.shape[2]]
            values = values[:, :, idx].sum(axis=2) if idx else values[:, :, 0]
        return values

    # -- Stage 2 -----------------------------------------------------------
    def deviation_scores(self, X: np.ndarray) -> np.ndarray:
        """Per-feature deviation from the benign profile, in sigmas."""
        return (X - self.benign_mean_) / self.benign_std_

    # -- combined ----------------------------------------------------------
    def explain(self, X: np.ndarray, verdicts: pd.DataFrame,
                raw: pd.DataFrame | None = None) -> list[dict[str, Any]]:
        """Explain each row; returns one dict per flow."""
        shap_vals = self.shap_values(X)
        devs = self.deviation_scores(X)
        names = self.feature_names
        out: list[dict[str, Any]] = []

        for i in range(len(X)):
            row = verdicts.iloc[i]
            stage1 = bool(row.get("stage1_fired", 0))
            # Attribute using SHAP when Stage 1 fired and SHAP is available;
            # otherwise fall back to benign-profile deviation.
            if stage1 and shap_vals is not None:
                weights = shap_vals[i]
                basis = "shap"
            else:
                weights = devs[i]
                basis = "deviation"

            order = np.argsort(-np.abs(weights))[: self.top_k]
            contributions = [
                {
                    "feature": names[j],
                    "label": pretty(names[j]),
                    "weight": float(weights[j]),
                    "deviation_sigma": float(devs[i, j]),
                    "direction": "high" if devs[i, j] > 0 else "low",
                    "value": (float(raw.iloc[i][names[j]])
                              if raw is not None and names[j] in raw.columns
                              and pd.notna(raw.iloc[i][names[j]]) else None),
                }
                for j in order
            ]
            out.append({
                "basis": basis,
                "reason": row.get("reason", ""),
                "contributions": contributions,
                "summary": _sentence(row, contributions),
            })
        return out


def _sentence(row: pd.Series, contributions: list[dict[str, Any]]) -> str:
    """Render one alert as a sentence an analyst can act on."""
    reason = str(row.get("reason", ""))
    if reason in ("", "benign"):
        return "No alert: consistent with normal encrypted traffic."

    parts = []
    for c in contributions[:3]:
        val = c["value"]
        shown = f" ({val:,.1f})" if isinstance(val, float) else ""
        parts.append(f"{'unusually high' if c['direction'] == 'high' else 'unusually low'} "
                     f"{c['label']}{shown}")
    drivers = "; ".join(parts) if parts else "no dominant single feature"

    if reason == "known-attack":
        head = (f"Matches the known family "
                f"'{row.get('predicted_family', 'unknown')}' "
                f"(p={float(row.get('p_malicious', 0)):.2f})")
    elif reason == "anomalous":
        head = (f"Does not match any known traffic profile "
                f"({float(row.get('anomaly_sigma', 0)):.1f} sigma from normal)"
                f" - possible unseen threat")
    else:
        head = (f"Matches '{row.get('predicted_family', 'unknown')}' "
                f"and is {float(row.get('anomaly_sigma', 0)):.1f} sigma "
                f"from normal")

    return f"{head}. Driven by: {drivers}."


# ---------------------------------------------------------------------------
def global_importance_plot(detector, out_path: str | Path, top_n: int = 20) -> None:
    """Bar chart of the Stage-1 global feature importances."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    importances = detector.supervised.feature_importances()
    if not importances:
        log.warning("No feature importances available; skipping plot")
        return
    items = list(importances.items())[:top_n][::-1]
    labels = [pretty(k) for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 0.38 * len(items) + 1.6))
    ax.barh(labels, values, color="#3b6ea5")
    ax.set_xlabel("Gain-based importance")
    ax.set_title(f"Stage 1 - top {len(items)} features")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Feature-importance plot -> %s", out_path)


def shap_summary_plot(explainer: Explainer, X: np.ndarray, out_path: str | Path,
                      max_display: int = 20) -> None:
    """Standard SHAP beeswarm over a sample of flows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    values = explainer.shap_values(X)
    if values is None:
        log.warning("No SHAP values available; skipping beeswarm plot")
        return
    labels = [pretty(n) for n in explainer.feature_names]
    plt.figure()
    shap.summary_plot(values, X, feature_names=labels, max_display=max_display,
                      show=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log.info("SHAP summary plot -> %s", out_path)
