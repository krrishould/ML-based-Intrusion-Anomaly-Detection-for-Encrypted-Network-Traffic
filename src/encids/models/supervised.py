"""Stage 1 - supervised classifier for *known* attack / traffic families.

Random Forest or XGBoost over the flow + TLS feature matrix.  This stage is
strong on what it has seen and, by construction, blind to what it has not -
which is exactly why Stage 2 exists.

The classifier is multi-class (per attack family) and also exposes a calibrated
binary P(malicious) obtained by summing the probability mass over all malicious
classes, which is what the fusion layer thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..features import schema
from ..utils.logging_utils import get_logger

log = get_logger("models.supervised")


@dataclass
class SupervisedDetector:
    """Wraps the Stage-1 classifier plus its label encoding."""

    model_type: str = "xgboost"
    params: dict[str, Any] = field(default_factory=dict)
    class_weight: str | None = "balanced"
    model: Any = None
    label_encoder: LabelEncoder | None = None
    malicious_class_indices: list[int] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)

    # -- construction ------------------------------------------------------
    def _build(self, n_classes: int):
        if self.model_type == "random_forest":
            return RandomForestClassifier(
                random_state=42,
                class_weight="balanced_subsample" if self.class_weight else None,
                **self.params,
            )
        if self.model_type == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
                random_state=42,
                **self.params,
            )
        raise ValueError(f"Unknown model_type: {self.model_type!r}")

    # -- training ----------------------------------------------------------
    def fit(self, X: np.ndarray, labels, feature_names: list[str] | None = None,
            malicious_mask=None) -> "SupervisedDetector":
        """Train on the multi-class family label.

        ``malicious_mask`` is the per-row 0/1 malicious flag; it is used only to
        work out which encoded classes count as malicious, so P(malicious) can
        be derived from the multi-class probabilities.
        """
        labels = np.asarray(labels).astype(str)
        self.label_encoder = LabelEncoder().fit(labels)
        y = self.label_encoder.transform(labels)
        n_classes = len(self.label_encoder.classes_)
        self.feature_names = feature_names or []

        if malicious_mask is not None:
            mask = np.asarray(malicious_mask).astype(int)
            self.malicious_class_indices = sorted({
                int(cls) for cls, m in zip(y, mask) if m == 1
            })
        else:
            self.malicious_class_indices = [
                i for i, c in enumerate(self.label_encoder.classes_)
                if c != schema.BENIGN_LABEL
            ]

        self.model = self._build(n_classes)
        sample_weight = (compute_sample_weight("balanced", y)
                         if self.class_weight and self.model_type == "xgboost"
                         else None)

        log.info("Training %s on %d flows, %d classes%s", self.model_type,
                 len(y), n_classes, " (class-balanced)" if sample_weight is not None
                 or self.class_weight else "")
        if sample_weight is not None:
            self.model.fit(X, y, sample_weight=sample_weight)
        else:
            self.model.fit(X, y)
        return self

    # -- inference ---------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        idx = self.model.predict(X)
        return self.label_encoder.inverse_transform(np.asarray(idx).astype(int))

    def predict_malicious_proba(self, X: np.ndarray) -> np.ndarray:
        """P(flow is malicious) = sum of probability over malicious classes."""
        proba = self.predict_proba(X)
        if not self.malicious_class_indices:
            return np.zeros(len(X))
        cols = [i for i in self.malicious_class_indices if i < proba.shape[1]]
        return proba[:, cols].sum(axis=1)

    # -- interpretation ----------------------------------------------------
    def feature_importances(self) -> dict[str, float]:
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None or not self.feature_names:
            return {}
        pairs = zip(self.feature_names, [float(v) for v in importances])
        return dict(sorted(pairs, key=lambda kv: kv[1], reverse=True))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Stage-1 model -> %s", path)

    @staticmethod
    def load(path: str | Path) -> "SupervisedDetector":
        return joblib.load(path)


def build_from_config(cfg) -> SupervisedDetector:
    """Construct the Stage-1 detector described by ``config.yaml``."""
    model_type = cfg.get_path("supervised.model", "xgboost")
    params = dict(cfg.get_path(f"supervised.{model_type}", {}) or {})
    params = {k: v for k, v in params.items() if v is not None}
    return SupervisedDetector(
        model_type=model_type,
        params=params,
        class_weight=cfg.get_path("supervised.class_weight", "balanced"),
    )
