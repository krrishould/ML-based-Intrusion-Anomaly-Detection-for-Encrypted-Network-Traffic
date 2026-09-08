"""Stage 2 - unsupervised anomaly detector for *unseen* (zero-day) threats.

Trained on **benign traffic only**.  It never sees an attack during training,
so it cannot be over-fitted to known attack families - it learns the shape of
normal and flags departures from it.  That is what gives the system a chance
against threats that were not in any training set.

Two interchangeable detectors:

  * ``IsolationForest`` - fast, no tuning, works well on tabular flow features.
  * ``Autoencoder``     - a small PyTorch MLP; the reconstruction error is the
    anomaly score.  Slower, but tends to capture feature *interactions* that a
    tree-based isolation score misses.

Both expose the same interface: ``fit(X_benign)``, ``score(X)`` (higher = more
anomalous), and ``predict(X)`` against a threshold calibrated as a percentile
of the benign training scores.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from ..utils.logging_utils import get_logger

log = get_logger("models.anomaly")


# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------
class _AutoencoderNet:
    """Small symmetric MLP autoencoder (lazily imports torch)."""

    def __init__(self, n_features: int, hidden_dims: list[int]):
        import torch
        import torch.nn as nn

        dims = [n_features] + list(hidden_dims)
        encoder_layers: list[Any] = []
        for a, b in zip(dims, dims[1:]):
            encoder_layers += [nn.Linear(a, b), nn.ReLU()]
        decoder_layers: list[Any] = []
        rev = dims[::-1]
        for i, (a, b) in enumerate(zip(rev, rev[1:])):
            decoder_layers.append(nn.Linear(a, b))
            if i < len(rev) - 2:                 # no activation on the output
                decoder_layers.append(nn.ReLU())

        self.torch = torch
        self.net = nn.Sequential(*encoder_layers, *decoder_layers)

    def __call__(self, x):
        return self.net(x)


@dataclass
class AutoencoderDetector:
    """Reconstruction-error anomaly detector."""

    hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])
    epochs: int = 60
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    early_stopping_patience: int = 8
    _net: Any = None
    _n_features: int = 0

    def fit(self, X: np.ndarray, val_fraction: float = 0.1) -> "AutoencoderDetector":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(X, dtype=np.float32)
        self._n_features = X.shape[1]
        n_val = max(1, int(len(X) * val_fraction))
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(X))
        val_X, train_X = X[idx[:n_val]], X[idx[n_val:]]

        wrapper = _AutoencoderNet(self._n_features, self.hidden_dims)
        net = wrapper.net
        optimiser = torch.optim.Adam(net.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()
        loader = DataLoader(TensorDataset(torch.from_numpy(train_X)),
                            batch_size=self.batch_size, shuffle=True)
        val_tensor = torch.from_numpy(val_X)

        best_loss, best_state, patience = float("inf"), None, 0
        log.info("Training autoencoder: %d benign flows, dims %s",
                 len(train_X), [self._n_features] + self.hidden_dims)
        for epoch in range(self.epochs):
            net.train()
            for (batch,) in loader:
                optimiser.zero_grad()
                loss = criterion(net(batch), batch)
                loss.backward()
                optimiser.step()

            net.eval()
            with torch.no_grad():
                val_loss = float(criterion(net(val_tensor), val_tensor))

            if val_loss < best_loss - 1e-6:
                best_loss, patience = val_loss, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                patience += 1
                if patience >= self.early_stopping_patience:
                    log.info("Early stop at epoch %d (val MSE %.6f)", epoch + 1,
                             best_loss)
                    break
            if (epoch + 1) % 10 == 0:
                log.info("  epoch %3d | val MSE %.6f", epoch + 1, val_loss)

        if best_state is not None:
            net.load_state_dict(best_state)
        self._net = net
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-flow reconstruction error (higher = more anomalous)."""
        import torch

        X = np.asarray(X, dtype=np.float32)
        self._net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X)
            recon = self._net(tensor)
            return ((recon - tensor) ** 2).mean(dim=1).numpy()


# ---------------------------------------------------------------------------
# Unified Stage-2 wrapper
# ---------------------------------------------------------------------------
@dataclass
class AnomalyDetector:
    """Stage-2 detector: fitted on benign traffic, thresholded by percentile."""

    model_type: str = "isolation_forest"
    params: dict[str, Any] = field(default_factory=dict)
    threshold_percentile: float = 99.0
    exclude_features: list[str] = field(default_factory=list)
    model: Any = None
    threshold_: float = 0.0
    train_score_mean_: float = 0.0
    train_score_std_: float = 1.0
    feature_names: list[str] = field(default_factory=list)
    _keep_idx: list[int] = field(default_factory=list)

    def _select(self, X: np.ndarray) -> np.ndarray:
        """Stage 2's own view of the feature matrix.

        Hashing-trick buckets (``ja3_bucket``) are ordinal in name only - bucket
        200 is not "larger" than bucket 3 - so a distance/reconstruction-based
        detector treats their arbitrary numbering as real structure.  Trees in
        Stage 1 can carve them into arbitrary subsets and are unharmed, so the
        exclusion applies to this stage only.
        """
        if not self._keep_idx:
            return X
        return X[:, self._keep_idx]

    def fit(self, X_benign: np.ndarray,
            feature_names: list[str] | None = None) -> "AnomalyDetector":
        self.feature_names = feature_names or []
        if self.exclude_features and self.feature_names:
            self._keep_idx = [i for i, name in enumerate(self.feature_names)
                              if name not in set(self.exclude_features)]
            dropped = len(self.feature_names) - len(self._keep_idx)
            if dropped:
                log.info("Stage 2 excludes %d non-ordinal feature(s): %s", dropped,
                         ", ".join(n for n in self.feature_names
                                   if n in set(self.exclude_features)))
        X_fit = self._select(X_benign)

        if self.model_type == "isolation_forest":
            self.model = IsolationForest(random_state=42, **self.params)
            log.info("Training IsolationForest on %d benign flows", len(X_fit))
            self.model.fit(X_fit)
        elif self.model_type == "autoencoder":
            self.model = AutoencoderDetector(**self.params).fit(X_fit)
        else:
            raise ValueError(f"Unknown anomaly model: {self.model_type!r}")

        train_scores = self.score(X_benign)
        self.threshold_ = float(np.percentile(train_scores,
                                              self.threshold_percentile))
        self.train_score_mean_ = float(train_scores.mean())
        self.train_score_std_ = float(train_scores.std() or 1.0)
        log.info("Stage-2 threshold = %.6f (p%.1f of benign scores)",
                 self.threshold_, self.threshold_percentile)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score, oriented so that higher always means more anomalous."""
        X = self._select(X)
        if self.model_type == "isolation_forest":
            # sklearn's score_samples is higher-is-more-normal; negate it.
            return -self.model.score_samples(X)
        return self.model.score(X)

    def normalised_score(self, X: np.ndarray) -> np.ndarray:
        """Score expressed in benign standard deviations - comparable across
        detector types, which is what the weighted fusion rule needs."""
        return (self.score(X) - self.train_score_mean_) / self.train_score_std_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomalous, 0 = normal."""
        return (self.score(X) > self.threshold_).astype(int)

    def recalibrate(self, benign_scores: np.ndarray, percentile: float | None = None
                    ) -> float:
        """Re-derive the threshold from a fresh benign sample.

        Used to adapt to a new deployment environment without retraining: the
        definition of "normal" on a campus network is not the definition of
        normal in a capture lab.
        """
        pct = percentile if percentile is not None else self.threshold_percentile
        self.threshold_ = float(np.percentile(benign_scores, pct))
        return self.threshold_

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info("Stage-2 model -> %s", path)

    @staticmethod
    def load(path: str | Path) -> "AnomalyDetector":
        return joblib.load(path)


def build_from_config(cfg) -> AnomalyDetector:
    """Construct the Stage-2 detector described by ``config.yaml``."""
    model_type = cfg.get_path("anomaly.model", "isolation_forest")
    params = dict(cfg.get_path(f"anomaly.{model_type}", {}) or {})
    if model_type == "isolation_forest":
        params.setdefault("contamination", cfg.get_path("anomaly.contamination", 0.02))
    return AnomalyDetector(
        model_type=model_type,
        params=params,
        threshold_percentile=cfg.get_path("anomaly.threshold_percentile", 99.0),
        exclude_features=list(cfg.get_path("anomaly.exclude_features", []) or []),
    )
