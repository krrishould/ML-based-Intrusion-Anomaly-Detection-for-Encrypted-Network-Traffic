"""Train the two-stage detector.

Stage 1 is trained on the labelled training split.  Stage 2 is trained on the
**benign rows of that same split only** - it must never see an attack, or its
whole purpose (catching families nobody labelled) is defeated.

Both stages, the fitted preprocessing pipeline and the SHAP background sample
are written to ``models/`` so that evaluation, the live pipeline and the
dashboard all load exactly the same artefacts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config, Paths, load_config, set_seed
from .data.build_dataset import load_feature_table
from .features import schema
from .features.build_features import FeaturePipeline, summarise
from .models import anomaly as anomaly_mod
from .models import fusion as fusion_mod
from .models import supervised as supervised_mod
from .models.explain import Explainer
from .utils.logging_utils import banner, get_logger, timed

log = get_logger("train")


@dataclass
class TrainingArtefacts:
    detector: Any
    explainer: Explainer
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: np.ndarray
    X_test: np.ndarray
    metadata: dict[str, Any]


def split_data(df: pd.DataFrame, cfg: Config
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the family label."""
    test_size = cfg.get_path("supervised.test_size", 0.2)
    seed = cfg.get_path("project.seed", 42)
    labels = df[schema.TARGET_COLUMN].astype(str)

    # Stratification needs at least two rows per class; fold the rare ones in.
    counts = labels.value_counts()
    stratify = labels if (cfg.get_path("supervised.stratify", True)
                          and counts.min() >= 2) else None
    if stratify is None:
        log.warning("Stratification disabled (a class has <2 samples)")

    return train_test_split(df, test_size=test_size, random_state=seed,
                            stratify=stratify)


def _subsample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Proportional stratified sub-sample, keeping every class represented.

    Plain proportional sampling would round the rarest families down to zero
    rows - CTU-13's Sogou botnet is 63 flows out of ~600k - and they would
    vanish from both training and evaluation without anything being reported.
    Each class therefore keeps a floor of two rows, which is also the minimum
    the stratified train/test split needs.
    """
    fraction = max_rows / len(df)
    rng = np.random.default_rng(seed)
    keep: list[pd.Index] = []
    for _, group in df.groupby(schema.TARGET_COLUMN, sort=False):
        n = min(len(group), max(2, int(round(len(group) * fraction))))
        chosen = rng.choice(len(group), size=n, replace=False)
        keep.append(group.index[chosen])
    return df.loc[np.concatenate(keep)].reset_index(drop=True)


def train(cfg: Config | None = None,
          use_tls: bool | None = None,
          hold_out_family: str | None = None,
          model_dir: str | Path | None = None,
          df: pd.DataFrame | None = None,
          source: str | None = None,
          max_rows: int | None = None) -> TrainingArtefacts:
    """Fit the full two-stage system.

    Parameters
    ----------
    use_tls
        Override ``features.use_tls_metadata``.  Setting it False produces the
        flow-statistics-only baseline that Objective 5 compares against.
    hold_out_family
        Remove this attack family from **Stage 1 training only**.  Stage 2 is
        unaffected (it never sees attacks anyway), so the held-out family
        becomes a genuine zero-day test at evaluation time.
    source
        Train on one dataset only (``synthetic``, ``ctu13``, ...).  Worth using
        whenever the sources differ in which features they can provide - a
        netflow corpus has no TLS handshake metadata at all, so a model blended
        across it and a TLS-bearing corpus produces an ablation that measures
        the source mix rather than the signal.
    max_rows
        Sub-sample the table before training, preserving the class balance.
    """
    cfg = cfg or load_config()
    set_seed(cfg.get_path("project.seed", 42))
    paths = Paths.from_config(cfg)
    model_dir = Path(model_dir) if model_dir else paths.models

    use_flow = cfg.get_path("features.use_flow_stats", True)
    use_tls = (cfg.get_path("features.use_tls_metadata", True)
               if use_tls is None else use_tls)

    banner(f"Training two-stage detector (flow={use_flow}, tls={use_tls})")
    df = load_feature_table(cfg) if df is None else df

    if source:
        before = len(df)
        df = df[df[schema.SOURCE_COLUMN].astype(str) == source]
        if df.empty:
            raise ValueError(f"No rows with source='{source}'")
        log.info("Source filter '%s': %d -> %d flows", source, before, len(df))

    if max_rows and len(df) > max_rows:
        df = _subsample(df, max_rows, cfg.get_path("project.seed", 42))
        log.info("Sub-sampled to %d flows (class balance preserved)", len(df))

    train_df, test_df = split_data(df, cfg)
    log.info("Split: %d train / %d test flows", len(train_df), len(test_df))

    # ---- preprocessing (fitted on train only - never on test) ------------
    pipeline = FeaturePipeline(use_flow=use_flow, use_tls=use_tls)
    with timed("Fitting the feature pipeline", log):
        # Recompute ja3_rarity from the TRAINING split alone. The assembled
        # table carries a rarity column derived from the whole corpus, which
        # would let held-out rows influence a feature used to score them.
        pipeline.record_fingerprints(train_df)
        train_df = pipeline.apply_fingerprints(
            train_df, n_buckets=cfg.get_path("features.ja3_hash_buckets", 256))
        test_df = pipeline.apply_fingerprints(
            test_df, n_buckets=cfg.get_path("features.ja3_hash_buckets", 256))
        X_train = pipeline.fit_transform(train_df)
    X_test = pipeline.transform(test_df)

    # ---- Stage 1 ---------------------------------------------------------
    stage1_df, stage1_X = train_df, X_train
    if hold_out_family:
        mask = (train_df[schema.TARGET_COLUMN].astype(str) != hold_out_family).to_numpy()
        stage1_df, stage1_X = train_df[mask], X_train[mask]
        log.info("Zero-day setup: '%s' withheld from Stage 1 (%d flows removed)",
                 hold_out_family, int((~mask).sum()))

    supervised = supervised_mod.build_from_config(cfg)
    with timed("Training Stage 1 (supervised)", log):
        supervised.fit(
            stage1_X,
            stage1_df[schema.TARGET_COLUMN].astype(str),
            feature_names=pipeline.columns,
            malicious_mask=stage1_df[schema.BINARY_TARGET_COLUMN],
        )

    # ---- Stage 2: benign traffic only ------------------------------------
    benign_mask = (train_df[schema.BINARY_TARGET_COLUMN] == 0).to_numpy()
    X_benign = X_train[benign_mask]
    if len(X_benign) < 50:
        raise ValueError(f"Only {len(X_benign)} benign training flows - Stage 2 "
                         "needs a substantial benign sample")
    log.info("Stage 2 trains on %d benign flows (0 attacks, by design)",
             len(X_benign))

    detector_stage2 = anomaly_mod.build_from_config(cfg)
    with timed("Training Stage 2 (unsupervised)", log):
        detector_stage2.fit(X_benign, feature_names=pipeline.columns)

    # ---- fusion ----------------------------------------------------------
    detector = fusion_mod.build_from_config(cfg, pipeline, supervised,
                                            detector_stage2)

    # ---- explainability --------------------------------------------------
    explainer = Explainer(
        detector=detector,
        top_k=cfg.get_path("explain.top_k_features", 6),
        background_samples=cfg.get_path("explain.background_samples", 200),
    )
    if cfg.get_path("explain.enabled", True):
        with timed("Preparing SHAP explainer", log):
            explainer.fit(X_benign)

    # ---- persist ---------------------------------------------------------
    detector.save(model_dir)
    joblib.dump(explainer, model_dir / "explainer.joblib")

    metadata = {
        "use_flow_stats": use_flow,
        "use_tls_metadata": use_tls,
        "held_out_family": hold_out_family,
        "source_filter": source,
        "n_features": len(pipeline.columns),
        "feature_names": pipeline.columns,
        "stage1_model": supervised.model_type,
        "stage2_model": detector_stage2.model_type,
        "stage2_threshold": detector_stage2.threshold_,
        "fusion_rule": detector.rule,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_benign_train": int(len(X_benign)),
        "classes": list(map(str, supervised.label_encoder.classes_)),
        "dataset": summarise(df),
    }
    (model_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    log.info("Training metadata -> %s", model_dir / "training_metadata.json")

    # Keep the exact split so evaluation never accidentally scores on train.
    test_df.to_parquet(paths.processed / "test_split.parquet", index=False)

    return TrainingArtefacts(detector, explainer, train_df, test_df,
                             X_train, X_test, metadata)


def load_artefacts(cfg: Config | None = None, model_dir: str | Path | None = None):
    """Load a trained detector + explainer from disk."""
    cfg = cfg or load_config()
    model_dir = Path(model_dir) if model_dir else Paths.from_config(cfg).models
    detector = fusion_mod.TwoStageDetector.load(model_dir)
    explainer_path = model_dir / "explainer.joblib"
    explainer = joblib.load(explainer_path) if explainer_path.exists() else None
    if explainer is not None:
        # The pickle deliberately omits the detector and the SHAP explainer;
        # both are reattached here from the freshly loaded models.
        explainer.rebind(detector)
    return detector, explainer
