"""Evaluation - the numbers that go in the report.

Produces, for the held-out test split:

  1. Binary detection metrics for Stage 1 alone, Stage 2 alone, and the fused
     system (accuracy, precision, recall, F1, FPR, ROC-AUC, PR-AUC).
  2. Multi-class family classification metrics for Stage 1.
  3. Per-family detection rates - so a strong headline number cannot hide a
     family the system never catches.
  4. **Ablation**: flow-statistics-only vs. flow + TLS metadata.  This is the
     experiment that decides whether the dual-signal claim holds.
  5. **Zero-day experiment**: retrain Stage 1 without one attack family and
     measure how much of it Stage 2 recovers on its own.
  6. Scoring latency, for the deployability argument.

Everything is written to ``reports/metrics/`` as JSON + CSV and to
``reports/figures/`` as PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import Config, Paths, load_config
from .data.build_dataset import load_feature_table
from .features import schema
from .models import metrics as M
from .models.explain import global_importance_plot, shap_summary_plot
from .train import train
from .utils.logging_utils import banner, get_logger, timed

log = get_logger("evaluate")


# ---------------------------------------------------------------------------
def evaluate_detector(detector, test_df: pd.DataFrame) -> dict[str, Any]:
    """Score the test split and compute every headline metric."""
    y_true = test_df[schema.BINARY_TARGET_COLUMN].astype(int).to_numpy()
    families = test_df[schema.TARGET_COLUMN].astype(str).to_numpy()
    verdicts = detector.score_frame(test_df)

    results: dict[str, Any] = {
        "fused": M.binary_metrics(y_true, verdicts["alert"], verdicts["risk"]),
        "stage1_only": M.binary_metrics(y_true, verdicts["stage1_fired"],
                                        verdicts["p_malicious"]),
        "stage2_only": M.binary_metrics(y_true, verdicts["stage2_fired"],
                                        verdicts["anomaly_score"]),
        "family_classification": M.multiclass_metrics(
            families, verdicts["predicted_family"]),
        "per_family_detection_rate": M.per_class_detection_rate(
            families, verdicts["alert"]),
        "alert_reason_counts": {str(k): int(v) for k, v in
                                verdicts["reason"].value_counts().items()},
    }
    # How much of the malicious traffic did Stage 2 catch that Stage 1 missed?
    mal = y_true == 1
    if mal.any():
        s1 = verdicts["stage1_fired"].to_numpy()[mal]
        s2 = verdicts["stage2_fired"].to_numpy()[mal]
        results["stage2_unique_contribution"] = float(((s2 == 1) & (s1 == 0)).mean())

    # Stage-2 operating points: the threshold is a deployment choice, so the
    # whole FPR/recall trade-off curve is reported, not just the configured one.
    benign_scores = verdicts.loc[y_true == 0, "anomaly_score"]
    results["stage2_operating_points"] = M.threshold_sweep(
        y_true, verdicts["anomaly_score"], benign_scores).to_dict(orient="records")

    # Performance on deliberately evasive traffic, where it is available.
    if "evasion_level" in test_df.columns:
        table = M.stratify_by_evasion(y_true, verdicts["alert"],
                                      test_df["evasion_level"])
        if not table.empty:
            results["detection_by_evasion_level"] = table.to_dict(orient="records")

    # Per-source breakdown.  Blending a real netflow corpus (no TLS metadata)
    # with a synthetic one (full TLS metadata) makes a single aggregate score
    # uninterpretable, so each source is scored separately as well.
    if schema.SOURCE_COLUMN in test_df.columns:
        per_source = {}
        for source, group in test_df.groupby(schema.SOURCE_COLUMN):
            if len(group) < 50:
                continue
            rows = verdicts.loc[group.index]
            y = group[schema.BINARY_TARGET_COLUMN].astype(int).to_numpy()
            if len(set(y)) < 2:
                continue
            per_source[str(source)] = {
                "n_flows": int(len(group)),
                "malicious_rate": float(y.mean()),
                **M.binary_metrics(y, rows["alert"], rows["risk"]),
            }
        if len(per_source) > 1:
            results["per_source"] = per_source
    return results


# ---------------------------------------------------------------------------
def run_ablation(cfg: Config, df: pd.DataFrame, paths: Paths,
                 source: str | None = None,
                 max_rows: int | None = None) -> dict[str, Any]:
    """Flow-only baseline vs. flow + TLS metadata (Objective 5)."""
    banner("Ablation: flow statistics only vs. flow + TLS metadata")

    with timed("Baseline (flow statistics only)", log):
        baseline = train(cfg, use_tls=False, df=df, source=source,
                         max_rows=max_rows,
                         model_dir=paths.models / "baseline_flow_only")
    baseline_results = evaluate_detector(baseline.detector, baseline.test_df)

    with timed("Dual-signal (flow + TLS metadata)", log):
        full = train(cfg, use_tls=True, df=df, source=source,
                     max_rows=max_rows, model_dir=paths.models)
    full_results = evaluate_detector(full.detector, full.test_df)

    # A source with no attacks (ISCX VPN-nonVPN2016 is traffic-type only) has a
    # degenerate binary target - every row is class 0 - so precision/recall/AUC
    # are undefined and comparing them would print a table of zeros. There the
    # meaningful question is whether TLS metadata improves the *traffic-family*
    # classification, so the ablation switches to the multi-class metrics.
    has_both_classes = (
        full.test_df[schema.BINARY_TARGET_COLUMN].astype(int).nunique() > 1)

    if has_both_classes:
        table = M.compare(baseline_results["fused"], full_results["fused"])
        basis = "binary detection (fused)"
    else:
        log.info("Single-class target on this source - comparing traffic-family "
                 "classification instead of binary detection")
        table = M.compare(
            baseline_results["family_classification"],
            full_results["family_classification"],
            keys=("accuracy", "macro_f1", "weighted_f1"),
        )
        basis = "traffic-family classification (Stage 1)"

    log.info("Ablation basis: %s\n%s", basis, table.to_string(index=False))
    table.to_csv(paths.metrics / "ablation_flow_vs_flow_tls.csv", index=False)

    return {
        "flow_only": baseline_results,
        "flow_plus_tls": full_results,
        "ablation_basis": basis,
        "comparison": table.to_dict(orient="records"),
        "_artefacts": full,
    }


# ---------------------------------------------------------------------------
def run_zero_day(cfg: Config, df: pd.DataFrame, paths: Paths, family: str,
                 source: str | None = None,
                 max_rows: int | None = None) -> dict[str, Any]:
    """Withhold one attack family from Stage 1 and see what Stage 2 recovers."""
    banner(f"Zero-day experiment: '{family}' withheld from Stage 1")

    artefacts = train(cfg, hold_out_family=family, df=df, source=source,
                      max_rows=max_rows,
                      model_dir=paths.models / f"zeroday_{family}")
    test_df = artefacts.test_df
    verdicts = artefacts.detector.score_frame(test_df)
    families = test_df[schema.TARGET_COLUMN].astype(str).to_numpy()

    report = M.zero_day_report(families, verdicts["alert"],
                               verdicts["stage1_fired"], verdicts["stage2_fired"],
                               held_out=family)
    if report.get("n_flows"):
        log.info("'%s': %d flows | overall %.1f%% detected | Stage 2 alone "
                 "caught %.1f%% that Stage 1 missed", family, report["n_flows"],
                 100 * report["overall_detection_rate"],
                 100 * report["stage2_only_rate"])
    else:
        log.warning("Family '%s' absent from the test split", family)
    return report


# ---------------------------------------------------------------------------
def make_figures(detector, explainer, test_df: pd.DataFrame, verdicts: pd.DataFrame,
                 paths: Paths) -> None:
    """Confusion matrix, ROC/PR curves, score distributions, SHAP plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                                 RocCurveDisplay)

    figs = paths.figures
    figs.mkdir(parents=True, exist_ok=True)
    y_true = test_df[schema.BINARY_TARGET_COLUMN].astype(int).to_numpy()

    # --- confusion matrix ---
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, verdicts["alert"], display_labels=["benign", "malicious"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Fused two-stage detector")
    fig.tight_layout(); fig.savefig(figs / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # --- ROC & PR ---
    if len(set(y_true)) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        RocCurveDisplay.from_predictions(y_true, verdicts["risk"], ax=axes[0],
                                         name="fused")
        RocCurveDisplay.from_predictions(y_true, verdicts["p_malicious"],
                                         ax=axes[0], name="Stage 1")
        RocCurveDisplay.from_predictions(y_true, verdicts["anomaly_score"],
                                         ax=axes[0], name="Stage 2")
        axes[0].set_title("ROC"); axes[0].grid(alpha=0.3)
        PrecisionRecallDisplay.from_predictions(y_true, verdicts["risk"],
                                                ax=axes[1], name="fused")
        axes[1].set_title("Precision-Recall"); axes[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(figs / "roc_pr_curves.png", dpi=150)
        plt.close(fig)

    # --- anomaly score distribution, benign vs malicious ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    scores = verdicts["anomaly_score"].to_numpy()
    bins = np.linspace(np.percentile(scores, 0.5), np.percentile(scores, 99.5), 60)
    ax.hist(scores[y_true == 0], bins=bins, alpha=0.65, label="benign",
            color="#4c9f70", density=True)
    ax.hist(scores[y_true == 1], bins=bins, alpha=0.65, label="malicious",
            color="#c1442e", density=True)
    ax.axvline(detector.anomaly.threshold_, color="k", ls="--",
               label=f"threshold (p{detector.anomaly.threshold_percentile:g})")
    ax.set_xlabel("Stage-2 anomaly score"); ax.set_ylabel("density")
    ax.set_title("Stage 2 separates benign from malicious without ever seeing an attack")
    ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(figs / "anomaly_score_distribution.png", dpi=150)
    plt.close(fig)

    # --- per-family detection rate ---
    rates = M.per_class_detection_rate(
        test_df[schema.TARGET_COLUMN].astype(str), verdicts["alert"])
    items = sorted(rates.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(items) + 1.8))
    colours = ["#c1442e" if v < 0.8 else "#4c9f70" for _, v in items]
    ax.barh([k for k, _ in items], [v for _, v in items], color=colours)
    ax.set_xlim(0, 1); ax.set_xlabel("detection rate (alert rate)")
    ax.set_title("Per-family detection rate (benign rows = false-positive rate)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(figs / "per_family_detection.png", dpi=150)
    plt.close(fig)

    # --- SHAP / importance ---
    global_importance_plot(detector, figs / "stage1_feature_importance.png")
    if explainer is not None:
        sample = detector.pipeline.transform(test_df.iloc[:400])
        try:
            shap_summary_plot(explainer, sample, figs / "shap_summary.png")
        except Exception as exc:
            log.warning("SHAP beeswarm failed (%s) - skipping", exc)

    log.info("Figures written to %s", figs)


# ---------------------------------------------------------------------------
def main(cfg: Config | None = None, skip_ablation: bool = False,
         zero_day_family: str = "c2_beacon", source: str | None = None,
         max_rows: int | None = None) -> dict[str, Any]:
    cfg = cfg or load_config()
    paths = Paths.from_config(cfg)
    paths.metrics.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(cfg)
    results: dict[str, Any] = {"source_filter": source}

    if skip_ablation:
        artefacts = train(cfg, df=df, source=source, max_rows=max_rows)
        results["flow_plus_tls"] = evaluate_detector(artefacts.detector,
                                                     artefacts.test_df)
    else:
        ablation = run_ablation(cfg, df, paths, source=source, max_rows=max_rows)
        artefacts = ablation.pop("_artefacts")
        results.update(ablation)

    detector, explainer = artefacts.detector, artefacts.explainer
    test_df = artefacts.test_df
    verdicts = detector.score_frame(test_df)

    banner("Latency")
    results["latency"] = M.measure_latency(detector, test_df)
    for batch, stats in results["latency"].items():
        log.info("%-10s %7.3f ms/flow  (%s flows/s)", batch,
                 stats["per_flow_ms"], f"{stats['flows_per_second']:,.0f}")

    if zero_day_family:
        results["zero_day"] = run_zero_day(cfg, df, paths, zero_day_family,
                                           source=source, max_rows=max_rows)

    with timed("Generating figures", log):
        make_figures(detector, explainer, test_df, verdicts, paths)

    # --- worked example explanations, for the report and the viva ---------
    alerts = verdicts[verdicts["alert"] == 1].head(5)
    if not alerts.empty and explainer is not None:
        X = detector.pipeline.transform(test_df.loc[alerts.index])
        explanations = explainer.explain(X, alerts, raw=test_df.loc[alerts.index])
        results["example_explanations"] = [
            {"true_label": str(test_df.loc[idx, schema.TARGET_COLUMN]),
             "summary": exp["summary"],
             "top_features": exp["contributions"][:4]}
            for idx, exp in zip(alerts.index, explanations)
        ]
        banner("Example explanations")
        for item in results["example_explanations"]:
            log.info("[%s] %s", item["true_label"], item["summary"])

    suffix = f"_{source}" if source else ""
    out = paths.metrics / f"evaluation{suffix}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    log.info("Evaluation results -> %s", out)

    _print_headline(results)
    return results


def _print_headline(results: dict[str, Any]) -> None:
    banner("Headline results")
    key = "flow_plus_tls" if "flow_plus_tls" in results else "fused"
    block = results.get(key, results)
    fused = block.get("fused", block)
    for name in ("accuracy", "precision", "recall", "f1",
                 "false_positive_rate", "roc_auc"):
        if name in fused:
            log.info("%-22s %.4f", name, fused[name])
    if "zero_day" in results and results["zero_day"].get("n_flows"):
        z = results["zero_day"]
        log.info("zero-day (%s)      %.1f%% detected, %.1f%% by Stage 2 alone",
                 z["held_out"], 100 * z["overall_detection_rate"],
                 100 * z["stage2_only_rate"])


if __name__ == "__main__":
    main()
