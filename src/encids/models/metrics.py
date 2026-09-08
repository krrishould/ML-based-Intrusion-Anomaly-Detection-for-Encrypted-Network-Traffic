"""Evaluation metrics.

Objective 5 asks for accuracy, F1, false-positive rate and detection latency,
measured against a flow-statistics-only baseline.

False-positive rate is reported prominently and deliberately: on a network
carrying millions of flows an hour, a 1% FPR is tens of thousands of alerts a
day, so a model that wins on accuracy while losing on FPR has not actually won.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(y_true, y_pred, scores=None) -> dict[str, Any]:
    """Standard binary detection metrics plus FPR/FNR."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_positives": int(tp), "false_positives": int(fp),
        "true_negatives": int(tn), "false_negatives": int(fn),
    }
    if scores is not None and len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    return out


def multiclass_metrics(y_true, y_pred) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",
                                      zero_division=0)),
        "per_class": classification_report(y_true, y_pred, output_dict=True,
                                           zero_division=0),
    }


def per_class_detection_rate(y_true_family, alert, benign_label="benign"
                             ) -> dict[str, float]:
    """Recall broken down by attack family.

    This is the table that tells you whether the headline F1 is carried
    entirely by one easy, over-represented family.
    """
    df = pd.DataFrame({"family": np.asarray(y_true_family).astype(str),
                       "alert": np.asarray(alert).astype(int)})
    rates = df.groupby("family")["alert"].mean().to_dict()
    return {str(k): float(v) for k, v in sorted(rates.items())}


def measure_latency(detector, df: pd.DataFrame, n_repeats: int = 5,
                    batch_sizes: tuple[int, ...] = (1, 32, 256)) -> dict[str, Any]:
    """Per-flow scoring latency - the deployability number.

    A detector that needs 40 ms per flow cannot keep up with a gigabit link, no
    matter how good its F1 is.
    """
    out: dict[str, Any] = {}
    for size in batch_sizes:
        if len(df) < size:
            continue
        sample = df.iloc[:size]
        timings = []
        for _ in range(n_repeats):
            start = time.perf_counter()
            detector.score_frame(sample)
            timings.append(time.perf_counter() - start)
        best = float(np.median(timings))
        out[f"batch_{size}"] = {
            "total_ms": round(best * 1000, 3),
            "per_flow_ms": round(best * 1000 / size, 4),
            "flows_per_second": round(size / best, 1),
        }
    return out


def zero_day_report(y_family, alert, stage1_fired, stage2_fired,
                    held_out: str) -> dict[str, Any]:
    """How the two stages split the work on a family Stage 1 never saw.

    This is the central experiment for the project's zero-day claim: if Stage 2
    is doing its job, ``stage2_only_rate`` on the held-out family should be
    substantially above zero.
    """
    family = np.asarray(y_family).astype(str)
    mask = family == held_out
    if not mask.any():
        return {"held_out": held_out, "n_flows": 0,
                "note": "family not present in the evaluation set"}

    a = np.asarray(alert).astype(int)[mask]
    s1 = np.asarray(stage1_fired).astype(int)[mask]
    s2 = np.asarray(stage2_fired).astype(int)[mask]
    n = int(mask.sum())
    return {
        "held_out": held_out,
        "n_flows": n,
        "overall_detection_rate": float(a.mean()),
        "stage1_rate": float(s1.mean()),
        "stage2_rate": float(s2.mean()),
        "stage2_only_rate": float(((s2 == 1) & (s1 == 0)).mean()),
        "missed": int((a == 0).sum()),
    }


def threshold_sweep(y_true, scores, benign_scores=None,
                    percentiles: tuple[float, ...] = (90, 95, 97.5, 99, 99.5, 99.9)
                    ) -> pd.DataFrame:
    """Detection rate vs. false-positive rate across operating points.

    The Stage-2 threshold is a deployment choice, not a property of the model.
    This table is what lets an operator pick it: at p99 the system raises ~1
    false alert per 100 benign flows, at p95 five times as many for a higher
    catch rate.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    reference = (np.asarray(benign_scores, dtype=float)
                 if benign_scores is not None else scores[y_true == 0])

    rows = []
    for pct in percentiles:
        threshold = float(np.percentile(reference, pct))
        pred = (scores > threshold).astype(int)
        m = binary_metrics(y_true, pred)
        rows.append({
            "percentile": pct,
            "threshold": round(threshold, 6),
            "recall": round(m["recall"], 4),
            "precision": round(m["precision"], 4),
            "f1": round(m["f1"], 4),
            "false_positive_rate": round(m["false_positive_rate"], 4),
        })
    return pd.DataFrame(rows)


def stratify_by_evasion(y_true, alert, evasion_level,
                        bins: tuple[float, ...] = (0.0, 0.01, 0.3, 0.6, 1.01)
                        ) -> pd.DataFrame:
    """Detection rate on malicious flows, split by how hard they were shaped.

    An aggregate recall hides the only number that matters for an evasion
    argument: whether the system still fires on traffic that was deliberately
    made to look benign.
    """
    mask = np.asarray(y_true).astype(int) == 1
    if not mask.any():
        return pd.DataFrame()
    df = pd.DataFrame({
        "evasion": pd.to_numeric(pd.Series(evasion_level), errors="coerce")
        .to_numpy()[mask],
        "alert": np.asarray(alert).astype(int)[mask],
    })
    # Sources other than the synthetic generator carry no evasion annotation;
    # an all-NaN column must produce no table rather than a table of NaNs.
    df = df.dropna(subset=["evasion"])
    if df.empty:
        return pd.DataFrame()
    labels = ["none", "light (0-0.3)", "moderate (0.3-0.6)", "heavy (0.6-1.0)"]
    df["band"] = pd.cut(df["evasion"], bins=list(bins), labels=labels,
                        include_lowest=True, right=False)
    out = df.groupby("band", observed=False).agg(
        n_flows=("alert", "size"), detection_rate=("alert", "mean")).reset_index()
    out["detection_rate"] = out["detection_rate"].round(4)
    return out


def compare(baseline: dict[str, Any], full: dict[str, Any],
            keys: tuple[str, ...] = ("accuracy", "f1", "recall", "precision",
                                     "false_positive_rate", "roc_auc")
            ) -> pd.DataFrame:
    """Side-by-side table: flow-only baseline vs. the dual-signal system."""
    rows = []
    for key in keys:
        if key in baseline and key in full:
            b, f = float(baseline[key]), float(full[key])
            rows.append({
                "metric": key,
                "flow_only_baseline": round(b, 4),
                "flow_plus_tls": round(f, 4),
                "delta": round(f - b, 4),
                "relative_%": round(100 * (f - b) / b, 2) if b else None,
            })
    return pd.DataFrame(rows)
