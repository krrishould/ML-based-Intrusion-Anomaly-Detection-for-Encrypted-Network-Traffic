"""Tests for the evaluation metrics.

These guard the numbers that go into the report. A metric that is quietly wrong
is worse than no metric, because it still looks like evidence.
"""
from __future__ import annotations

import numpy as np
import pytest

from encids.models import metrics as M


class TestBinaryMetrics:
    def test_perfect_prediction(self):
        y = [0, 0, 1, 1]
        m = M.binary_metrics(y, y)
        assert m["accuracy"] == 1.0
        assert m["f1"] == 1.0
        assert m["false_positive_rate"] == 0.0
        assert m["false_negative_rate"] == 0.0

    def test_confusion_counts(self):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 1, 0]
        m = M.binary_metrics(y_true, y_pred)
        assert (m["true_negatives"], m["false_positives"]) == (3, 1)
        assert (m["false_negatives"], m["true_positives"]) == (1, 3)

    def test_false_positive_rate_is_over_negatives_only(self):
        """FPR must be FP/(FP+TN), not FP/total - the class balance here is 9:1."""
        y_true = [0] * 90 + [1] * 10
        y_pred = [0] * 81 + [1] * 9 + [1] * 10
        m = M.binary_metrics(y_true, y_pred)
        assert m["false_positive_rate"] == pytest.approx(9 / 90)

    def test_all_negative_prediction_scores_zero_recall(self):
        m = M.binary_metrics([0, 1, 1], [0, 0, 0])
        assert m["recall"] == 0.0
        assert m["precision"] == 0.0          # zero_division guard
        assert m["false_positive_rate"] == 0.0

    def test_auc_added_only_when_scores_given(self):
        y = [0, 0, 1, 1]
        assert "roc_auc" not in M.binary_metrics(y, y)
        assert "roc_auc" in M.binary_metrics(y, y, scores=[0.1, 0.2, 0.8, 0.9])

    def test_single_class_ground_truth_does_not_crash(self):
        m = M.binary_metrics([0, 0, 0], [0, 0, 1], scores=[0.1, 0.2, 0.9])
        assert m["mcc"] == 0.0
        assert "roc_auc" not in m           # undefined with one class


class TestPerClassDetection:
    def test_rate_per_family(self):
        families = ["benign"] * 4 + ["c2"] * 4
        alerts = [0, 0, 0, 1, 1, 1, 1, 0]
        rates = M.per_class_detection_rate(families, alerts)
        assert rates["benign"] == pytest.approx(0.25)   # this is the FPR
        assert rates["c2"] == pytest.approx(0.75)


class TestThresholdSweep:
    def test_recall_falls_as_the_threshold_rises(self):
        rng = np.random.default_rng(0)
        benign = rng.normal(0, 1, 2000)
        malicious = rng.normal(2.5, 1, 500)
        y = np.r_[np.zeros(2000), np.ones(500)]
        scores = np.r_[benign, malicious]

        table = M.threshold_sweep(y, scores)
        assert list(table["recall"]) == sorted(table["recall"], reverse=True)
        assert list(table["false_positive_rate"]) == sorted(
            table["false_positive_rate"], reverse=True)

    def test_fpr_matches_the_percentile_used(self):
        rng = np.random.default_rng(1)
        benign = rng.normal(0, 1, 5000)
        y = np.r_[np.zeros(5000), np.ones(100)]
        scores = np.r_[benign, rng.normal(3, 1, 100)]
        table = M.threshold_sweep(y, scores, benign_scores=benign,
                                  percentiles=(95.0, 99.0))
        by_pct = dict(zip(table["percentile"], table["false_positive_rate"]))
        assert by_pct[95.0] == pytest.approx(0.05, abs=0.01)
        assert by_pct[99.0] == pytest.approx(0.01, abs=0.005)


class TestEvasionStratification:
    def test_splits_malicious_flows_into_bands(self):
        y = [1] * 8 + [0] * 4
        alert = [1, 1, 1, 1, 1, 0, 0, 0] + [0] * 4
        evasion = [0.0, 0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 0.9] + [0.0] * 4
        table = M.stratify_by_evasion(y, alert, evasion)
        assert int(table["n_flows"].sum()) == 8          # benign rows excluded
        rates = dict(zip(table["band"].astype(str), table["detection_rate"]))
        assert rates["none"] == 1.0
        assert rates["heavy (0.6-1.0)"] == 0.0

    def test_empty_when_nothing_is_malicious(self):
        assert M.stratify_by_evasion([0, 0], [0, 0], [0.0, 0.0]).empty


class TestZeroDayReport:
    def test_splits_credit_between_the_stages(self):
        families = ["tor"] * 10 + ["benign"] * 10
        alert = [1] * 6 + [0] * 4 + [0] * 10
        stage1 = [1] * 3 + [0] * 7 + [0] * 10
        stage2 = [0, 0, 0, 1, 1, 1] + [0] * 4 + [0] * 10

        report = M.zero_day_report(families, alert, stage1, stage2, "tor")
        assert report["n_flows"] == 10
        assert report["overall_detection_rate"] == pytest.approx(0.6)
        assert report["stage1_rate"] == pytest.approx(0.3)
        assert report["stage2_only_rate"] == pytest.approx(0.3)
        assert report["missed"] == 4

    def test_absent_family_is_reported_not_crashed(self):
        report = M.zero_day_report(["benign"], [0], [0], [0], "nonexistent")
        assert report["n_flows"] == 0
        assert "note" in report


class TestComparison:
    def test_builds_a_delta_table(self):
        baseline = {"accuracy": 0.90, "f1": 0.80, "false_positive_rate": 0.05}
        full = {"accuracy": 0.95, "f1": 0.88, "false_positive_rate": 0.02}
        table = M.compare(baseline, full)
        row = table.set_index("metric").loc["f1"]
        assert row["delta"] == pytest.approx(0.08)
        assert row["relative_%"] == pytest.approx(10.0)

    def test_skips_metrics_missing_from_either_side(self):
        table = M.compare({"accuracy": 0.9}, {"accuracy": 0.95, "f1": 0.8})
        assert list(table["metric"]) == ["accuracy"]
