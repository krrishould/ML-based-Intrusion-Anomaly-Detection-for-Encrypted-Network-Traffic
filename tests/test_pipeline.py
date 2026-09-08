"""Tests for feature preprocessing, the two stages, fusion, and the live scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from encids.config import load_config
from encids.data import synthetic
from encids.data.pcap_to_flows import add_fingerprint_buckets
from encids.features import schema
from encids.features.build_features import FeaturePipeline, assemble, clean_flow_table
from encids.models.anomaly import AnomalyDetector
from encids.models.fusion import TwoStageDetector
from encids.models.supervised import SupervisedDetector


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return assemble([synthetic.generate(n_flows=2500, seed=11)])


@pytest.fixture(scope="module")
def trained(table):
    """A small two-stage detector, trained once for the whole module."""
    pipeline = FeaturePipeline(use_flow=True, use_tls=True)
    X = pipeline.fit_transform(table)

    stage1 = SupervisedDetector(
        model_type="random_forest",
        params={"n_estimators": 60, "n_jobs": -1},
    ).fit(X, table[schema.TARGET_COLUMN], feature_names=pipeline.columns,
          malicious_mask=table[schema.BINARY_TARGET_COLUMN])

    benign = X[(table[schema.BINARY_TARGET_COLUMN] == 0).to_numpy()]
    stage2 = AnomalyDetector(
        model_type="isolation_forest",
        params={"n_estimators": 60},
        exclude_features=["ja3_bucket", "ja3s_bucket"],
    ).fit(benign, feature_names=pipeline.columns)

    return TwoStageDetector(pipeline=pipeline, supervised=stage1, anomaly=stage2)


# ---------------------------------------------------------------------------
class TestSyntheticGenerator:
    def test_is_deterministic_for_a_given_seed(self):
        a = synthetic.generate(n_flows=200, seed=5)
        b = synthetic.generate(n_flows=200, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_differ(self):
        a = synthetic.generate(n_flows=200, seed=1)
        b = synthetic.generate(n_flows=200, seed=2)
        assert not a["total_bytes"].equals(b["total_bytes"])

    def test_attack_fraction_is_respected(self):
        df = synthetic.generate(n_flows=2000, attack_fraction=0.3, seed=3)
        assert df[schema.BINARY_TARGET_COLUMN].mean() == pytest.approx(0.3, abs=0.03)

    def test_labels_and_binary_target_agree(self):
        df = synthetic.generate(n_flows=800, seed=4)
        for label, group in df.groupby(schema.TARGET_COLUMN):
            expected = int(label in synthetic.ATTACK_CLASSES)
            assert set(group[schema.BINARY_TARGET_COLUMN]) == {expected}

    def test_excluded_classes_are_absent(self):
        df = synthetic.generate(n_flows=800, seed=6,
                                exclude_classes=("c2_beacon", "port_scan"))
        assert "c2_beacon" not in set(df[schema.TARGET_COLUMN])
        assert "port_scan" not in set(df[schema.TARGET_COLUMN])

    def test_derived_statistics_are_internally_consistent(self):
        """Generating packets (not columns) should keep totals self-consistent."""
        df = synthetic.generate(n_flows=400, seed=8)
        assert (df["total_packets"] == df["fwd_packets"] + df["bwd_packets"]).all()
        assert (df["total_bytes"] == df["fwd_bytes"] + df["bwd_bytes"]).all()
        finite = df[df["total_packets"] > 0]
        expected = finite["total_bytes"] / finite["total_packets"]
        assert np.allclose(finite["avg_packet_size"], expected)

    def test_evasion_produces_a_hard_tail(self):
        df = synthetic.generate(n_flows=2000, seed=9, evasion_rate=0.5)
        malicious = df[df[schema.BINARY_TARGET_COLUMN] == 1]
        assert (malicious["evasion_level"] > 0).mean() == pytest.approx(0.5, abs=0.08)

    def test_no_evasion_when_disabled(self):
        df = synthetic.generate(n_flows=500, seed=10, evasion_rate=0.0)
        assert (df["evasion_level"] == 0).all()

    def test_browser_pool_is_not_the_reference_list(self):
        """Guard against a circular TLS feature.

        If the generator drew browser fingerprints from the very set that
        `ja3_is_known_browser` checks against, that feature would become a
        near-perfect label proxy and the TLS ablation would measure a shared
        lookup table rather than a real signal.
        """
        from encids.features.tls_features import KNOWN_BROWSER_JA3

        pool = set(synthetic._BROWSER_JA3_POOL)
        assert pool != KNOWN_BROWSER_JA3
        assert pool - KNOWN_BROWSER_JA3, "pool must contain unlisted browsers"
        assert pool & KNOWN_BROWSER_JA3, "some overlap keeps the feature useful"

    def test_benign_traffic_is_not_perfectly_identified_by_the_browser_flag(self):
        df = synthetic.generate(n_flows=3000, seed=44)
        benign = df[df[schema.BINARY_TARGET_COLUMN] == 0]
        # If this were ~1.0 the feature would be a giveaway rather than a hint.
        assert benign["ja3_is_known_browser"].mean() < 0.6

    def test_flows_include_tcp_handshake_packets(self):
        """Real flows always carry 40-byte control packets; synthetic ones must
        too, or a model trained here flags every real flow as anomalous."""
        df = synthetic.generate(n_flows=300, seed=45)
        assert (df["fwd_pkt_len_min"] == 40).mean() > 0.9
        assert (df["syn_count"] >= 1).all()


# ---------------------------------------------------------------------------
class TestFeaturePipeline:
    def test_identifier_columns_never_reach_the_model(self, table):
        pipeline = FeaturePipeline().fit(table)
        leaked = set(pipeline.columns) & set(schema.ID_COLUMNS)
        assert not leaked, f"identifier columns leaked into the model: {leaked}"

    def test_label_columns_never_reach_the_model(self, table):
        pipeline = FeaturePipeline().fit(table)
        assert schema.TARGET_COLUMN not in pipeline.columns
        assert schema.BINARY_TARGET_COLUMN not in pipeline.columns

    def test_output_is_finite(self, table):
        X = FeaturePipeline().fit_transform(table)
        assert np.isfinite(X).all()

    def test_infinities_are_handled(self, table):
        dirty = table.copy()
        dirty.loc[dirty.index[:20], "flow_bytes_per_s"] = np.inf
        dirty.loc[dirty.index[20:40], "flow_packets_per_s"] = -np.inf
        X = FeaturePipeline().fit_transform(dirty)
        assert np.isfinite(X).all()

    def test_missing_values_are_imputed(self, table):
        dirty = table.copy()
        dirty.loc[dirty.index[:50], "pkt_len_std"] = np.nan
        X = FeaturePipeline().fit_transform(dirty)
        assert np.isfinite(X).all()

    def test_transform_is_stable_across_calls(self, table):
        pipeline = FeaturePipeline().fit(table)
        assert np.allclose(pipeline.transform(table), pipeline.transform(table))

    def test_column_order_is_fixed(self, table):
        a = FeaturePipeline().fit(table).columns
        b = FeaturePipeline().fit(table).columns
        assert a == b

    def test_tls_toggle_changes_the_feature_count(self, table):
        with_tls = FeaturePipeline(use_tls=True).fit(table)
        without = FeaturePipeline(use_tls=False).fit(table)
        assert len(with_tls.columns) > len(without.columns)
        assert not set(without.columns) & set(schema.TLS_FEATURES)

    def test_handles_a_frame_missing_optional_columns(self, table):
        pipeline = FeaturePipeline().fit(table)
        reduced = table.drop(columns=["sni_entropy", "cert_chain_len"])
        assert np.isfinite(pipeline.transform(reduced)).all()

    def test_roundtrips_through_disk(self, table, tmp_path):
        pipeline = FeaturePipeline().fit(table)
        path = tmp_path / "pipeline.joblib"
        pipeline.save(path)
        assert np.allclose(FeaturePipeline.load(path).transform(table),
                           pipeline.transform(table))

    def test_records_the_training_fingerprint_distribution(self, table):
        pipeline = FeaturePipeline().fit(table)
        assert pipeline.ja3_frequency_
        assert sum(pipeline.ja3_frequency_.values()) == pytest.approx(1.0, abs=1e-6)

    def test_all_nan_feature_is_dropped_not_imputed(self, table):
        """sklearn silently drops empty columns; the matrix must stay aligned."""
        dirty = table.copy()
        dirty["sni_entropy"] = np.nan
        pipeline = FeaturePipeline().fit(dirty)
        assert "sni_entropy" not in pipeline.columns
        assert pipeline.transform(dirty).shape[1] == len(pipeline.columns)


class TestFingerprintRarityAtInference:
    """Rarity must come from the training corpus, never from the live batch."""

    def test_rarity_uses_the_frozen_training_distribution(self, table):
        pipeline = FeaturePipeline().fit(table)
        live = synthetic.generate(n_flows=12, seed=101)
        out = pipeline.apply_fingerprints(live)

        expected = [
            1.0 / pipeline.ja3_frequency_[h] if h in pipeline.ja3_frequency_ else None
            for h in live["ja3_hash"].astype(str)
        ]
        for got, want in zip(out["ja3_rarity"], expected):
            if want is not None:
                assert got == pytest.approx(want)

    def test_a_small_batch_does_not_collapse_the_signal(self, table):
        """The bug this guards: per-batch counts make everything look rare."""
        pipeline = FeaturePipeline().fit(table)
        live = synthetic.generate(n_flows=10, seed=102)
        frozen = pipeline.apply_fingerprints(live)["ja3_rarity"]
        per_batch = add_fingerprint_buckets(live)["ja3_rarity"]
        # Per-batch counts can only take a handful of values in a 10-row window.
        assert frozen.nunique() >= per_batch.nunique()

    def test_unseen_fingerprint_gets_max_training_rarity_not_infinity(self, table):
        pipeline = FeaturePipeline().fit(table)
        live = synthetic.generate(n_flows=5, seed=103).copy()
        live["ja3_hash"] = "f" * 32                     # never seen in training
        rarity = pipeline.apply_fingerprints(live)["ja3_rarity"].to_numpy()
        assert np.isfinite(rarity).all()
        assert np.allclose(rarity, 1.0 / min(pipeline.ja3_frequency_.values()))

    def test_buckets_stay_in_range(self, table):
        pipeline = FeaturePipeline().fit(table)
        live = synthetic.generate(n_flows=30, seed=104)
        out = pipeline.apply_fingerprints(live, n_buckets=64)
        assert out["ja3_bucket"].between(0, 63).all()
        assert out["ja3s_bucket"].between(0, 63).all()


class TestCleaning:
    def test_drops_single_packet_flows(self):
        df = synthetic.generate(n_flows=500, seed=12)
        df.loc[df.index[:10], "total_packets"] = 1
        cleaned = clean_flow_table(df, min_packets=2)
        assert (cleaned["total_packets"] >= 2).all()

    def test_drops_exact_duplicates(self):
        df = synthetic.generate(n_flows=200, seed=13)
        duplicated = pd.concat([df, df], ignore_index=True)
        assert len(clean_flow_table(duplicated)) <= len(df)

    def test_fingerprint_buckets_are_in_range(self, table):
        out = add_fingerprint_buckets(table, n_buckets=64)
        assert out["ja3_bucket"].between(0, 63).all()
        assert (out["ja3_rarity"] >= 0).all()


# ---------------------------------------------------------------------------
class TestStages:
    def test_stage1_beats_chance(self, trained, table):
        X = trained.pipeline.transform(table)
        predicted = trained.supervised.predict_labels(X)
        accuracy = (predicted == table[schema.TARGET_COLUMN].to_numpy()).mean()
        assert accuracy > 0.5

    def test_malicious_probability_is_a_probability(self, trained, table):
        p = trained.supervised.predict_malicious_proba(
            trained.pipeline.transform(table))
        assert ((p >= 0) & (p <= 1)).all()

    def test_stage2_flags_roughly_the_configured_fraction_of_benign(self, trained,
                                                                    table):
        benign = table[table[schema.BINARY_TARGET_COLUMN] == 0]
        X = trained.pipeline.transform(benign)
        flagged = trained.anomaly.predict(X).mean()
        # Threshold is the 99th percentile of benign training scores.
        assert flagged < 0.05

    def test_stage2_excludes_the_configured_features(self, trained):
        assert len(trained.anomaly._keep_idx) == \
            len(trained.pipeline.columns) - 2

    def test_stage2_scores_higher_for_malicious_on_average(self, trained, table):
        X = trained.pipeline.transform(table)
        scores = trained.anomaly.score(X)
        malicious = table[schema.BINARY_TARGET_COLUMN].to_numpy() == 1
        assert scores[malicious].mean() > scores[~malicious].mean()

    def test_threshold_recalibration(self, trained, table):
        original = trained.anomaly.threshold_
        benign = table[table[schema.BINARY_TARGET_COLUMN] == 0]
        scores = trained.anomaly.score(trained.pipeline.transform(benign))
        trained.anomaly.recalibrate(scores, percentile=50)
        assert trained.anomaly.threshold_ < original
        trained.anomaly.threshold_ = original      # restore for other tests


class TestFusion:
    def test_or_rule_fires_when_either_stage_fires(self, trained, table):
        trained.rule = "or"
        verdicts = trained.score_frame(table)
        expected = (verdicts["stage1_fired"] | verdicts["stage2_fired"])
        assert (verdicts["alert"] == expected).all()

    def test_and_rule_requires_both(self, trained, table):
        trained.rule = "and"
        verdicts = trained.score_frame(table)
        expected = (verdicts["stage1_fired"] & verdicts["stage2_fired"])
        assert (verdicts["alert"] == expected).all()
        trained.rule = "or"

    def test_or_alerts_at_least_as_often_as_either_stage(self, trained, table):
        trained.rule = "or"
        v = trained.score_frame(table)
        assert v["alert"].sum() >= v["stage1_fired"].sum()
        assert v["alert"].sum() >= v["stage2_fired"].sum()

    def test_reason_is_benign_exactly_when_no_alert(self, trained, table):
        v = trained.score_frame(table)
        assert (v.loc[v["alert"] == 0, "reason"] == "benign").all()
        assert (v.loc[v["alert"] == 1, "reason"] != "benign").all()

    def test_reason_matches_which_stage_fired(self, trained, table):
        trained.rule = "or"
        v = trained.score_frame(table)
        both = v[(v["stage1_fired"] == 1) & (v["stage2_fired"] == 1)]
        assert (both["reason"] == "known-attack+anomalous").all()
        only1 = v[(v["stage1_fired"] == 1) & (v["stage2_fired"] == 0)]
        assert (only1["reason"] == "known-attack").all()
        only2 = v[(v["stage1_fired"] == 0) & (v["stage2_fired"] == 1)]
        assert (only2["reason"] == "anomalous").all()

    def test_risk_is_bounded(self, trained, table):
        v = trained.score_frame(table)
        assert v["risk"].between(0, 1).all()

    def test_unknown_rule_raises(self, trained, table):
        trained.rule = "nonsense"
        with pytest.raises(ValueError, match="Unknown fusion rule"):
            trained.score_frame(table.head(10))
        trained.rule = "or"

    def test_roundtrips_through_disk(self, trained, table, tmp_path):
        trained.save(tmp_path)
        reloaded = TwoStageDetector.load(tmp_path)
        a = trained.score_frame(table.head(200))
        b = reloaded.score_frame(table.head(200))
        assert (a["alert"] == b["alert"]).all()
        assert np.allclose(a["p_malicious"], b["p_malicious"])


class TestLiveScorer:
    def test_scores_a_batch_of_records(self, trained):
        from encids.live.scorer import LiveScorer

        scorer = LiveScorer(detector=trained, explainer=None)
        records = synthetic.generate(n_flows=40, seed=21).to_dict(orient="records")
        out = scorer.score_batch(records)

        assert len(out) == 40
        assert {"alert", "reason", "risk"} <= set(out.columns)
        assert scorer.stats()["total_flows"] == 40

    def test_empty_batch_is_safe(self, trained):
        from encids.live.scorer import LiveScorer

        assert LiveScorer(detector=trained).score_batch([]).empty

    def test_snapshot_is_newest_first(self, trained):
        from encids.live.scorer import LiveScorer

        scorer = LiveScorer(detector=trained, explainer=None)
        scorer.score_batch(synthetic.generate(n_flows=10, seed=22)
                           .to_dict(orient="records"))
        scorer.score_batch(synthetic.generate(n_flows=10, seed=23)
                           .to_dict(orient="records"))
        assert len(scorer.snapshot()) == 20

    def test_rolling_window_is_bounded(self, trained):
        from encids.live.scorer import LiveScorer

        scorer = LiveScorer(detector=trained, explainer=None, max_rows=25)
        for seed in range(4):
            scorer.score_batch(synthetic.generate(n_flows=20, seed=seed)
                               .to_dict(orient="records"))
        assert len(scorer.snapshot()) == 25
        assert scorer.stats()["total_flows"] == 80


class TestSubsampling:
    """Sub-sampling must not silently delete rare attack families."""

    def test_respects_the_row_budget(self, table):
        from encids.train import _subsample

        out = _subsample(table, max_rows=500, seed=1)
        assert 400 <= len(out) <= 700          # floors can push it slightly over

    def test_keeps_every_class(self):
        from encids.train import _subsample

        df = synthetic.generate(n_flows=3000, seed=41)
        # Make one family genuinely rare, the way CTU-13's Sogou botnet is.
        rare = df[df[schema.TARGET_COLUMN] == "crypto_mining"].head(4)
        rest = df[df[schema.TARGET_COLUMN] != "crypto_mining"]
        skewed = pd.concat([rest, rare], ignore_index=True)

        out = _subsample(skewed, max_rows=300, seed=1)
        assert set(out[schema.TARGET_COLUMN]) == set(skewed[schema.TARGET_COLUMN])
        assert (out[schema.TARGET_COLUMN] == "crypto_mining").sum() >= 2

    def test_preserves_all_columns(self, table):
        from encids.train import _subsample

        out = _subsample(table, max_rows=400, seed=1)
        assert list(out.columns) == list(table.columns)

    def test_is_deterministic(self, table):
        from encids.train import _subsample

        a = _subsample(table, max_rows=400, seed=7)
        b = _subsample(table, max_rows=400, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestConfig:
    def test_loads_and_exposes_dotted_paths(self):
        cfg = load_config()
        assert cfg.get_path("supervised.model") in ("random_forest", "xgboost")
        assert cfg.get_path("anomaly.model") in ("isolation_forest", "autoencoder")
        assert cfg.get_path("fusion.rule") in ("or", "and", "weighted")

    def test_missing_path_returns_the_default(self):
        assert load_config().get_path("nope.not.here", "fallback") == "fallback"

    def test_paths_resolve_against_the_project_root(self):
        assert load_config().resolve("paths.models").is_absolute()


class TestISCXLabelling:
    """ISCX filenames are inconsistent; a silent fallback mislabelled 7 of 23."""

    def _label(self, filename: str) -> str:
        from pathlib import Path

        from encids.data.dataset_loaders import _iscx_label

        return _iscx_label(Path(filename))

    @pytest.mark.parametrize("filename,expected", [
        ("aim_chat_3a.pcap", "chat"),
        ("AIMchat1.pcapng", "chat"),
        ("facebook_chat_4a.pcap", "chat"),
        ("facebookchat1.pcapng", "chat"),        # no separator - was the bug
        ("facebook_video1a.pcap", "streaming"),  # no key at all - was the bug
        ("facebook_audio3.pcapng", "voip"),
        ("email1a.pcap", "email"),
        ("youtube2.pcap", "streaming"),
        ("skype_file1.pcap", "file_transfer"),
        ("torrent01.pcap", "p2p"),
    ])
    def test_known_applications_are_categorised(self, filename, expected):
        assert self._label(filename) == expected

    def test_separator_style_does_not_change_the_label(self):
        assert self._label("facebookchat1.pcapng") == \
            self._label("facebook_chat_1.pcap")

    def test_vpn_prefix_marks_traffic_as_tunnelled(self):
        assert self._label("vpn_youtube_A.pcap") == "vpn_streaming"

    def test_nonvpn_prefix_is_not_read_as_tunnelled(self):
        """'nonvpn' contains 'vpn' - it must not flip the tunnelled flag."""
        assert not self._label("nonvpn_chat1.pcap").startswith("vpn_")

    def test_longer_keys_win_over_shorter_ones(self):
        assert self._label("skype_file2.pcap") == "file_transfer"
        assert self._label("skype_audio1.pcap") == "voip"

    def test_unknown_application_falls_back_and_can_warn(self, caplog):
        from pathlib import Path

        from encids.data.dataset_loaders import _iscx_label

        with caplog.at_level("WARNING", logger="encids.data.loaders"):
            label = _iscx_label(Path("totally_unknown_app7.pcap"),
                                warn_on_fallback=True)
        assert label == "web_browsing"
        assert "matched no known application" in caplog.text

    def test_no_warning_when_the_application_is_known(self, caplog):
        from pathlib import Path

        from encids.data.dataset_loaders import _iscx_label

        with caplog.at_level("WARNING", logger="encids.data.loaders"):
            _iscx_label(Path("email1a.pcap"), warn_on_fallback=True)
        assert "matched no known application" not in caplog.text
