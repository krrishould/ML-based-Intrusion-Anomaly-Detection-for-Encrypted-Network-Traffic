"""End-to-end test: write a pcap, read it back, check the extracted features.

This is the test that matters most for deployment correctness. The offline
training path and the live scoring path both go through
:func:`flows_from_pcap` / :class:`FlowTable`, so if a real capture does not
round-trip into sensible features, every metric computed downstream is
measuring the wrong thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encids.features.flow_stats import flows_from_pcap  # noqa: E402

scapy = pytest.importorskip("scapy.all", reason="scapy is required")


@pytest.fixture(scope="module")
def sample_pcap(tmp_path_factory) -> Path:
    from make_sample_pcap import build

    path = tmp_path_factory.mktemp("pcap") / "sample.pcap"
    build(path, n_flows=25, seed=7)
    return path


@pytest.fixture(scope="module")
def records(sample_pcap):
    return flows_from_pcap(str(sample_pcap))


class TestPcapRoundTrip:
    def test_produces_flows(self, records):
        assert len(records) > 0

    def test_flow_count_matches_the_connections_written(self, records):
        # 25 connections in, 25 flows out - nothing split or merged.
        assert len(records) == 25

    def test_totals_are_self_consistent(self, records):
        for r in records:
            assert r["total_packets"] == r["fwd_packets"] + r["bwd_packets"]
            assert r["total_bytes"] == r["fwd_bytes"] + r["bwd_bytes"]

    def test_every_flow_has_a_positive_duration_or_a_single_packet(self, records):
        for r in records:
            assert r["duration_ms"] >= 0
            if r["total_packets"] > 1:
                assert r["duration_ms"] > 0

    def test_tls_handshakes_are_parsed(self, records):
        tls = [r for r in records if r["is_tls"]]
        assert tls, "no TLS handshake was recovered from the capture"
        for r in tls:
            assert len(r["ja3_hash"]) == 32

    def test_sni_is_recovered(self, records):
        hostnames = {r.get("sni") for r in records if r.get("sni")}
        assert "www.example.com" in hostnames

    def test_browser_and_malware_fingerprints_differ(self, records):
        """Different cipher/extension sets must yield different JA3 hashes."""
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        browser = by_host.get("www.example.com")
        malware = by_host.get("a7f3k9q2m1x8.net")
        assert browser and malware
        assert browser["ja3_hash"] != malware["ja3_hash"]

    def test_grease_is_detected_for_browser_like_traffic(self, records):
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        assert by_host["www.example.com"]["has_grease"] == 1
        assert by_host["a7f3k9q2m1x8.net"]["has_grease"] == 0

    def test_alpn_is_detected(self, records):
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        assert by_host["www.example.com"]["alpn_is_h2"] == 1

    def test_high_entropy_hostname_scores_above_a_readable_one(self, records):
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        assert (by_host["a7f3k9q2m1x8.net"]["sni_entropy"]
                > by_host["www.example.com"]["sni_entropy"])

    def test_exfiltration_flow_is_upload_heavy(self, records):
        """The direction convention must survive a real pcap round trip."""
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        exfil = by_host["backup-sync.net"]
        assert exfil["fwd_bytes"] > exfil["bwd_bytes"], \
            "client-to-server bytes should dominate an exfiltration flow"
        assert exfil["fwd_bytes_fraction"] > 0.5

    def test_browsing_flow_is_download_heavy(self, records):
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        browsing = by_host["www.example.com"]
        assert browsing["bwd_bytes"] > browsing["fwd_bytes"], \
            "server-to-client bytes should dominate normal browsing"

    def test_beacon_timing_is_more_regular_than_browsing(self, records):
        """The core behavioural signal: automated traffic has low IAT variance."""
        by_host = {r["sni"]: r for r in records if r.get("sni")}
        beacon = by_host["a7f3k9q2m1x8.net"]
        browsing = by_host["www.example.com"]
        beacon_cv = beacon["flow_iat_std"] / max(beacon["flow_iat_mean"], 1e-9)
        browsing_cv = browsing["flow_iat_std"] / max(browsing["flow_iat_mean"], 1e-9)
        assert beacon_cv < browsing_cv

    def test_port_scan_flows_carry_a_reset(self, records):
        scans = [r for r in records if r["total_packets"] <= 5
                 and r["rst_count"] > 0]
        assert scans, "the port-scan flows should show RST responses"

    def test_scoring_a_pcap_end_to_end(self, records, tmp_path):
        """Train a tiny model and score real pcap-derived flows with it."""
        import pandas as pd

        from encids.data import synthetic
        from encids.data.pcap_to_flows import add_fingerprint_buckets
        from encids.features import schema
        from encids.features.build_features import FeaturePipeline, assemble
        from encids.models.anomaly import AnomalyDetector
        from encids.models.fusion import TwoStageDetector
        from encids.models.supervised import SupervisedDetector

        train_df = assemble([synthetic.generate(n_flows=1200, seed=31)])
        pipeline = FeaturePipeline()
        X = pipeline.fit_transform(train_df)
        stage1 = SupervisedDetector(
            model_type="random_forest", params={"n_estimators": 40}
        ).fit(X, train_df[schema.TARGET_COLUMN], feature_names=pipeline.columns,
              malicious_mask=train_df[schema.BINARY_TARGET_COLUMN])
        benign = X[(train_df[schema.BINARY_TARGET_COLUMN] == 0).to_numpy()]
        stage2 = AnomalyDetector(model_type="isolation_forest",
                                 params={"n_estimators": 40}).fit(
            benign, feature_names=pipeline.columns)
        detector = TwoStageDetector(pipeline, stage1, stage2)

        live = add_fingerprint_buckets(pd.DataFrame.from_records(records))
        verdicts = detector.score_frame(live)

        assert len(verdicts) == len(records)
        assert verdicts["risk"].between(0, 1).all()
        assert set(verdicts["alert"].unique()) <= {0, 1}


class TestLinkLayerRecovery:
    """Scapy on Windows/Npcap can hand back undecoded frames as bare Raw.

    Capture then looks healthy - packets arrive, no errors - while yielding
    zero flows, because nothing has an IP layer to key on.
    """

    def _frame(self):
        from scapy.all import IP, TCP, Ether

        return (Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
                / IP(src="10.0.0.5", dst="93.184.216.34")
                / TCP(sport=44321, dport=443, flags="PA"))

    def test_raw_frame_is_recovered(self):
        from scapy.all import IP, Raw

        from encids.live.capture import _ensure_link_layer

        undecoded = Raw(bytes(self._frame()))       # what scapy actually returns
        assert not undecoded.haslayer(IP)

        recovered = _ensure_link_layer(undecoded)
        assert recovered.haslayer(IP)
        assert recovered[IP].dst == "93.184.216.34"

    def test_recovered_frame_converts_to_an_internal_packet(self):
        from scapy.all import Raw

        from encids.live.capture import _scapy_to_packet

        undecoded = Raw(bytes(self._frame()))
        undecoded.time = 1_700_000_000.0

        pkt = _scapy_to_packet(undecoded)
        assert pkt is not None, "a recoverable frame must not be dropped"
        assert pkt.dst_port == 443
        assert pkt.protocol == 6

    def test_already_decoded_frame_is_returned_unchanged(self):
        from encids.live.capture import _ensure_link_layer

        frame = self._frame()
        assert _ensure_link_layer(frame) is frame

    def test_undecodable_payload_does_not_raise(self):
        from scapy.all import Raw

        from encids.live.capture import _ensure_link_layer

        assert _ensure_link_layer(Raw(b"\x00\x01\x02")) is not None
