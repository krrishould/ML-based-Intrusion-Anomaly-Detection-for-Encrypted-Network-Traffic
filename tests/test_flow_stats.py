"""Tests for flow aggregation and the derived flow statistics."""
from __future__ import annotations

import math

import pytest

from encids.features.flow_stats import (
    FlowAccumulator,
    FlowTable,
    Packet,
    flows_from_packets,
)
from encids.features.schema import FLOW_FEATURES, TLS_FEATURES


def packet(ts: float, src: str = "10.0.0.1", dst: str = "93.184.216.34",
           sport: int = 50000, dport: int = 443, length: int = 500,
           flags: int = 0, payload: bytes = b"") -> Packet:
    return Packet(ts=ts, src_ip=src, dst_ip=dst, src_port=sport, dst_port=dport,
                  protocol=6, length=length, header_len=40, flags=flags,
                  window=64240, payload=payload)


class TestFlowAccumulator:
    def test_counts_and_totals(self):
        acc = FlowAccumulator("10.0.0.1", "1.2.3.4", 1234, 443, 6, first_ts=0.0)
        for i in range(4):
            acc.add(packet(float(i), length=100), forward=True)
        for i in range(2):
            acc.add(packet(float(i) + 0.5, length=200), forward=False)

        record = acc.to_record()
        assert record["fwd_packets"] == 4
        assert record["bwd_packets"] == 2
        assert record["total_packets"] == 6
        assert record["fwd_bytes"] == 400
        assert record["bwd_bytes"] == 400
        assert record["total_bytes"] == 800

    def test_packet_size_statistics(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        for i, size in enumerate([100, 200, 300]):
            acc.add(packet(float(i), length=size), forward=True)

        record = acc.to_record()
        assert record["fwd_pkt_len_min"] == 100
        assert record["fwd_pkt_len_max"] == 300
        assert record["fwd_pkt_len_mean"] == pytest.approx(200.0)
        # sample standard deviation of [100, 200, 300]
        assert record["fwd_pkt_len_std"] == pytest.approx(100.0)

    def test_inter_arrival_times_are_milliseconds(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        for i in range(4):
            acc.add(packet(i * 0.25), forward=True)   # 250 ms apart

        record = acc.to_record()
        assert record["flow_iat_mean"] == pytest.approx(250.0)
        assert record["flow_iat_min"] == pytest.approx(250.0)
        assert record["flow_iat_max"] == pytest.approx(250.0)
        assert record["flow_iat_std"] == pytest.approx(0.0)

    def test_perfectly_regular_timing_has_zero_iat_std(self):
        """This is the signal that exposes beaconing, so it must be exact."""
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        for i in range(10):
            acc.add(packet(i * 60.0), forward=True)   # a 60 s beacon
        assert acc.to_record()["flow_iat_std"] == pytest.approx(0.0, abs=1e-6)

    def test_rates_use_duration(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        acc.add(packet(0.0, length=1000), forward=True)
        acc.add(packet(2.0, length=1000), forward=True)

        record = acc.to_record()
        assert record["duration_ms"] == pytest.approx(2000.0)
        assert record["flow_bytes_per_s"] == pytest.approx(1000.0)
        assert record["flow_packets_per_s"] == pytest.approx(1.0)

    def test_zero_duration_flow_does_not_divide_by_zero(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=5.0)
        acc.add(packet(5.0), forward=True)
        record = acc.to_record()
        assert record["duration_ms"] == 0.0
        assert record["flow_bytes_per_s"] == 0.0
        assert math.isfinite(record["flow_packets_per_s"])

    def test_single_packet_flow_has_zero_variance(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        acc.add(packet(0.0), forward=True)
        record = acc.to_record()
        assert record["fwd_pkt_len_std"] == 0.0
        assert record["flow_iat_mean"] == 0.0

    def test_direction_ratios(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        acc.add(packet(0.0, length=100), forward=True)
        acc.add(packet(1.0, length=900), forward=False)

        record = acc.to_record()
        assert record["down_up_byte_ratio"] == pytest.approx(9.0)
        assert record["fwd_bytes_fraction"] == pytest.approx(0.1)
        assert record["avg_packet_size"] == pytest.approx(500.0)

    def test_tcp_flag_counting(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        acc.add(packet(0.0, flags=0x02), forward=True)              # SYN
        acc.add(packet(1.0, flags=0x12), forward=False)             # SYN+ACK
        acc.add(packet(2.0, flags=0x18), forward=True)              # PSH+ACK
        acc.add(packet(3.0, flags=0x04), forward=False)             # RST

        record = acc.to_record()
        assert record["syn_count"] == 2
        assert record["ack_count"] == 2
        assert record["psh_count"] == 1
        assert record["rst_count"] == 1

    def test_record_contains_the_whole_schema(self):
        acc = FlowAccumulator("a", "b", 1, 2, 6, first_ts=0.0)
        acc.add(packet(0.0), forward=True)
        acc.add(packet(1.0), forward=False)
        record = acc.to_record()

        missing = [c for c in FLOW_FEATURES if c not in record]
        assert not missing, f"missing flow features: {missing}"
        # ja3_bucket / rarity are added later by add_fingerprint_buckets
        derived_later = {"ja3_bucket", "ja3s_bucket", "ja3_rarity"}
        missing_tls = [c for c in TLS_FEATURES
                       if c not in record and c not in derived_later]
        assert not missing_tls, f"missing TLS features: {missing_tls}"


class TestFlowTable:
    def test_groups_both_directions_into_one_flow(self):
        table = FlowTable(idle_timeout=100, active_timeout=1000)
        table.add(packet(0.0, src="10.0.0.1", dst="1.2.3.4",
                         sport=5000, dport=443))
        table.add(packet(0.5, src="1.2.3.4", dst="10.0.0.1",
                         sport=443, dport=5000))

        records = table.flush()
        assert len(records) == 1
        assert records[0]["fwd_packets"] == 1
        assert records[0]["bwd_packets"] == 1

    def test_separate_connections_stay_separate(self):
        table = FlowTable(idle_timeout=100, active_timeout=1000)
        table.add(packet(0.0, sport=5000))
        table.add(packet(0.1, sport=5001))
        assert len(table.flush()) == 2

    def test_idle_timeout_closes_a_flow(self):
        table = FlowTable(idle_timeout=10, active_timeout=1000)
        table.add(packet(0.0, sport=5000))
        expired = table.add(packet(50.0, sport=6000))    # 50 s later
        assert len(expired) == 1
        assert expired[0]["src_port"] in (5000, 443)

    def test_active_timeout_closes_a_long_lived_flow(self):
        table = FlowTable(idle_timeout=1000, active_timeout=30)
        for i in range(5):
            expired = table.add(packet(i * 10.0, sport=5000))
        assert expired, "a flow older than the active timeout must be emitted"

    def test_flush_empties_the_table(self):
        table = FlowTable()
        table.add(packet(0.0))
        assert len(table) == 1
        table.flush()
        assert len(table) == 0

    def test_direction_is_fixed_by_the_first_packet_seen(self):
        """Whoever spoke first is 'forward', regardless of arrival order."""
        table = FlowTable(idle_timeout=100)
        table.add(packet(0.0, src="10.0.0.1", dst="1.2.3.4",
                         sport=5000, dport=443, length=100))
        table.add(packet(1.0, src="1.2.3.4", dst="10.0.0.1",
                         sport=443, dport=5000, length=900))
        record = table.flush()[0]
        assert record["fwd_bytes"] == 100
        assert record["bwd_bytes"] == 900


class TestFlowsFromPackets:
    def test_end_to_end_aggregation(self):
        packets = [packet(i * 0.1, sport=5000 + (i % 3)) for i in range(30)]
        records = flows_from_packets(packets)
        assert len(records) == 3
        assert sum(r["total_packets"] for r in records) == 30

    def test_empty_input_yields_no_flows(self):
        assert flows_from_packets([]) == []
