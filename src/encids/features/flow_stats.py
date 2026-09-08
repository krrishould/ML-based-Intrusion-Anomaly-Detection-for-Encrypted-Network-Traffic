"""Native single-pass flow builder.

Reads a pcap (or a stream of already-decoded packets) and aggregates packets
into bidirectional flows, computing every column in the canonical schema -
including the TLS handshake metadata - in one pass.

This exists alongside the NFStream backend for two reasons:

  1. It is the fallback when NFStream / libpcap is unavailable (common on
     Windows without Npcap installed).
  2. NFStream does not expose TCP header-byte counts or initial window sizes,
     and its JA3 support depends on the nDPI build.  Doing our own pass keeps
     the offline and live feature vectors *identical*, which matters: a model
     trained on features computed one way and served features computed another
     way silently degrades.

Only headers are read.  No payload beyond the cleartext TLS handshake is ever
inspected.
"""
from __future__ import annotations

import math
import socket
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from .tls_features import (
    TLSInfo,
    count_certificates,
    parse_client_hello,
    parse_server_hello,
)

TCP_FIN, TCP_SYN, TCP_RST, TCP_PSH, TCP_ACK, TCP_URG = 1, 2, 4, 8, 16, 32


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class Packet:
    """A minimal, protocol-agnostic view of one captured packet."""

    ts: float                 # epoch seconds
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int             # 6 = TCP, 17 = UDP
    length: int               # total IP-layer length
    header_len: int           # IP + transport header bytes
    flags: int = 0            # TCP flags bitmask
    window: int = 0           # TCP window size
    payload: bytes = b""      # transport payload (used only for TLS handshake)


@dataclass
class FlowAccumulator:
    """Running per-flow statistics.  One instance per bidirectional flow."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    first_ts: float
    last_ts: float = 0.0

    fwd_lengths: list[int] = field(default_factory=list)
    bwd_lengths: list[int] = field(default_factory=list)
    all_times: list[float] = field(default_factory=list)
    fwd_times: list[float] = field(default_factory=list)
    bwd_times: list[float] = field(default_factory=list)

    fwd_header_bytes: int = 0
    bwd_header_bytes: int = 0
    fwd_init_win: int = -1
    bwd_init_win: int = -1

    syn: int = 0
    fin: int = 0
    rst: int = 0
    psh: int = 0
    ack: int = 0
    urg: int = 0

    tls: TLSInfo = field(default_factory=TLSInfo)

    # -- ingest ------------------------------------------------------------
    def add(self, pkt: Packet, forward: bool) -> None:
        self.last_ts = pkt.ts
        self.all_times.append(pkt.ts)
        if forward:
            self.fwd_lengths.append(pkt.length)
            self.fwd_times.append(pkt.ts)
            self.fwd_header_bytes += pkt.header_len
            if self.fwd_init_win < 0:
                self.fwd_init_win = pkt.window
        else:
            self.bwd_lengths.append(pkt.length)
            self.bwd_times.append(pkt.ts)
            self.bwd_header_bytes += pkt.header_len
            if self.bwd_init_win < 0:
                self.bwd_init_win = pkt.window

        f = pkt.flags
        self.syn += bool(f & TCP_SYN)
        self.fin += bool(f & TCP_FIN)
        self.rst += bool(f & TCP_RST)
        self.psh += bool(f & TCP_PSH)
        self.ack += bool(f & TCP_ACK)
        self.urg += bool(f & TCP_URG)

        if pkt.payload:
            self._maybe_parse_tls(pkt, forward)

    def _maybe_parse_tls(self, pkt: Packet, forward: bool) -> None:
        """Opportunistically pull JA3/JA3S out of the cleartext handshake."""
        payload = pkt.payload
        if len(payload) < 6 or payload[0] != 0x16:
            return
        if forward and not self.tls.ja3_hash:
            info = parse_client_hello(payload)
            if info is not None:
                info.client_hello_ts = pkt.ts
                # keep any server-side data already collected
                info.ja3s = self.tls.ja3s
                info.ja3s_hash = self.tls.ja3s_hash
                info.cert_chain_len = self.tls.cert_chain_len
                info.server_hello_ts = self.tls.server_hello_ts
                self.tls = info
        elif not forward:
            if not self.tls.ja3s_hash:
                server = parse_server_hello(payload)
                if server is not None:
                    self.tls.is_tls = 1
                    self.tls.ja3s = server["ja3s"]
                    self.tls.ja3s_hash = server["ja3s_hash"]
                    self.tls.server_hello_ts = pkt.ts
            if not self.tls.cert_chain_len:
                n = count_certificates(payload)
                if n:
                    self.tls.cert_chain_len = n

    # -- derive ------------------------------------------------------------
    @staticmethod
    def _iats(times: list[float]) -> list[float]:
        """Inter-arrival times in milliseconds."""
        if len(times) < 2:
            return []
        ordered = sorted(times)
        return [(b - a) * 1000.0 for a, b in zip(ordered, ordered[1:])]

    def to_record(self) -> dict[str, Any]:
        fwd_n, bwd_n = len(self.fwd_lengths), len(self.bwd_lengths)
        fwd_bytes, bwd_bytes = sum(self.fwd_lengths), sum(self.bwd_lengths)
        total_packets = fwd_n + bwd_n
        total_bytes = fwd_bytes + bwd_bytes
        duration_ms = max(0.0, (self.last_ts - self.first_ts) * 1000.0)
        duration_s = duration_ms / 1000.0

        all_lengths = self.fwd_lengths + self.bwd_lengths
        flow_iat = self._iats(self.all_times)
        fwd_iat = self._iats(self.fwd_times)
        bwd_iat = self._iats(self.bwd_times)

        record: dict[str, Any] = {
            # identifiers (display / forensics only - never model inputs)
            "flow_id": (f"{self.src_ip}:{self.src_port}-{self.dst_ip}:"
                        f"{self.dst_port}-{self.protocol}"),
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "first_seen_ms": self.first_ts * 1000.0,
            "last_seen_ms": self.last_ts * 1000.0,

            # volume & duration
            "duration_ms": duration_ms,
            "fwd_packets": fwd_n,
            "bwd_packets": bwd_n,
            "total_packets": total_packets,
            "fwd_bytes": fwd_bytes,
            "bwd_bytes": bwd_bytes,
            "total_bytes": total_bytes,

            # packet-size distribution
            "fwd_pkt_len_min": min(self.fwd_lengths) if fwd_n else 0.0,
            "fwd_pkt_len_max": max(self.fwd_lengths) if fwd_n else 0.0,
            "fwd_pkt_len_mean": _mean(self.fwd_lengths),
            "fwd_pkt_len_std": _std(self.fwd_lengths),
            "bwd_pkt_len_min": min(self.bwd_lengths) if bwd_n else 0.0,
            "bwd_pkt_len_max": max(self.bwd_lengths) if bwd_n else 0.0,
            "bwd_pkt_len_mean": _mean(self.bwd_lengths),
            "bwd_pkt_len_std": _std(self.bwd_lengths),
            "pkt_len_mean": _mean(all_lengths),
            "pkt_len_std": _std(all_lengths),
            "pkt_len_var": _std(all_lengths) ** 2,

            # inter-arrival timing
            "flow_iat_min": min(flow_iat) if flow_iat else 0.0,
            "flow_iat_max": max(flow_iat) if flow_iat else 0.0,
            "flow_iat_mean": _mean(flow_iat),
            "flow_iat_std": _std(flow_iat),
            "fwd_iat_mean": _mean(fwd_iat),
            "fwd_iat_std": _std(fwd_iat),
            "bwd_iat_mean": _mean(bwd_iat),
            "bwd_iat_std": _std(bwd_iat),

            # rates
            "flow_bytes_per_s": _safe_ratio(total_bytes, duration_s),
            "flow_packets_per_s": _safe_ratio(total_packets, duration_s),
            "fwd_packets_per_s": _safe_ratio(fwd_n, duration_s),
            "bwd_packets_per_s": _safe_ratio(bwd_n, duration_s),

            # directionality
            "down_up_byte_ratio": _safe_ratio(bwd_bytes, fwd_bytes),
            "down_up_packet_ratio": _safe_ratio(bwd_n, fwd_n),
            "fwd_bytes_fraction": _safe_ratio(fwd_bytes, total_bytes),
            "avg_packet_size": _safe_ratio(total_bytes, total_packets),
            "fwd_segment_size_avg": _safe_ratio(fwd_bytes, fwd_n),
            "bwd_segment_size_avg": _safe_ratio(bwd_bytes, bwd_n),

            # TCP control flags
            "syn_count": self.syn,
            "fin_count": self.fin,
            "rst_count": self.rst,
            "psh_count": self.psh,
            "ack_count": self.ack,
            "urg_count": self.urg,

            # header / window behaviour
            "fwd_header_bytes": self.fwd_header_bytes,
            "bwd_header_bytes": self.bwd_header_bytes,
            "fwd_init_win_bytes": max(self.fwd_init_win, 0),
            "bwd_init_win_bytes": max(self.bwd_init_win, 0),
        }
        record.update(self.tls.to_features())
        record["sni"] = self.tls.sni
        return record


class FlowTable:
    """Aggregates packets into flows with idle/active timeout expiry."""

    def __init__(self, idle_timeout: float = 15.0, active_timeout: float = 120.0):
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self._flows: dict[tuple, FlowAccumulator] = {}

    @staticmethod
    def _key(pkt: Packet) -> tuple:
        """Direction-independent key, so both halves of a conversation match.

        Sorting the two endpoints is only a way to get a stable dictionary key.
        It deliberately does *not* decide the forward direction - see
        :meth:`add`.
        """
        a = (pkt.src_ip, pkt.src_port)
        b = (pkt.dst_ip, pkt.dst_port)
        return (a, b, pkt.protocol) if a <= b else (b, a, pkt.protocol)

    def add(self, pkt: Packet) -> list[dict[str, Any]]:
        """Add a packet; returns records for any flows that expired."""
        expired = self._expire(pkt.ts)
        key = self._key(pkt)
        flow = self._flows.get(key)

        if flow is None:
            # The endpoint that sent the first packet we saw is the client, and
            # defines the forward direction for the whole flow.  Deriving it
            # from the sorted key instead would assign direction by IP string
            # order, silently swapping upload-heavy and download-heavy flows -
            # which is exactly the signal data-exfiltration detection rests on.
            flow = FlowAccumulator(
                src_ip=pkt.src_ip, dst_ip=pkt.dst_ip,
                src_port=pkt.src_port, dst_port=pkt.dst_port,
                protocol=pkt.protocol, first_ts=pkt.ts,
            )
            self._flows[key] = flow
            forward = True
        else:
            forward = (pkt.src_ip == flow.src_ip and pkt.src_port == flow.src_port)

        flow.add(pkt, forward)
        return expired

    def _expire(self, now: float) -> list[dict[str, Any]]:
        done = []
        for key, flow in list(self._flows.items()):
            idle = now - flow.last_ts
            age = now - flow.first_ts
            if idle > self.idle_timeout or age > self.active_timeout:
                done.append(flow.to_record())
                del self._flows[key]
        return done

    def flush(self) -> list[dict[str, Any]]:
        """Close out every remaining flow (end of capture)."""
        records = [f.to_record() for f in self._flows.values()]
        self._flows.clear()
        return records

    def __len__(self) -> int:
        return len(self._flows)


# ---------------------------------------------------------------------------
# pcap reading
# ---------------------------------------------------------------------------
def iter_pcap_packets(path: str) -> Iterator[Packet]:
    """Yield :class:`Packet` objects from a pcap/pcapng file using dpkt."""
    import dpkt

    opener = dpkt.pcapng.Reader if str(path).endswith("ng") else dpkt.pcap.Reader
    with open(path, "rb") as fh:
        try:
            reader = opener(fh)
        except ValueError:
            fh.seek(0)
            reader = (dpkt.pcap.Reader(fh) if opener is dpkt.pcapng.Reader
                      else dpkt.pcapng.Reader(fh))
        linktype = reader.datalink()

        for ts, buf in reader:
            pkt = _decode(ts, buf, linktype, dpkt)
            if pkt is not None:
                yield pkt


def _decode(ts: float, buf: bytes, linktype: int, dpkt) -> Packet | None:
    """Decode one raw frame down to the transport layer."""
    try:
        if linktype == dpkt.pcap.DLT_EN10MB:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data
        elif linktype in (dpkt.pcap.DLT_RAW, 101, 12):
            ip = dpkt.ip.IP(buf)
        elif linktype == dpkt.pcap.DLT_LINUX_SLL:
            ip = dpkt.sll.SLL(buf).data
        elif linktype == dpkt.pcap.DLT_NULL:
            ip = dpkt.loopback.Loopback(buf).data
        else:
            eth = dpkt.ethernet.Ethernet(buf)
            ip = eth.data

        if isinstance(ip, dpkt.ip.IP):
            src = socket.inet_ntop(socket.AF_INET, ip.src)
            dst = socket.inet_ntop(socket.AF_INET, ip.dst)
            ip_hdr_len = ip.hl * 4
            total_len = ip.len or len(bytes(ip))
        elif isinstance(ip, dpkt.ip6.IP6):
            src = socket.inet_ntop(socket.AF_INET6, ip.src)
            dst = socket.inet_ntop(socket.AF_INET6, ip.dst)
            ip_hdr_len = 40
            total_len = ip.plen + 40
        else:
            return None

        transport = ip.data
        if isinstance(transport, dpkt.tcp.TCP):
            return Packet(
                ts=ts, src_ip=src, dst_ip=dst,
                src_port=transport.sport, dst_port=transport.dport,
                protocol=6, length=total_len,
                header_len=ip_hdr_len + transport.off * 4,
                flags=transport.flags, window=transport.win,
                payload=bytes(transport.data)[:2048],
            )
        if isinstance(transport, dpkt.udp.UDP):
            return Packet(
                ts=ts, src_ip=src, dst_ip=dst,
                src_port=transport.sport, dst_port=transport.dport,
                protocol=17, length=total_len,
                header_len=ip_hdr_len + 8,
            )
        return None
    except Exception:
        # A malformed / truncated frame must never kill a multi-GB capture.
        return None


def flows_from_packets(
    packets: Iterable[Packet],
    idle_timeout: float = 15.0,
    active_timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Aggregate an iterable of packets into completed flow records."""
    table = FlowTable(idle_timeout, active_timeout)
    records: list[dict[str, Any]] = []
    for pkt in packets:
        records.extend(table.add(pkt))
    records.extend(table.flush())
    return records


def flows_from_pcap(
    path: str,
    idle_timeout: float = 15.0,
    active_timeout: float = 120.0,
) -> list[dict[str, Any]]:
    """Read a pcap and return one record per bidirectional flow."""
    return flows_from_packets(iter_pcap_packets(path), idle_timeout, active_timeout)
