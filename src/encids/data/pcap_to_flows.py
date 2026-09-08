"""pcap -> canonical flow-feature table.

Two interchangeable backends:

  * ``nfstream`` - the library named in the project methodology.  Fast (C/nDPI
    core) and gives us application labels for free.
  * ``native``   - the pure-Python dpkt implementation in
    :mod:`encids.features.flow_stats`.  Slower, but has no libpcap dependency
    and produces byte-identical features to the live pipeline.

Both emit exactly the columns in :mod:`encids.features.schema`, so downstream
code never needs to know which one ran.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..features import schema
from ..features.flow_stats import flows_from_pcap
from ..features.tls_features import KNOWN_BROWSER_JA3, ja3_bucket
from ..utils.logging_utils import get_logger

log = get_logger("data.pcap")


def _nfstream_available() -> bool:
    try:
        import nfstream  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# NFStream backend
# ---------------------------------------------------------------------------
def _nfstream_row_to_record(r: Any) -> dict[str, Any]:
    """Map one NFStream bidirectional flow onto the canonical schema."""
    g = lambda name, default=0.0: float(getattr(r, name, default) or 0.0)  # noqa: E731

    fwd_n, bwd_n = g("src2dst_packets"), g("dst2src_packets")
    fwd_bytes, bwd_bytes = g("src2dst_bytes"), g("dst2src_bytes")
    total_packets = g("bidirectional_packets") or (fwd_n + bwd_n)
    total_bytes = g("bidirectional_bytes") or (fwd_bytes + bwd_bytes)
    duration_ms = g("bidirectional_duration_ms")
    duration_s = duration_ms / 1000.0
    ratio = lambda a, b: (a / b) if b else 0.0  # noqa: E731

    # NFStream exposes JA3/JA3S as client_fingerprint / server_fingerprint
    # when nDPI dissection is enabled.
    ja3_hash = str(getattr(r, "client_fingerprint", "") or "")
    ja3s_hash = str(getattr(r, "server_fingerprint", "") or "")
    sni = str(getattr(r, "requested_server_name", "") or "")

    from ..features.tls_features import shannon_entropy

    app = str(getattr(r, "application_name", "") or "")
    is_tls = int(bool(ja3_hash) or "TLS" in app.upper() or "HTTPS" in app.upper()
                 or int(g("dst_port")) == 443)

    return {
        "flow_id": (f"{getattr(r, 'src_ip', '')}:{int(g('src_port'))}-"
                    f"{getattr(r, 'dst_ip', '')}:{int(g('dst_port'))}-"
                    f"{int(g('protocol'))}"),
        "src_ip": getattr(r, "src_ip", ""),
        "dst_ip": getattr(r, "dst_ip", ""),
        "src_port": int(g("src_port")),
        "dst_port": int(g("dst_port")),
        "protocol": int(g("protocol")),
        "first_seen_ms": g("bidirectional_first_seen_ms"),
        "last_seen_ms": g("bidirectional_last_seen_ms"),

        "duration_ms": duration_ms,
        "fwd_packets": fwd_n,
        "bwd_packets": bwd_n,
        "total_packets": total_packets,
        "fwd_bytes": fwd_bytes,
        "bwd_bytes": bwd_bytes,
        "total_bytes": total_bytes,

        "fwd_pkt_len_min": g("src2dst_min_ps"),
        "fwd_pkt_len_max": g("src2dst_max_ps"),
        "fwd_pkt_len_mean": g("src2dst_mean_ps"),
        "fwd_pkt_len_std": g("src2dst_stddev_ps"),
        "bwd_pkt_len_min": g("dst2src_min_ps"),
        "bwd_pkt_len_max": g("dst2src_max_ps"),
        "bwd_pkt_len_mean": g("dst2src_mean_ps"),
        "bwd_pkt_len_std": g("dst2src_stddev_ps"),
        "pkt_len_mean": g("bidirectional_mean_ps"),
        "pkt_len_std": g("bidirectional_stddev_ps"),
        "pkt_len_var": g("bidirectional_stddev_ps") ** 2,

        "flow_iat_min": g("bidirectional_min_piat_ms"),
        "flow_iat_max": g("bidirectional_max_piat_ms"),
        "flow_iat_mean": g("bidirectional_mean_piat_ms"),
        "flow_iat_std": g("bidirectional_stddev_piat_ms"),
        "fwd_iat_mean": g("src2dst_mean_piat_ms"),
        "fwd_iat_std": g("src2dst_stddev_piat_ms"),
        "bwd_iat_mean": g("dst2src_mean_piat_ms"),
        "bwd_iat_std": g("dst2src_stddev_piat_ms"),

        "flow_bytes_per_s": ratio(total_bytes, duration_s),
        "flow_packets_per_s": ratio(total_packets, duration_s),
        "fwd_packets_per_s": ratio(fwd_n, duration_s),
        "bwd_packets_per_s": ratio(bwd_n, duration_s),

        "down_up_byte_ratio": ratio(bwd_bytes, fwd_bytes),
        "down_up_packet_ratio": ratio(bwd_n, fwd_n),
        "fwd_bytes_fraction": ratio(fwd_bytes, total_bytes),
        "avg_packet_size": ratio(total_bytes, total_packets),
        "fwd_segment_size_avg": ratio(fwd_bytes, fwd_n),
        "bwd_segment_size_avg": ratio(bwd_bytes, bwd_n),

        "syn_count": g("bidirectional_syn_packets"),
        "fin_count": g("bidirectional_fin_packets"),
        "rst_count": g("bidirectional_rst_packets"),
        "psh_count": g("bidirectional_psh_packets"),
        "ack_count": g("bidirectional_ack_packets"),
        "urg_count": g("bidirectional_urg_packets"),

        # NFStream does not expose these; the native backend does.
        "fwd_header_bytes": 0.0,
        "bwd_header_bytes": 0.0,
        "fwd_init_win_bytes": 0.0,
        "bwd_init_win_bytes": 0.0,

        "is_tls": is_tls,
        "tls_version": 0.0,
        "ja3_hash": ja3_hash,
        "ja3s_hash": ja3s_hash,
        "ja3_is_known_browser": int(ja3_hash in KNOWN_BROWSER_JA3),
        "n_cipher_suites": 0.0,
        "n_extensions": 0.0,
        "n_elliptic_curves": 0.0,
        "n_ec_point_formats": 0.0,
        "has_grease": 0.0,
        "has_sni": int(bool(sni)),
        "sni": sni,
        "sni_length": len(sni),
        "sni_entropy": round(shannon_entropy(sni), 4),
        "sni_digit_ratio": (round(sum(c.isdigit() for c in sni) / len(sni), 4)
                            if sni else 0.0),
        "has_alpn": 0.0,
        "alpn_is_h2": 0.0,
        "cert_chain_len": 0.0,
        "handshake_duration_ms": 0.0,
        "application_name": app,
    }


def _extract_with_nfstream(pcap: Path, idle: float, active: float) -> list[dict]:
    from nfstream import NFStreamer

    streamer = NFStreamer(
        source=str(pcap),
        decode_tunnels=True,
        statistical_analysis=True,      # packet-size / IAT distributions
        splt_analysis=0,
        n_dissections=20,               # enough to reach the TLS handshake
        idle_timeout=int(idle),
        active_timeout=int(active),
        accounting_mode=0,
    )
    return [_nfstream_row_to_record(flow) for flow in streamer]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def pcap_to_dataframe(
    pcap: str | Path,
    backend: str = "auto",
    idle_timeout: float = 15.0,
    active_timeout: float = 120.0,
) -> pd.DataFrame:
    """Convert a single pcap into a flow-feature DataFrame."""
    pcap = Path(pcap)
    if backend == "auto":
        # Native is the default even when NFStream is installed, and the reason
        # is fingerprint consistency rather than preference.
        #
        # NFStream 6.6 fills `client_fingerprint` with a **JA4** string
        # (`t12d2110h1_f51103c65f73_...`), while the live capture path computes
        # **JA3** with this project's own parser. Training on JA4 and then
        # scoring live traffic described by JA3 means `ja3_bucket` and
        # `ja3_rarity` are drawn from two different fingerprint spaces - the
        # feature would look healthy and carry no transferable signal.
        #
        # Native keeps offline training and live scoring on one scheme. Pass
        # backend="nfstream" explicitly for a faster run when TLS features are
        # not being used (it is several times quicker on large pcaps).
        backend = "native"

    if backend == "nfstream":
        try:
            records = _extract_with_nfstream(pcap, idle_timeout, active_timeout)
        except Exception as exc:
            log.warning("NFStream failed on %s (%s); falling back to native "
                        "backend", pcap.name, exc)
            records = flows_from_pcap(str(pcap), idle_timeout, active_timeout)
    else:
        records = flows_from_pcap(str(pcap), idle_timeout, active_timeout)

    df = pd.DataFrame.from_records(records)
    log.info("%-45s -> %6d flows (%s)", pcap.name, len(df), backend)
    return df


def pcaps_to_dataframe(
    pcaps: Iterable[str | Path],
    label_fn=None,
    backend: str = "auto",
    idle_timeout: float = 15.0,
    active_timeout: float = 120.0,
) -> pd.DataFrame:
    """Convert many pcaps, tagging each with a label derived from its path.

    ``label_fn`` maps a :class:`Path` to a label string; the default marks
    everything benign.
    """
    label_fn = label_fn or (lambda p: schema.BENIGN_LABEL)
    frames: list[pd.DataFrame] = []
    for pcap in pcaps:
        pcap = Path(pcap)
        df = pcap_to_dataframe(pcap, backend, idle_timeout, active_timeout)
        if df.empty:
            continue
        label = label_fn(pcap)
        df[schema.TARGET_COLUMN] = label
        df[schema.BINARY_TARGET_COLUMN] = int(label != schema.BENIGN_LABEL)
        frames.append(df)
    if not frames:
        return schema.empty_frame()
    return pd.concat(frames, ignore_index=True)


def add_fingerprint_buckets(df: pd.DataFrame, n_buckets: int = 256,
                            rarity_from: pd.Series | None = None) -> pd.DataFrame:
    """Turn raw JA3/JA3S hash strings into model-usable numeric features.

    ``ja3_rarity`` is deliberately computed from the *training* corpus and then
    applied unchanged at inference time - recomputing it on live traffic would
    make every fingerprint look rare and destroy the signal.
    """
    df = df.copy()
    for col, out in (("ja3_hash", "ja3_bucket"), ("ja3s_hash", "ja3s_bucket")):
        source = df[col] if col in df.columns else pd.Series([""] * len(df))
        df[out] = source.fillna("").astype(str).map(
            lambda h: ja3_bucket(h, n_buckets)
        )

    counts_source = rarity_from if rarity_from is not None else df.get(
        "ja3_hash", pd.Series([""] * len(df))
    )
    counts = counts_source.fillna("").astype(str).value_counts()
    total = max(int(counts.sum()), 1)
    ja3 = df.get("ja3_hash", pd.Series([""] * len(df))).fillna("").astype(str)
    df["ja3_rarity"] = ja3.map(lambda h: 1.0 / (counts.get(h, 0) / total)
                               if counts.get(h, 0) else 0.0)
    df["ja3_rarity"] = df["ja3_rarity"].clip(upper=1e6)

    return df
