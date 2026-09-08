"""Canonical feature schema.

Every dataset loader - ISCX pcaps, CIC-Darknet CSVs, CTU-13 netflows, the
synthetic generator and the live capture pipeline - must emit a DataFrame with
these columns.  Having one schema is what makes it possible to train on offline
datasets and then score *live* traffic with the very same model.

Two families of features, matching the project's "dual-signal" design:

  * FLOW_FEATURES - packet size distributions, inter-arrival timing, duration,
    byte/direction ratios and TCP flag counts.  Observable regardless of
    encryption.
  * TLS_FEATURES  - handshake metadata that is sent *in the clear* before the
    session goes encrypted: JA3/JA3S fingerprints, offered cipher suites,
    extensions, SNI shape, ALPN and certificate-chain length.

Nothing here requires decrypting a single byte of payload.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Identifier columns - kept for display/forensics, NEVER fed to a model.
# Training on IPs or ports leaks the dataset's addressing plan and produces
# models that look excellent offline and fail completely in the wild.
# --------------------------------------------------------------------------
ID_COLUMNS: list[str] = [
    "flow_id",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "first_seen_ms",
    "last_seen_ms",
]

# --------------------------------------------------------------------------
# Stage-A signal: flow-level statistics
# --------------------------------------------------------------------------
FLOW_FEATURES: list[str] = [
    # volume & duration
    "duration_ms",
    "fwd_packets",
    "bwd_packets",
    "total_packets",
    "fwd_bytes",
    "bwd_bytes",
    "total_bytes",
    # packet-size distribution (the "shape" of the conversation)
    "fwd_pkt_len_min",
    "fwd_pkt_len_max",
    "fwd_pkt_len_mean",
    "fwd_pkt_len_std",
    "bwd_pkt_len_min",
    "bwd_pkt_len_max",
    "bwd_pkt_len_mean",
    "bwd_pkt_len_std",
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    # inter-arrival timing (exposes beaconing / automated C2 "rhythm")
    "flow_iat_min",
    "flow_iat_max",
    "flow_iat_mean",
    "flow_iat_std",
    "fwd_iat_mean",
    "fwd_iat_std",
    "bwd_iat_mean",
    "bwd_iat_std",
    # rates
    "flow_bytes_per_s",
    "flow_packets_per_s",
    "fwd_packets_per_s",
    "bwd_packets_per_s",
    # directionality
    "down_up_byte_ratio",
    "down_up_packet_ratio",
    "fwd_bytes_fraction",
    "avg_packet_size",
    "fwd_segment_size_avg",
    "bwd_segment_size_avg",
    # TCP control-flag counts
    "syn_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "ack_count",
    "urg_count",
    # header/window behaviour
    "fwd_header_bytes",
    "bwd_header_bytes",
    "fwd_init_win_bytes",
    "bwd_init_win_bytes",
]

# --------------------------------------------------------------------------
# Stage-B signal: TLS handshake metadata (the project's differentiator)
# --------------------------------------------------------------------------
TLS_FEATURES: list[str] = [
    "is_tls",
    "tls_version",              # numeric: 771 = TLS1.2, 772 = TLS1.3
    "ja3_bucket",               # hashing-trick bucket of the JA3 string
    "ja3s_bucket",              # hashing-trick bucket of the JA3S string
    "ja3_is_known_browser",     # 1 if JA3 matches a common browser fingerprint
    "ja3_rarity",               # 1/(corpus frequency) - rare fingerprints score high
    "n_cipher_suites",
    "n_extensions",
    "n_elliptic_curves",
    "n_ec_point_formats",
    "has_grease",               # GREASE values -> modern browser behaviour
    "has_sni",
    "sni_length",
    "sni_entropy",              # high entropy -> DGA-style / random hostnames
    "sni_digit_ratio",
    "has_alpn",
    "alpn_is_h2",
    # Certificate-derived features stop at the chain length. Certificate
    # validity dates and self-signed detection would need X.509/DER parsing,
    # and in TLS 1.3 the Certificate message is itself encrypted - so those
    # features would be unavailable on the majority of modern traffic and are
    # deliberately not claimed here.
    "cert_chain_len",
    "handshake_duration_ms",
]

TARGET_COLUMN = "label"          # multi-class attack/traffic family
BINARY_TARGET_COLUMN = "is_malicious"   # 0 = benign, 1 = malicious
SOURCE_COLUMN = "source_dataset"

BENIGN_LABEL = "benign"


def feature_columns(use_flow: bool = True, use_tls: bool = True) -> list[str]:
    """The exact model input columns, in a stable, reproducible order."""
    cols: list[str] = []
    if use_flow:
        cols += FLOW_FEATURES
    if use_tls:
        cols += TLS_FEATURES
    if not cols:
        raise ValueError("At least one of use_flow / use_tls must be True")
    return cols


ALL_COLUMNS: list[str] = (
    ID_COLUMNS + FLOW_FEATURES + TLS_FEATURES
    + [TARGET_COLUMN, BINARY_TARGET_COLUMN, SOURCE_COLUMN]
)


def empty_frame():
    """An empty DataFrame carrying the full canonical schema."""
    import pandas as pd

    return pd.DataFrame({c: pd.Series(dtype="object" if c in
                        ("flow_id", "src_ip", "dst_ip", TARGET_COLUMN, SOURCE_COLUMN)
                        else "float64") for c in ALL_COLUMNS})
