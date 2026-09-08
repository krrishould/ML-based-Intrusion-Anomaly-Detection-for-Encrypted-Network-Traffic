"""Loaders for the three public datasets named in the project proposal.

  * **ISCX VPN-nonVPN 2016** - raw pcaps.  Labelled from the filename
    (``vpn_youtube_A.pcap`` -> vpn/streaming).  Goes through NFStream.
  * **CIC-Darknet2020** - pre-extracted CICFlowMeter CSVs.  Column names drift
    between CICFlowMeter versions, so names are normalised before mapping.
  * **CTU-13** - Argus ``.binetflow`` netflow records.  Coarser than the other
    two (no per-packet size distribution), so unavailable columns are left NaN
    and imputed downstream rather than silently zero-filled - a zero would be a
    *claim* about the traffic, NaN is an admission we do not know.

Every loader returns the canonical schema of :mod:`encids.features.schema`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import schema
from ..utils.logging_utils import get_logger
from .pcap_to_flows import pcaps_to_dataframe

log = get_logger("data.loaders")

# ---------------------------------------------------------------------------
# ISCX VPN-nonVPN 2016
# ---------------------------------------------------------------------------
# ISCX filenames are inconsistent about separators - the same application
# appears as `facebook_chat_4a.pcap` and `facebookchat1.pcapng` - so both the
# keys and the filename are stripped of separators before matching. Matching on
# the literal strings silently sent every `facebookchat*` capture to the
# fallback category.
#
# Longer keys are checked first so `facebook_video` wins over a bare `facebook`
# style prefix and `skype_file` is never shadowed by `skype_chat`.
_ISCX_CATEGORIES = {
    "aim": "chat", "icq": "chat", "facebook_chat": "chat", "hangouts_chat": "chat",
    "skype_chat": "chat", "email": "email", "gmail": "email",
    "ftps": "file_transfer", "sftp": "file_transfer", "scp": "file_transfer",
    "skype_file": "file_transfer", "netflix": "streaming", "youtube": "streaming",
    "vimeo": "streaming", "spotify": "streaming", "facebook_video": "streaming",
    "skype_video": "streaming", "hangouts_video": "streaming",
    "hangouts_audio": "voip", "skype_audio": "voip", "voipbuster": "voip",
    "facebook_audio": "voip", "torrent": "p2p", "bittorrent": "p2p",
}

_ISCX_DEFAULT_CATEGORY = "web_browsing"


def _normalise_name(text: str) -> str:
    """Lowercase and drop separators, so `facebookchat1` == `facebook_chat`."""
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _iscx_label(path: Path, warn_on_fallback: bool = False) -> str:
    """Derive a traffic-category label from an ISCX pcap filename."""
    name = _normalise_name(path.stem)
    # `nonvpn_*` must not be read as tunnelled just because it contains "vpn".
    tunnelled = name.startswith("vpn")

    category = None
    for key in sorted(_ISCX_CATEGORIES, key=len, reverse=True):
        if _normalise_name(key) in name:
            category = _ISCX_CATEGORIES[key]
            break

    if category is None:
        category = _ISCX_DEFAULT_CATEGORY
        if warn_on_fallback:
            # A silent default is how 7 of 23 captures previously ended up
            # labelled `web_browsing` when they were video and chat traffic.
            log.warning("ISCX: '%s' matched no known application - labelling it "
                        "'%s'. Add a key to _ISCX_CATEGORIES if that is wrong.",
                        path.name, category)

    return f"vpn_{category}" if tunnelled else category


# Local discovery / housekeeping services. A capture made on a Windows host
# contains far more of this than of the application being recorded - in
# NonVPN-PCAPs-01, LLMNR alone accounted for 39,435 of 48,620 flows (81%).
# Because ISCX labels come from the *filename*, every one of those broadcast
# flows would be labelled with the recorded application ("voip", "chat", ...),
# which is simply false and would dominate training.
_ISCX_NOISE_PORTS = {
    5355,           # LLMNR
    5353,           # mDNS
    137, 138, 139,  # NetBIOS name/datagram/session
    1900,           # SSDP / UPnP
    67, 68,         # DHCP
    546, 547,       # DHCPv6
    123,            # NTP
    17500,          # Dropbox LAN sync
}


def _is_broadcast_or_multicast(ip: str) -> bool:
    """True for multicast / broadcast / link-local destinations."""
    text = str(ip)
    if text in ("255.255.255.255", ""):
        return True
    if ":" in text:                                  # IPv6
        low = text.lower()
        return low.startswith("ff") or low.startswith("fe80")
    head = text.split(".")[0]
    if not head.isdigit():
        return False
    first = int(head)
    return 224 <= first <= 239 or first == 255       # multicast / broadcast


def _drop_iscx_background(df: pd.DataFrame) -> pd.DataFrame:
    """Remove local-network chatter that the filename label does not describe.

    Without this the labels are wrong for the overwhelming majority of rows.
    DNS (port 53) is kept: it is genuine outbound traffic generated by the
    recorded application, unlike LLMNR/NetBIOS broadcast noise.
    """
    if df.empty:
        return df

    before = len(df)
    dst_port = pd.to_numeric(df.get("dst_port"), errors="coerce")
    src_port = pd.to_numeric(df.get("src_port"), errors="coerce")

    noise = dst_port.isin(_ISCX_NOISE_PORTS) | src_port.isin(_ISCX_NOISE_PORTS)
    if "dst_ip" in df.columns:
        noise |= df["dst_ip"].map(_is_broadcast_or_multicast)

    kept = df[~noise.fillna(False)].reset_index(drop=True)
    log.info("ISCX: dropped %d/%d background flows (LLMNR, NetBIOS, mDNS, "
             "DHCP, SSDP, broadcast) - %d application flows remain",
             before - len(kept), before, len(kept))
    return kept


def load_iscx_vpn2016(root: str | Path, backend: str = "auto",
                      limit: int | None = None) -> pd.DataFrame:
    """Load ISCX VPN-nonVPN 2016 pcaps into flow records.

    Note: this dataset labels *traffic type*, not attacks.  It is used here for
    the encrypted/VPN-tunnel traffic-discrimination task and as benign
    background; it contributes no malicious labels on its own.

    Labels come from the filename, so local-network background traffic present
    in every capture is filtered out first - see :func:`_drop_iscx_background`.
    """
    root = Path(root)
    pcaps = sorted([p for ext in ("*.pcap", "*.pcapng")
                    for p in root.rglob(ext)])
    if not pcaps:
        log.warning("No pcaps under %s - skipping ISCX VPN-nonVPN2016", root)
        return schema.empty_frame()
    if limit:
        pcaps = pcaps[:limit]

    log.info("ISCX VPN-nonVPN2016: %d pcap file(s)", len(pcaps))
    df = pcaps_to_dataframe(
        pcaps, label_fn=lambda p: _iscx_label(p, warn_on_fallback=True),
        backend=backend)
    if df.empty:
        return df
    df = _drop_iscx_background(df)
    if df.empty:
        return df
    # Traffic-type dataset: nothing here is an attack.
    df[schema.BINARY_TARGET_COLUMN] = 0
    df[schema.SOURCE_COLUMN] = "iscx_vpn2016"
    return df


# ---------------------------------------------------------------------------
# CIC-Darknet2020  (CICFlowMeter CSV)
# ---------------------------------------------------------------------------
def _normalise(col: str) -> str:
    """CICFlowMeter column name -> lowercase snake_case, version-agnostic."""
    c = col.strip().lower()
    c = c.replace("/s", "_per_s").replace("/", "_")
    c = re.sub(r"[^a-z0-9]+", "_", c).strip("_")
    # CICFlowMeter renamed several columns between releases
    aliases = {
        "total_fwd_packet": "total_fwd_packets",
        "total_bwd_packets": "total_backward_packets",
        "total_length_of_fwd_packet": "total_length_of_fwd_packets",
        "total_length_of_bwd_packet": "total_length_of_bwd_packets",
        "packet_length_mean": "packet_length_mean",
        "fwd_init_win_bytes": "init_win_bytes_forward",
        "bwd_init_win_bytes": "init_win_bytes_backward",
        "fwd_seg_size_avg": "avg_fwd_segment_size",
        "bwd_seg_size_avg": "avg_bwd_segment_size",
        "fwd_header_length": "fwd_header_length",
        "cwe_flag_count": "cwr_flag_count",
    }
    return aliases.get(c, c)


# canonical column  <-  candidate CICFlowMeter names (first match wins)
_CIC_MAP: dict[str, tuple[str, ...]] = {
    "src_ip": ("src_ip", "source_ip"),
    "dst_ip": ("dst_ip", "destination_ip"),
    "src_port": ("src_port", "source_port"),
    "dst_port": ("dst_port", "destination_port"),
    "protocol": ("protocol",),
    "duration_ms": ("flow_duration",),
    "fwd_packets": ("total_fwd_packets",),
    "bwd_packets": ("total_backward_packets",),
    "fwd_bytes": ("total_length_of_fwd_packets",),
    "bwd_bytes": ("total_length_of_bwd_packets",),
    "fwd_pkt_len_min": ("fwd_packet_length_min",),
    "fwd_pkt_len_max": ("fwd_packet_length_max",),
    "fwd_pkt_len_mean": ("fwd_packet_length_mean",),
    "fwd_pkt_len_std": ("fwd_packet_length_std",),
    "bwd_pkt_len_min": ("bwd_packet_length_min",),
    "bwd_pkt_len_max": ("bwd_packet_length_max",),
    "bwd_pkt_len_mean": ("bwd_packet_length_mean",),
    "bwd_pkt_len_std": ("bwd_packet_length_std",),
    "pkt_len_mean": ("packet_length_mean",),
    "pkt_len_std": ("packet_length_std",),
    "pkt_len_var": ("packet_length_variance",),
    "flow_iat_min": ("flow_iat_min",),
    "flow_iat_max": ("flow_iat_max",),
    "flow_iat_mean": ("flow_iat_mean",),
    "flow_iat_std": ("flow_iat_std",),
    "fwd_iat_mean": ("fwd_iat_mean",),
    "fwd_iat_std": ("fwd_iat_std",),
    "bwd_iat_mean": ("bwd_iat_mean",),
    "bwd_iat_std": ("bwd_iat_std",),
    "flow_bytes_per_s": ("flow_bytes_per_s",),
    "flow_packets_per_s": ("flow_packets_per_s",),
    "fwd_packets_per_s": ("fwd_packets_per_s",),
    "bwd_packets_per_s": ("bwd_packets_per_s",),
    "down_up_byte_ratio": ("down_up_ratio",),
    "avg_packet_size": ("average_packet_size", "avg_packet_size", "packet_length_mean"),
    "fwd_segment_size_avg": ("avg_fwd_segment_size",),
    "bwd_segment_size_avg": ("avg_bwd_segment_size",),
    "syn_count": ("syn_flag_count",),
    "fin_count": ("fin_flag_count",),
    "rst_count": ("rst_flag_count",),
    "psh_count": ("psh_flag_count",),
    "ack_count": ("ack_flag_count",),
    "urg_count": ("urg_flag_count",),
    "fwd_header_bytes": ("fwd_header_length",),
    "bwd_header_bytes": ("bwd_header_length",),
    "fwd_init_win_bytes": ("init_win_bytes_forward",),
    "bwd_init_win_bytes": ("init_win_bytes_backward",),
}

# CIC-Darknet2020's positive class is "darknet" = Tor + VPN, per the dataset's
# own design (Label is one of Tor / Non-Tor / VPN / NonVPN).
#
# READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SOURCE:
# the positive class here means "anonymised / tunnelled traffic", NOT "attack".
# CIC-Darknet2020 contains no attacks at all - neither does ISCX VPN-nonVPN2016.
# CTU-13 is the only source in this project with genuinely malicious traffic.
# A high score on this dataset says the model can tell Tor/VPN from ordinary
# traffic; it says nothing about intrusion detection, and reporting it as
# attack-detection performance would be wrong. It is also a debatable target on
# its own terms - VPN use is legitimate, and a deployed detector that alerted on
# every VPN user would be useless.
#
# The dataset convention is kept anyway, so results stay comparable to published
# work on it; the framing is what has to be accurate.
_DARKNET_MALICIOUS = {"tor", "vpn", "darknet"}


def load_cic_darknet2020(root: str | Path) -> pd.DataFrame:
    """Load CIC-Darknet2020 CICFlowMeter CSVs."""
    root = Path(root)
    csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        log.warning("No CSVs under %s - skipping CIC-Darknet2020", root)
        return schema.empty_frame()

    frames = []
    for path in csvs:
        raw = pd.read_csv(path, low_memory=False, encoding_errors="ignore")
        raw.columns = [_normalise(c) for c in raw.columns]
        df = _map_columns(raw, _CIC_MAP)

        # Label columns: 'label' = Tor/VPN/Non-Tor/NonVPN, 'label_1' = app category
        primary = raw.get("label", pd.Series([""] * len(raw))).astype(str).str.lower()
        secondary = raw.get("label_1", raw.get("label1", pd.Series([""] * len(raw))))
        secondary = secondary.astype(str).str.lower().str.replace(r"\W+", "_",
                                                                  regex=True)
        malicious = primary.str.strip().isin(_DARKNET_MALICIOUS).astype(int)
        df[schema.TARGET_COLUMN] = np.where(
            malicious == 1,
            primary.str.strip() + "_" + secondary.str.strip(),
            secondary.str.strip().replace("", schema.BENIGN_LABEL),
        )
        df[schema.BINARY_TARGET_COLUMN] = malicious
        df[schema.SOURCE_COLUMN] = "cic_darknet2020"
        frames.append(df)
        log.info("%-45s -> %6d flows", path.name, len(df))

    out = pd.concat(frames, ignore_index=True)
    return _finalise(out)


def _map_columns(raw: pd.DataFrame, mapping: dict[str, tuple[str, ...]]
                 ) -> pd.DataFrame:
    """Project a source DataFrame onto the canonical schema."""
    out = pd.DataFrame(index=raw.index)
    for canonical, candidates in mapping.items():
        for cand in candidates:
            if cand in raw.columns:
                out[canonical] = pd.to_numeric(raw[cand], errors="coerce") \
                    if canonical not in ("src_ip", "dst_ip") else raw[cand]
                break
    return out


# ---------------------------------------------------------------------------
# CTU-13  (Argus binetflow)
# ---------------------------------------------------------------------------
# CTU-13 labels its botnet flows by capture version (``From-Botnet-V47-...``).
# Mapping those to the actual malware family gives ~7 meaningful classes instead
# of ~100 hyper-specific ones like "botnet-v47-tcp-attempt-spam", most of which
# would have too few samples to train or evaluate on.
# Source: Garcia et al., "An empirical comparison of botnet detection methods",
# Computers & Security 45 (2014) - Table 3.
_CTU13_MALWARE: dict[str, str] = {
    "42": "neris", "43": "neris", "50": "neris",
    "44": "rbot", "45": "rbot", "51": "rbot", "52": "rbot",
    "46": "virut", "54": "virut",
    "47": "menti",
    "48": "sogou",
    "49": "murlo",
    "53": "nsis_ay",
}


def _ctu13_family(labels: pd.Series) -> pd.Series:
    """``flow=From-Botnet-V47-TCP-Attempt-SPAM`` -> ``botnet_menti``."""
    version = labels.str.extract(r"[Bb]otnet-?[Vv](\d+)", expand=False)
    family = version.map(_CTU13_MALWARE).fillna("unknown")
    return "botnet_" + family


def load_ctu13(root: str | Path, chunksize: int | None = 500_000,
               max_rows_per_file: int | None = 1_500_000) -> pd.DataFrame:
    """Load CTU-13 netflow files (``.binetflow`` and ``.binetflow.2format``).

    CTU-13 is netflow-level: it carries totals but no per-packet size or IAT
    distributions.  Those columns are returned as NaN so the imputer can handle
    them honestly, and so a mixed-source training run can be told what is real.

    The ``.2format`` variant published on the CTU server is richer - it has true
    directional packet and byte counts (``SrcPkts``/``DstPkts``) plus TCP window
    sizes - so those are used directly when present instead of being apportioned.

    These files reach 1 GB, so they are read in chunks and capped at
    ``max_rows_per_file`` rows (the captures are dominated by background
    traffic; the botnet flows are not concentrated at the end).
    """
    root = Path(root)
    files = sorted(set(list(root.rglob("*.binetflow"))
                       + list(root.rglob("*.binetflow.2format"))
                       + list(root.rglob("*.netflow"))))
    if not files:
        log.warning("No binetflow files under %s - skipping CTU-13", root)
        return schema.empty_frame()

    frames = []
    for path in files:
        raw = _read_netflow(path, chunksize, max_rows_per_file)
        raw.columns = [c.strip() for c in raw.columns]

        dur_s = pd.to_numeric(raw.get("Dur"), errors="coerce").fillna(0.0)
        tot_pkts = pd.to_numeric(raw.get("TotPkts"), errors="coerce").fillna(0.0)
        tot_bytes = pd.to_numeric(raw.get("TotBytes"), errors="coerce").fillna(0.0)
        src_bytes = pd.to_numeric(raw.get("SrcBytes"), errors="coerce").fillna(0.0)

        if "DstBytes" in raw.columns:
            dst_bytes = pd.to_numeric(raw["DstBytes"], errors="coerce").fillna(0.0)
        else:
            dst_bytes = (tot_bytes - src_bytes).clip(lower=0)

        if "SrcPkts" in raw.columns and "DstPkts" in raw.columns:
            # .2format: real directional counts, no estimation needed.
            fwd_pkts = pd.to_numeric(raw["SrcPkts"], errors="coerce").fillna(0.0)
            bwd_pkts = pd.to_numeric(raw["DstPkts"], errors="coerce").fillna(0.0)
        else:
            # Classic binetflow gives only a total; apportion it by bytes.
            share_est = np.where(tot_bytes > 0,
                                 src_bytes / tot_bytes.replace(0, np.nan), 0.5)
            share_est = pd.Series(share_est, index=raw.index).fillna(0.5)
            fwd_pkts = (tot_pkts * share_est).round()
            bwd_pkts = (tot_pkts - fwd_pkts).clip(lower=0)

        share = pd.Series(
            np.where(tot_bytes > 0, src_bytes / tot_bytes.replace(0, np.nan), 0.5),
            index=raw.index).fillna(0.5)

        df = pd.DataFrame({
            "src_ip": raw.get("SrcAddr"),
            "dst_ip": raw.get("DstAddr"),
            "src_port": pd.to_numeric(raw.get("Sport"), errors="coerce"),
            "dst_port": pd.to_numeric(raw.get("Dport"), errors="coerce"),
            "protocol": raw.get("Proto", "").astype(str).str.lower().map(
                {"tcp": 6, "udp": 17, "icmp": 1}).fillna(0),
            "duration_ms": dur_s * 1000.0,
            "fwd_packets": fwd_pkts,
            "bwd_packets": bwd_pkts,
            "total_packets": tot_pkts,
            "fwd_bytes": src_bytes,
            "bwd_bytes": dst_bytes,
            "total_bytes": tot_bytes,
            "flow_bytes_per_s": np.where(dur_s > 0, tot_bytes / dur_s, 0.0),
            "flow_packets_per_s": np.where(dur_s > 0, tot_pkts / dur_s, 0.0),
            "fwd_packets_per_s": np.where(dur_s > 0, fwd_pkts / dur_s, 0.0),
            "bwd_packets_per_s": np.where(dur_s > 0, bwd_pkts / dur_s, 0.0),
            "down_up_byte_ratio": np.where(src_bytes > 0, dst_bytes / src_bytes, 0.0),
            "down_up_packet_ratio": np.where(fwd_pkts > 0, bwd_pkts / fwd_pkts, 0.0),
            "fwd_bytes_fraction": share,
            "avg_packet_size": np.where(tot_pkts > 0, tot_bytes / tot_pkts, 0.0),
            "fwd_segment_size_avg": np.where(fwd_pkts > 0, src_bytes / fwd_pkts, 0.0),
            "bwd_segment_size_avg": np.where(bwd_pkts > 0, dst_bytes / bwd_pkts, 0.0),
        })

        # .2format also carries TCP window sizes, which map onto the schema.
        if "SrcWin" in raw.columns:
            df["fwd_init_win_bytes"] = pd.to_numeric(raw["SrcWin"],
                                                     errors="coerce").fillna(0)
        if "DstWin" in raw.columns:
            df["bwd_init_win_bytes"] = pd.to_numeric(raw["DstWin"],
                                                     errors="coerce").fillna(0)
        # Average packet size per direction is a genuine size statistic; the
        # min/max/std of the distribution remain unavailable and stay NaN.
        df["fwd_pkt_len_mean"] = df["fwd_segment_size_avg"]
        df["bwd_pkt_len_mean"] = df["bwd_segment_size_avg"]
        df["pkt_len_mean"] = df["avg_packet_size"]

        label_raw = raw.get("Label", pd.Series([""] * len(raw))).astype(str)
        is_botnet = label_raw.str.contains("botnet", case=False, na=False)
        df[schema.TARGET_COLUMN] = np.where(
            is_botnet, _ctu13_family(label_raw), schema.BENIGN_LABEL)
        df[schema.BINARY_TARGET_COLUMN] = is_botnet.astype(int)
        df[schema.SOURCE_COLUMN] = "ctu13"
        frames.append(df)
        log.info("%-45s -> %6d flows (%d botnet)", path.name, len(df),
                 int(is_botnet.sum()))

    return _finalise(pd.concat(frames, ignore_index=True))


def _read_netflow(path: Path, chunksize: int | None,
                  max_rows: int | None) -> pd.DataFrame:
    """Read a large netflow CSV in chunks, keeping every botnet row.

    Truncating a capture by simply taking the first N rows would bias the class
    balance, because the botnet activity is not uniformly distributed in time.
    Instead every malicious row is retained and only background traffic is
    sub-sampled to reach the row budget.
    """
    if chunksize is None:
        return pd.read_csv(path, low_memory=False, encoding_errors="ignore")

    kept: list[pd.DataFrame] = []
    rows = 0
    rng = np.random.default_rng(42)
    reader = pd.read_csv(path, low_memory=False, encoding_errors="ignore",
                         chunksize=chunksize)
    for chunk in reader:
        label_col = next((c for c in chunk.columns if c.strip() == "Label"), None)
        if label_col is None or max_rows is None:
            kept.append(chunk)
        else:
            is_bot = chunk[label_col].astype(str).str.contains("otnet", na=False)
            budget = max(max_rows - rows, 0)
            background = chunk[~is_bot]
            take = min(len(background), max(budget - int(is_bot.sum()), 0))
            if take < len(background):
                background = background.iloc[
                    rng.choice(len(background), take, replace=False)]
            kept.append(pd.concat([chunk[is_bot], background]))
        rows += len(kept[-1])
        if max_rows is not None and rows >= max_rows:
            log.info("%s: capped at %d rows (all botnet flows kept)",
                     path.name, rows)
            break

    return (pd.concat(kept, ignore_index=True) if kept
            else pd.DataFrame())


# ---------------------------------------------------------------------------
def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    """Fill in derived columns and guarantee the full canonical schema."""
    if df.empty:
        return df
    df = df.copy()
    if "total_packets" not in df or df["total_packets"].isna().all():
        df["total_packets"] = df.get("fwd_packets", 0) + df.get("bwd_packets", 0)
    if "total_bytes" not in df or df["total_bytes"].isna().all():
        df["total_bytes"] = df.get("fwd_bytes", 0) + df.get("bwd_bytes", 0)
    if "down_up_packet_ratio" not in df:
        fwd = df.get("fwd_packets", pd.Series(0, index=df.index))
        df["down_up_packet_ratio"] = np.where(
            fwd > 0, df.get("bwd_packets", 0) / fwd.replace(0, np.nan), 0.0)
    if "fwd_bytes_fraction" not in df:
        tot = df.get("total_bytes", pd.Series(0, index=df.index))
        df["fwd_bytes_fraction"] = np.where(
            tot > 0, df.get("fwd_bytes", 0) / tot.replace(0, np.nan), 0.0)

    if "flow_id" not in df:
        df["flow_id"] = (df.get("src_ip", "").astype(str) + ":"
                         + df.get("src_port", 0).astype(str) + "-"
                         + df.get("dst_ip", "").astype(str) + ":"
                         + df.get("dst_port", 0).astype(str))

    # Columns this source genuinely cannot provide stay NaN (see module docstring).
    for col in schema.FLOW_FEATURES + schema.TLS_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for col in ("ja3_hash", "ja3s_hash", "sni"):
        if col not in df.columns:
            df[col] = ""
    # Binary TLS indicators are safe to default: absence of evidence of TLS.
    for col in ("is_tls", "has_sni", "has_alpn", "alpn_is_h2", "has_grease",
                "ja3_is_known_browser"):
        df[col] = df[col].fillna(0)
    return df


# ---------------------------------------------------------------------------
LOADERS = {
    "iscx_vpn2016": load_iscx_vpn2016,
    "cic_darknet2020": load_cic_darknet2020,
    "ctu13": load_ctu13,
}
