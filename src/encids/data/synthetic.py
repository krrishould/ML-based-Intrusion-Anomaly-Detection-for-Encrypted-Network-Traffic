"""Synthetic encrypted-traffic generator.

Purpose
-------
The public datasets (ISCX VPN-nonVPN2016, CIC-Darknet2020, CTU-13) are tens of
gigabytes and sit behind registration forms.  This module produces a
statistically plausible stand-in so that the *entire* pipeline - feature table,
two-stage training, fusion, SHAP, dashboard - is runnable and testable from a
clean checkout, and so the unit tests have deterministic data.

How it works
------------
Rather than fabricating the final feature columns directly (which would produce
mutually inconsistent numbers - a "mean" that does not match its "std" - and
would let a model latch onto that inconsistency), this generator synthesises
per-flow **packet sequences**: a list of sizes and arrival timestamps drawn from
class-conditional distributions.  Those packets are then pushed through exactly
the same :class:`~encids.features.flow_stats.FlowAccumulator` used for real
pcaps, so every derived statistic is internally consistent by construction.

Realism controls
----------------
Cleanly separated synthetic classes would produce a meaningless 100% score, so
three sources of genuine, irreducible error are built in:

* **Evasion / mimicry** - a configurable share of malicious flows have their
  distribution parameters blended toward a randomly chosen benign profile.
  This models what evasive malware actually does: beacon jitter, padding,
  domain fronting and deliberate traffic shaping to look like browsing.  Such a
  flow keeps its malicious label, so it is correctly-labelled-but-hard, not
  label noise.
* **Atypical benign traffic** - a small share of benign flows are drawn with
  inflated variance (a bulk sync, a stalled connection, an odd IoT client).
  These are what a deployed system generates false positives on.
* **Fingerprint overlap** - malicious flows borrow mainstream browser JA3
  hashes at the rate given by each profile, and some benign flows carry rare
  fingerprints from niche clients.  A model that keys purely on JA3 therefore
  cannot win, which is the documented weakness of TLS-fingerprint-only work.

This is scaffolding for development, not evidence.  Reported results must come
from the real datasets; every artefact produced from synthetic data is labelled
as such.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..features import schema
from ..features.flow_stats import FlowAccumulator
from ..features.tls_features import KNOWN_BROWSER_JA3, TLSInfo

BENIGN_CLASSES = ("web_browsing", "streaming", "voip", "file_transfer", "vpn_tunnel")
ATTACK_CLASSES = ("c2_beacon", "data_exfiltration", "tor_darknet", "port_scan",
                  "ddos_flood", "crypto_mining")


@dataclass(frozen=True)
class TrafficProfile:
    """Class-conditional distribution parameters for one traffic family."""

    name: str
    malicious: bool
    n_fwd: tuple[float, float]        # lognormal (mu, sigma) for fwd packet count
    n_bwd_ratio: tuple[float, float]  # normal (mean, std) bwd/fwd packet ratio
    fwd_size: tuple[float, float]     # lognormal (mu, sigma) for fwd packet size
    bwd_size: tuple[float, float]
    iat_ms: tuple[float, float]       # lognormal (mu, sigma) inter-arrival ms
    iat_regularity: float             # 0 = bursty/human, 1 = metronomic/automated
    tls_prob: float
    browser_ja3_prob: float           # P(uses a mainstream browser fingerprint)
    size_quantum: int = 0             # >0 pads sizes to a multiple (Tor cells, VPN MTU)
    evasion: float = 0.0              # how far this instance was blended to benign


# Each malicious family is deliberately placed near a benign "twin" that it is
# genuinely confusable with in the literature, rather than in its own corner of
# feature space:
#
#     c2_beacon         ~ voip / web_browsing   (small, periodic, low volume)
#     data_exfiltration ~ file_transfer         (upload-heavy, large packets)
#     tor_darknet       ~ vpn_tunnel            (padded, uniform packet sizes)
#     crypto_mining     ~ voip                  (long-lived, steady, small)
#     ddos_flood        ~ streaming             (very high packet rate)
#     port_scan         ~ short failed sessions (few packets, no payload)
#
# The separating signal is therefore subtle - timing regularity, direction
# ratio, fingerprint rarity - which is the point of the project.
PROFILES: dict[str, TrafficProfile] = {
    # ---- benign -----------------------------------------------------------
    "web_browsing": TrafficProfile(
        "web_browsing", False, (3.0, 1.3), (2.2, 1.4), (5.4, 1.0), (6.4, 1.1),
        (4.0, 1.9), 0.05, 0.90, 0.82),
    "streaming": TrafficProfile(
        "streaming", False, (4.0, 1.1), (4.2, 2.6), (5.1, 0.9), (7.0, 0.7),
        (2.8, 1.4), 0.30, 0.88, 0.75),
    "voip": TrafficProfile(
        "voip", False, (5.0, 1.0), (1.0, 0.35), (5.0, 0.5), (5.0, 0.5),
        (3.4, 0.8), 0.72, 0.50, 0.32),
    "file_transfer": TrafficProfile(
        "file_transfer", False, (5.2, 1.1), (0.40, 0.30), (7.0, 0.5), (4.6, 1.0),
        (2.2, 1.2), 0.25, 0.72, 0.55),
    "vpn_tunnel": TrafficProfile(
        "vpn_tunnel", False, (4.4, 1.3), (1.4, 0.8), (6.7, 0.55), (6.7, 0.6),
        (3.0, 1.4), 0.22, 0.62, 0.38, size_quantum=64),
    # ---- malicious --------------------------------------------------------
    "c2_beacon": TrafficProfile(          # periodic malware check-in ("rhythm")
        "c2_beacon", True, (2.6, 1.0), (1.20, 0.55), (5.2, 0.8), (5.6, 0.9),
        (6.2, 1.3), 0.62, 0.86, 0.30),
    "data_exfiltration": TrafficProfile(  # sustained upload, inverted ratio
        "data_exfiltration", True, (5.2, 1.0), (0.22, 0.18), (7.0, 0.45), (4.5, 0.9),
        (2.4, 1.1), 0.30, 0.78, 0.34),
    "tor_darknet": TrafficProfile(        # fixed 512-byte cell structure
        "tor_darknet", True, (4.3, 1.2), (1.5, 0.8), (6.30, 0.35), (6.30, 0.40),
        (3.2, 1.3), 0.26, 0.90, 0.22, size_quantum=512),
    "port_scan": TrafficProfile(          # few packets, SYN/RST heavy
        "port_scan", True, (0.9, 0.8), (0.55, 0.45), (4.2, 0.6), (4.2, 0.7),
        (1.8, 1.4), 0.15, 0.20, 0.14),
    "ddos_flood": TrafficProfile(         # very high rate, small packets
        "ddos_flood", True, (5.9, 1.0), (0.18, 0.20), (4.4, 0.7), (4.4, 0.8),
        (0.7, 1.1), 0.42, 0.30, 0.16),
    "crypto_mining": TrafficProfile(      # long-lived stratum session
        "crypto_mining", True, (4.2, 1.0), (1.1, 0.5), (5.6, 0.7), (5.9, 0.8),
        (5.6, 1.2), 0.55, 0.72, 0.26),
}

# Rare non-browser fingerprints, standing in for malware TLS libraries and
# niche clients (curl, python-requests, Go crypto/tls, custom C2 stacks).
_RARE_JA3_POOL = [
    hashlib.md5(f"rare-client-{i}".encode()).hexdigest() for i in range(40)
]

# Browser-like fingerprints.  Deliberately NOT the same set as
# `KNOWN_BROWSER_JA3`, which is what the `ja3_is_known_browser` feature checks
# against.  Drawing from the reference list itself would make that feature a
# near-perfect label proxy in synthetic data - benign traffic would always match
# and the ablation would credit TLS metadata with a discriminative power that
# only exists because the generator and the feature share a lookup table.
#
# The partial overlap below (3 of 12) is the realistic case: a fingerprint
# reference list covers some browser builds and misses others, so a real browser
# often does *not* match it.
_BROWSER_JA3_POOL = sorted(KNOWN_BROWSER_JA3)[:3] + [
    hashlib.md5(f"browser-build-{i}".encode()).hexdigest() for i in range(9)
]


def _lognormal(rng: np.random.Generator, mu: float, sigma: float,
               size: int | None = None) -> Any:
    return rng.lognormal(mean=mu, sigma=sigma, size=size)


def _blend(a: tuple[float, float], b: tuple[float, float],
           w: float) -> tuple[float, float]:
    """Linear interpolation between two distribution parameter pairs."""
    return (a[0] * (1 - w) + b[0] * w, a[1] * (1 - w) + b[1] * w)


def _evade(rng: np.random.Generator, attack: TrafficProfile,
           weight: float) -> TrafficProfile:
    """Blend an attack profile toward a random benign one.

    ``weight`` is how far it moves: 0 leaves the attack untouched, 1 makes it
    statistically indistinguishable from benign traffic.  The flow keeps its
    malicious label either way - this is the hard, correctly-labelled tail that
    stops the benchmark from being trivial.
    """
    cover = PROFILES[str(rng.choice(BENIGN_CLASSES))]
    return TrafficProfile(
        name=attack.name,
        malicious=True,
        n_fwd=_blend(attack.n_fwd, cover.n_fwd, weight),
        n_bwd_ratio=_blend(attack.n_bwd_ratio, cover.n_bwd_ratio, weight),
        fwd_size=_blend(attack.fwd_size, cover.fwd_size, weight),
        bwd_size=_blend(attack.bwd_size, cover.bwd_size, weight),
        iat_ms=_blend(attack.iat_ms, cover.iat_ms, weight),
        # jitter: evasive beacons randomise their check-in interval
        iat_regularity=attack.iat_regularity * (1 - weight),
        tls_prob=attack.tls_prob * (1 - weight) + cover.tls_prob * weight,
        # fingerprint spoofing scales with the evasion effort
        browser_ja3_prob=(attack.browser_ja3_prob * (1 - weight)
                          + cover.browser_ja3_prob * weight),
        # padding is dropped once the attacker is actively shaping traffic
        size_quantum=0 if weight > 0.45 else attack.size_quantum,
        evasion=weight,
    )


def _atypical(rng: np.random.Generator, profile: TrafficProfile
              ) -> TrafficProfile:
    """Inflate the variance of a benign profile - a legitimate but odd session."""
    s = float(rng.uniform(1.6, 2.8))
    return TrafficProfile(
        name=profile.name,
        malicious=False,
        n_fwd=(profile.n_fwd[0] + rng.normal(0, 0.7), profile.n_fwd[1] * s),
        n_bwd_ratio=(profile.n_bwd_ratio[0], profile.n_bwd_ratio[1] * s),
        fwd_size=(profile.fwd_size[0], profile.fwd_size[1] * s),
        bwd_size=(profile.bwd_size[0], profile.bwd_size[1] * s),
        iat_ms=(profile.iat_ms[0] + rng.normal(0, 1.2), profile.iat_ms[1] * s),
        iat_regularity=profile.iat_regularity,
        tls_prob=profile.tls_prob,
        browser_ja3_prob=profile.browser_ja3_prob * 0.5,
        size_quantum=profile.size_quantum,
    )


def _make_tls_info(rng: np.random.Generator, profile: TrafficProfile,
                   dst_port: int) -> TLSInfo:
    """Synthesise a handshake fingerprint consistent with the traffic family."""
    if rng.random() > profile.tls_prob:
        return TLSInfo(is_tls=0)

    browser_like = rng.random() < profile.browser_ja3_prob
    if browser_like:
        ja3 = str(rng.choice(_BROWSER_JA3_POOL))
        n_ciphers = int(rng.integers(13, 20))
        n_ext = int(rng.integers(10, 17))
        n_curves = int(rng.integers(3, 6))
        grease = int(rng.random() < 0.85)
        version = int(rng.choice([771, 772], p=[0.35, 0.65]))
    else:
        ja3 = str(rng.choice(_RARE_JA3_POOL))
        n_ciphers = int(rng.integers(3, 12))
        n_ext = int(rng.integers(2, 9))
        n_curves = int(rng.integers(1, 4))
        grease = int(rng.random() < 0.08)
        version = int(rng.choice([769, 770, 771, 772], p=[0.08, 0.12, 0.6, 0.2]))

    # SNI: benign traffic uses readable hostnames; C2 / darknet traffic is more
    # likely to use high-entropy, digit-heavy or algorithmically generated names.
    if profile.malicious and rng.random() < 0.55:
        sni = "".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"),
                                 size=int(rng.integers(14, 30)))) + ".net"
    elif rng.random() < 0.9:
        sni = str(rng.choice(["cdn.example.com", "api.service.io", "www.site.org",
                              "media.stream.tv", "mail.provider.com",
                              "static.assets.net", "chat.app.com"]))
    else:
        sni = ""

    info = TLSInfo(
        is_tls=1,
        tls_version=version,
        ja3_hash=ja3,
        ja3s_hash=hashlib.md5(f"{ja3}-server-{dst_port}".encode()).hexdigest(),
        n_cipher_suites=n_ciphers,
        n_extensions=n_ext,
        n_elliptic_curves=n_curves,
        n_ec_point_formats=int(rng.integers(1, 3)),
        has_grease=grease,
        sni=sni,
        has_sni=int(bool(sni)),
        alpn=["h2"] if browser_like and rng.random() < 0.9 else [],
        cert_chain_len=int(rng.integers(1, 2)) if profile.malicious
        else int(rng.integers(2, 5)),
    )
    info.has_alpn = int(bool(info.alpn))
    info.alpn_is_h2 = int("h2" in info.alpn)
    info.client_hello_ts = 0.0
    info.server_hello_ts = float(rng.lognormal(3.2, 0.8)) / 1000.0
    return info


def _synth_flow(rng: np.random.Generator, profile: TrafficProfile,
                flow_idx: int, host_jitter: float = 0.30) -> dict[str, Any]:
    """Generate one flow: packet sequence -> FlowAccumulator -> feature record.

    ``host_jitter`` adds a per-flow offset to the profile means, standing in for
    host, link and application variation.  Without it every flow of a family
    sits on an identical distribution and the families separate far too cleanly.
    """
    j = lambda mu: mu + rng.normal(0, host_jitter)  # noqa: E731

    n_fwd = int(np.clip(_lognormal(rng, j(profile.n_fwd[0]), profile.n_fwd[1]),
                        1, 20000))
    ratio = max(0.0, rng.normal(*profile.n_bwd_ratio))
    n_bwd = int(np.clip(n_fwd * ratio, 0, 40000))

    # --- packet sizes ---
    fwd_sizes = np.clip(
        _lognormal(rng, j(profile.fwd_size[0]), profile.fwd_size[1], n_fwd),
        40, 1514)
    bwd_sizes = (np.clip(
        _lognormal(rng, j(profile.bwd_size[0]), profile.bwd_size[1], n_bwd),
        40, 1514) if n_bwd else np.array([]))
    if profile.size_quantum:
        q = profile.size_quantum
        fwd_sizes = np.clip(np.ceil(fwd_sizes / q) * q, 40, 1514)
        if n_bwd:
            bwd_sizes = np.clip(np.ceil(bwd_sizes / q) * q, 40, 1514)

    # --- arrival timing ---
    # `iat_regularity` interpolates between human burstiness (heavy-tailed) and
    # machine periodicity (near-constant) - this is what exposes C2 beaconing.
    total = n_fwd + n_bwd
    base = float(np.exp(j(profile.iat_ms[0])))
    jitter_sigma = profile.iat_ms[1] * (1.0 - profile.iat_regularity)
    gaps = _lognormal(rng, np.log(base), max(jitter_sigma, 0.05), max(total - 1, 1))
    times = np.concatenate([[0.0], np.cumsum(gaps)])[:total] / 1000.0

    # interleave directions, keeping timestamps monotonic
    order = rng.permutation(total)
    is_fwd = np.zeros(total, dtype=bool)
    is_fwd[order[:n_fwd]] = True

    src_port = int(rng.integers(1024, 65535))
    dst_port = int(rng.choice([443, 443, 443, 8443, 993, 9001, 22, 53],
                              p=[0.5, 0.14, 0.1, 0.08, 0.05, 0.05, 0.04, 0.04]))
    src_ip = f"10.0.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    dst_ip = f"{rng.integers(1, 223)}.{rng.integers(0, 255)}." \
             f"{rng.integers(0, 255)}.{rng.integers(1, 254)}"
    start = 1.7e9 + flow_idx * 0.37

    acc = FlowAccumulator(
        src_ip=src_ip, dst_ip=dst_ip, src_port=src_port, dst_port=dst_port,
        protocol=6, first_ts=start,
    )

    # --- TCP connection setup and teardown -------------------------------
    # Every real TCP flow begins with a 3-way handshake and usually ends with a
    # FIN exchange: five 40-byte, payload-free packets. Omitting them was a real
    # train/serve mismatch - synthetic flows then had a minimum packet size of
    # ~150 bytes while every flow read from an actual pcap had a minimum of 40,
    # and a model trained on the former flagged *every* real flow as anomalous.
    rtt = float(rng.lognormal(np.log(18), 0.7)) / 1000.0     # ~18 ms typical
    control: list[tuple[float, bool, int]] = [
        (0.0, True, 40),                    # SYN
        (rtt, False, 40),                   # SYN-ACK
        (rtt * 1.02, True, 40),             # ACK
    ]
    data_offset = rtt * 1.05

    fi = bi = 0
    events: list[tuple[float, bool, int]] = list(control)
    for k in range(total):
        offset = data_offset + float(times[k])
        if is_fwd[k]:
            events.append((offset, True, int(fwd_sizes[fi]))); fi += 1
        else:
            events.append((offset, False, int(bwd_sizes[bi]))); bi += 1

    if events:
        end = max(e[0] for e in events)
        # A port scan never completes: SYN then RST, no teardown.
        if profile.name != "port_scan":
            events.append((end + rtt, True, 40))            # FIN
            events.append((end + rtt * 2, False, 40))       # FIN-ACK

    for offset, forward, size in sorted(events, key=lambda e: e[0]):
        ts = start + offset
        if forward:
            acc.fwd_lengths.append(size)
            acc.fwd_times.append(ts)
            acc.fwd_header_bytes += 40
            if acc.fwd_init_win < 0:
                acc.fwd_init_win = int(rng.choice([8192, 29200, 64240, 65535]))
        else:
            acc.bwd_lengths.append(size)
            acc.bwd_times.append(ts)
            acc.bwd_header_bytes += 40
            if acc.bwd_init_win < 0:
                acc.bwd_init_win = int(rng.choice([8192, 26847, 64240, 65535]))
        acc.all_times.append(ts)
    acc.last_ts = acc.all_times[-1] if acc.all_times else start

    # --- TCP control flags ---
    # Flag shaping is faded out by the evasion level too: an attacker shaping
    # packet timing and sizes is also completing handshakes normally.  Leaving
    # a hard-coded RST signature here would hand the classifier a giveaway that
    # no amount of traffic shaping could hide, which is not realistic.
    e = profile.evasion
    acc.syn = int(rng.integers(1, 3))
    acc.ack = max(0, total - acc.syn)
    acc.psh = int(rng.binomial(max(total, 1), rng.uniform(0.15, 0.4)))
    acc.fin = int(rng.integers(0, 3))
    acc.rst = int(rng.integers(0, 2))
    if profile.name == "port_scan" and rng.random() > e:
        acc.rst = int(rng.integers(1, 4))
        acc.ack = int(rng.integers(0, 2))
        acc.psh = 0
    elif profile.name == "ddos_flood" and rng.random() > e:
        acc.syn = int(max(1, n_fwd * rng.uniform(0.4, 1.0)))
        acc.rst = int(rng.integers(0, 5))
    # Benign traffic also produces resets: aborted loads, idle timeouts, and
    # servers closing keep-alives.  Without this, rst_count alone is a tell.
    elif rng.random() < 0.10:
        acc.rst = int(rng.integers(1, 4))
    acc.urg = int(rng.binomial(max(total, 1), 0.002))

    acc.tls = _make_tls_info(rng, profile, dst_port)
    if acc.tls.is_tls:
        acc.tls.client_hello_ts = start
        acc.tls.server_hello_ts = start + acc.tls.server_hello_ts

    record = acc.to_record()
    record[schema.TARGET_COLUMN] = profile.name
    record[schema.BINARY_TARGET_COLUMN] = int(profile.malicious)
    record[schema.SOURCE_COLUMN] = "synthetic"
    return record


def generate(
    n_flows: int = 60_000,
    attack_fraction: float = 0.28,
    seed: int = 42,
    exclude_classes: tuple[str, ...] = (),
    evasion_rate: float = 0.35,
    atypical_benign_rate: float = 0.08,
) -> pd.DataFrame:
    """Generate a synthetic flow table.

    Parameters
    ----------
    n_flows
        Total number of flows to synthesise.
    attack_fraction
        Share of flows drawn from malicious families.
    seed
        RNG seed - the same seed always yields the same table.
    exclude_classes
        Families to leave out entirely.  Used for the held-out zero-day
        experiment: train Stage 1 without ``c2_beacon``, then check whether the
        unsupervised Stage 2 still catches it.
    evasion_rate
        Share of malicious flows shaped to mimic benign traffic (see module
        docstring).  Set to 0 for a trivially separable dataset - useful only
        for debugging, never for reporting results.
    atypical_benign_rate
        Share of benign flows drawn with inflated variance.
    """
    rng = np.random.default_rng(seed)
    benign = [c for c in BENIGN_CLASSES if c not in exclude_classes]
    attack = [c for c in ATTACK_CLASSES if c not in exclude_classes]
    if not benign or not attack:
        raise ValueError("exclude_classes removed every benign or attack family")

    n_attack = int(n_flows * attack_fraction)
    n_benign = n_flows - n_attack

    # Non-uniform mixes: real traffic is dominated by web browsing, and scans /
    # floods are far more common than careful exfiltration.
    benign_w = np.array([{"web_browsing": 0.46, "streaming": 0.20, "voip": 0.12,
                          "file_transfer": 0.12, "vpn_tunnel": 0.10}[c]
                         for c in benign])
    attack_w = np.array([{"c2_beacon": 0.26, "data_exfiltration": 0.14,
                          "tor_darknet": 0.18, "port_scan": 0.20,
                          "ddos_flood": 0.12, "crypto_mining": 0.10}[c]
                         for c in attack])
    benign_w /= benign_w.sum()
    attack_w /= attack_w.sum()

    choices = list(rng.choice(benign, size=n_benign, p=benign_w)) + \
        list(rng.choice(attack, size=n_attack, p=attack_w))
    rng.shuffle(choices)

    records = []
    n_evaded = n_atypical = 0
    for i, name in enumerate(choices):
        profile = PROFILES[str(name)]
        evasion = 0.0
        if profile.malicious and rng.random() < evasion_rate:
            # Most evasion is partial; full mimicry is rare and expensive.
            evasion = float(rng.beta(2.2, 2.0)) * 0.95
            profile = _evade(rng, profile, evasion)
            n_evaded += 1
        elif not profile.malicious and rng.random() < atypical_benign_rate:
            profile = _atypical(rng, profile)
            n_atypical += 1

        record = _synth_flow(rng, profile, i)
        # Recorded so the evaluation can report performance on the hard tail
        # separately from the easy bulk - an aggregate score alone would hide it.
        record["evasion_level"] = round(evasion, 3)
        record["is_atypical_benign"] = int(
            (not profile.malicious) and profile is not PROFILES[str(name)])
        records.append(record)

    df = pd.DataFrame.from_records(records)
    df[schema.SOURCE_COLUMN] = "synthetic"
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
