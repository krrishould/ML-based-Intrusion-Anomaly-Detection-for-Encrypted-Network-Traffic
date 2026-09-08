"""TLS handshake metadata extraction -> JA3 / JA3S fingerprints.

The TLS handshake happens *before* the session is encrypted, so the
ClientHello and ServerHello are readable on the wire even though everything
after them is not.  This module parses those two messages straight out of the
raw TCP payload bytes and derives:

    JA3  = MD5(TLSVersion,CipherSuites,Extensions,EllipticCurves,ECPointFormats)
    JA3S = MD5(TLSVersion,CipherSuite,Extensions)

plus a set of derived numeric features (cipher/extension counts, SNI shape,
ALPN, GREASE presence).  This is the second signal of the dual-signal design
and costs almost nothing to compute.

Reference: Althouse et al., "TLS Fingerprinting with JA3 and JA3S" (Salesforce).
"""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import asdict, dataclass, field
from typing import Any

# GREASE values (RFC 8701).  Browsers inject these deliberately; malware
# libraries usually do not.  They must be stripped before hashing, and their
# mere presence is itself a useful feature.
GREASE = {
    0x0A0A, 0x1A1A, 0x2A2A, 0x3A3A, 0x4A4A, 0x5A5A, 0x6A6A, 0x7A7A,
    0x8A8A, 0x9A9A, 0xAAAA, 0xBABA, 0xCACA, 0xDADA, 0xEAEA, 0xFAFA,
}

# Extension type ids we care about
EXT_SERVER_NAME = 0x0000
EXT_SUPPORTED_GROUPS = 0x000A
EXT_EC_POINT_FORMATS = 0x000B
EXT_ALPN = 0x0010

# TLS record / handshake type constants
RECORD_HANDSHAKE = 0x16
HS_CLIENT_HELLO = 0x01
HS_SERVER_HELLO = 0x02
HS_CERTIFICATE = 0x0B

# A small reference set of JA3 hashes attributed to mainstream browsers.
#
# LIMITATIONS - read before relying on `ja3_is_known_browser`:
#   * This is a placeholder list, not a maintained fingerprint database. Real
#     deployments need a curated, regularly refreshed source (e.g. ja3er,
#     trisulnsm/ja3prints), because every browser release changes its JA3.
#   * A miss therefore means "not in this short list", NOT "not a browser".
#   * A hit is a hint, never a verdict: fingerprints are trivially spoofable,
#     which is precisely the weakness of TLS-fingerprint-only approaches that
#     this project's research gap identifies.
# The feature is kept because it is cheap and mildly informative; it is
# deliberately never used on its own.
KNOWN_BROWSER_JA3 = {
    "e7d705a3286e19ea42f587b344ee6865",  # Chrome (older builds)
    "b32309a26951912be7dba376398abc3b",  # Chrome
    "6734f37431670b3ab4292b8f60f29984",  # Firefox
    "3b5074b1b5d032e5620f69f9f700ff0e",  # Safari
    "cd08e31494f9531f560d64c695473da9",  # Edge
    "51c64c77e60f3980eea90869b68c58a8",  # Chromium
}


def shannon_entropy(text: str) -> float:
    """Entropy of a hostname - DGA/randomised domains score noticeably higher."""
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class TLSInfo:
    """Everything we learn about one TLS session from its handshake."""

    is_tls: int = 0
    tls_version: int = 0
    ja3: str = ""
    ja3_hash: str = ""
    ja3s: str = ""
    ja3s_hash: str = ""
    n_cipher_suites: int = 0
    n_extensions: int = 0
    n_elliptic_curves: int = 0
    n_ec_point_formats: int = 0
    has_grease: int = 0
    sni: str = ""
    has_sni: int = 0
    alpn: list[str] = field(default_factory=list)
    has_alpn: int = 0
    alpn_is_h2: int = 0
    cert_chain_len: int = 0
    client_hello_ts: float = 0.0
    server_hello_ts: float = 0.0

    def to_features(self) -> dict[str, Any]:
        """The derived numeric view that actually reaches the model."""
        return {
            "is_tls": self.is_tls,
            "tls_version": self.tls_version,
            "ja3_hash": self.ja3_hash,
            "ja3s_hash": self.ja3s_hash,
            "ja3_is_known_browser": int(self.ja3_hash in KNOWN_BROWSER_JA3),
            "n_cipher_suites": self.n_cipher_suites,
            "n_extensions": self.n_extensions,
            "n_elliptic_curves": self.n_elliptic_curves,
            "n_ec_point_formats": self.n_ec_point_formats,
            "has_grease": self.has_grease,
            "has_sni": self.has_sni,
            "sni_length": len(self.sni),
            "sni_entropy": round(shannon_entropy(self.sni), 4),
            "sni_digit_ratio": (
                round(sum(c.isdigit() for c in self.sni) / len(self.sni), 4)
                if self.sni else 0.0
            ),
            "has_alpn": self.has_alpn,
            "alpn_is_h2": self.alpn_is_h2,
            "cert_chain_len": self.cert_chain_len,
            "handshake_duration_ms": (
                max(0.0, (self.server_hello_ts - self.client_hello_ts) * 1000.0)
                if self.client_hello_ts and self.server_hello_ts else 0.0
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Low-level parsing helpers
# ---------------------------------------------------------------------------
def _u16(buf: bytes, off: int) -> int:
    return struct.unpack_from("!H", buf, off)[0]


def _u24(buf: bytes, off: int) -> int:
    return (buf[off] << 16) | (buf[off + 1] << 8) | buf[off + 2]


def _strip_grease(values: list[int]) -> list[int]:
    return [v for v in values if v not in GREASE]


def _ja3_join(items: list[int]) -> str:
    return "-".join(str(v) for v in items)


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def _parse_extensions(
    buf: bytes, off: int, end: int
) -> tuple[list[int], list[int], list[int], str, list[str], bool]:
    """Walk a TLS extension block.

    Returns (ext_ids, curves, ec_point_formats, sni, alpn, saw_grease).
    """
    ext_ids: list[int] = []
    curves: list[int] = []
    ec_fmts: list[int] = []
    sni = ""
    alpn: list[str] = []
    saw_grease = False

    while off + 4 <= end:
        ext_type = _u16(buf, off)
        ext_len = _u16(buf, off + 2)
        off += 4
        body_end = off + ext_len
        if body_end > end:
            break
        if ext_type in GREASE:
            saw_grease = True
        else:
            ext_ids.append(ext_type)

        try:
            if ext_type == EXT_SERVER_NAME and ext_len >= 5:
                # server_name_list: [list_len(2)][type(1)][name_len(2)][name]
                name_len = _u16(buf, off + 3)
                sni = buf[off + 5: off + 5 + name_len].decode("utf-8", "ignore")
            elif ext_type == EXT_SUPPORTED_GROUPS and ext_len >= 2:
                list_len = _u16(buf, off)
                raw = [_u16(buf, off + 2 + i) for i in range(0, list_len, 2)]
                curves = _strip_grease(raw)
                saw_grease = saw_grease or len(raw) != len(curves)
            elif ext_type == EXT_EC_POINT_FORMATS and ext_len >= 1:
                fmt_len = buf[off]
                ec_fmts = list(buf[off + 1: off + 1 + fmt_len])
            elif ext_type == EXT_ALPN and ext_len >= 2:
                p = off + 2
                while p < body_end:
                    plen = buf[p]
                    alpn.append(buf[p + 1: p + 1 + plen].decode("ascii", "ignore"))
                    p += 1 + plen
        except (struct.error, IndexError):
            pass  # malformed extension - skip it, keep the rest

        off = body_end

    return ext_ids, curves, ec_fmts, sni, alpn, saw_grease


def parse_client_hello(payload: bytes) -> TLSInfo | None:
    """Parse a TLS ClientHello out of raw TCP payload bytes -> JA3."""
    try:
        if len(payload) < 45 or payload[0] != RECORD_HANDSHAKE:
            return None
        off = 5                        # skip record header: type(1) ver(2) len(2)
        if payload[off] != HS_CLIENT_HELLO:
            return None
        off += 4                       # handshake type(1) + length(3)
        client_version = _u16(payload, off)
        off += 2 + 32                  # client_version + random
        session_id_len = payload[off]
        off += 1 + session_id_len

        cs_len = _u16(payload, off)
        off += 2
        raw_ciphers = [_u16(payload, off + i) for i in range(0, cs_len, 2)]
        ciphers = _strip_grease(raw_ciphers)
        saw_grease = len(raw_ciphers) != len(ciphers)
        off += cs_len

        comp_len = payload[off]
        off += 1 + comp_len

        ext_ids: list[int] = []
        curves: list[int] = []
        ec_fmts: list[int] = []
        sni = ""
        alpn: list[str] = []
        if off + 2 <= len(payload):
            ext_total = _u16(payload, off)
            off += 2
            ext_ids, curves, ec_fmts, sni, alpn, grease2 = _parse_extensions(
                payload, off, min(off + ext_total, len(payload))
            )
            saw_grease = saw_grease or grease2

        ja3 = ",".join([
            str(client_version),
            _ja3_join(ciphers),
            _ja3_join(ext_ids),
            _ja3_join(curves),
            _ja3_join(ec_fmts),
        ])

        return TLSInfo(
            is_tls=1,
            tls_version=client_version,
            ja3=ja3,
            ja3_hash=_md5(ja3),
            n_cipher_suites=len(ciphers),
            n_extensions=len(ext_ids),
            n_elliptic_curves=len(curves),
            n_ec_point_formats=len(ec_fmts),
            has_grease=int(saw_grease),
            sni=sni,
            has_sni=int(bool(sni)),
            alpn=alpn,
            has_alpn=int(bool(alpn)),
            alpn_is_h2=int(any(a.startswith("h2") for a in alpn)),
        )
    except (struct.error, IndexError, ValueError):
        return None


def parse_server_hello(payload: bytes) -> dict[str, Any] | None:
    """Parse a ServerHello -> JA3S (server-side fingerprint)."""
    try:
        if len(payload) < 45 or payload[0] != RECORD_HANDSHAKE:
            return None
        off = 5
        if payload[off] != HS_SERVER_HELLO:
            return None
        off += 4
        version = _u16(payload, off)
        off += 2 + 32
        session_id_len = payload[off]
        off += 1 + session_id_len
        cipher = _u16(payload, off)
        off += 2 + 1                   # cipher_suite + compression_method

        ext_ids: list[int] = []
        if off + 2 <= len(payload):
            ext_total = _u16(payload, off)
            off += 2
            ext_ids = _parse_extensions(
                payload, off, min(off + ext_total, len(payload))
            )[0]

        ja3s = f"{version},{cipher},{_ja3_join(ext_ids)}"
        return {"ja3s": ja3s, "ja3s_hash": _md5(ja3s), "server_version": version}
    except (struct.error, IndexError, ValueError):
        return None


def count_certificates(payload: bytes) -> int:
    """Length of the certificate chain in a TLS Certificate message."""
    try:
        if len(payload) < 15 or payload[0] != RECORD_HANDSHAKE:
            return 0
        off = 5
        if payload[off] != HS_CERTIFICATE:
            return 0
        off += 4
        chain_len = _u24(payload, off)
        off += 3
        end = min(off + chain_len, len(payload))
        count = 0
        while off + 3 <= end:
            cert_len = _u24(payload, off)
            off += 3 + cert_len
            count += 1
        return count
    except (struct.error, IndexError):
        return 0


def ja3_bucket(ja3_hash: str, n_buckets: int = 256) -> int:
    """Hashing trick: map an unbounded fingerprint space to a fixed feature.

    Using the raw hash as a categorical would explode dimensionality and would
    not generalise at all to fingerprints unseen during training.
    """
    if not ja3_hash:
        return 0
    return int(ja3_hash[:8], 16) % n_buckets
