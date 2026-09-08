"""Tests for JA3 / JA3S extraction from raw TLS handshake bytes."""
from __future__ import annotations

import hashlib
import struct

import pytest

from encids.features.tls_features import (
    GREASE,
    ja3_bucket,
    parse_client_hello,
    parse_server_hello,
    shannon_entropy,
)


def build_client_hello(version: int = 0x0303,
                       ciphers: list[int] | None = None,
                       extensions: list[tuple[int, bytes]] | None = None,
                       session_id: bytes = b"") -> bytes:
    """Assemble a syntactically valid TLS ClientHello record."""
    ciphers = ciphers if ciphers is not None else [0x1301, 0x1302, 0xC02B]
    extensions = extensions if extensions is not None else []

    body = struct.pack("!H", version) + b"\x00" * 32
    body += bytes([len(session_id)]) + session_id
    body += struct.pack("!H", len(ciphers) * 2)
    body += b"".join(struct.pack("!H", c) for c in ciphers)
    body += b"\x01\x00"                       # 1 compression method: null

    ext_blob = b"".join(struct.pack("!HH", t, len(v)) + v for t, v in extensions)
    body += struct.pack("!H", len(ext_blob)) + ext_blob

    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


def sni_extension(hostname: str) -> tuple[int, bytes]:
    name = hostname.encode()
    entry = b"\x00" + struct.pack("!H", len(name)) + name
    return 0x0000, struct.pack("!H", len(entry)) + entry


def supported_groups(curves: list[int]) -> tuple[int, bytes]:
    blob = b"".join(struct.pack("!H", c) for c in curves)
    return 0x000A, struct.pack("!H", len(blob)) + blob


def ec_point_formats(formats: list[int]) -> tuple[int, bytes]:
    return 0x000B, bytes([len(formats)]) + bytes(formats)


def alpn_extension(protocols: list[str]) -> tuple[int, bytes]:
    blob = b"".join(bytes([len(p)]) + p.encode() for p in protocols)
    return 0x0010, struct.pack("!H", len(blob)) + blob


# ---------------------------------------------------------------------------
class TestClientHello:
    def test_parses_a_minimal_hello(self):
        info = parse_client_hello(build_client_hello())
        assert info is not None
        assert info.is_tls == 1
        assert info.tls_version == 0x0303
        assert info.n_cipher_suites == 3
        assert len(info.ja3_hash) == 32

    def test_ja3_string_has_five_comma_separated_fields(self):
        info = parse_client_hello(build_client_hello(
            extensions=[supported_groups([23, 24]), ec_point_formats([0])]))
        assert info.ja3.count(",") == 4
        version, ciphers, exts, curves, formats = info.ja3.split(",")
        assert version == "771"
        assert ciphers == "4865-4866-49195"
        assert curves == "23-24"
        assert formats == "0"

    def test_ja3_hash_is_md5_of_the_ja3_string(self):
        info = parse_client_hello(build_client_hello())
        assert info.ja3_hash == hashlib.md5(info.ja3.encode()).hexdigest()

    def test_identical_hellos_give_identical_fingerprints(self):
        a = parse_client_hello(build_client_hello(session_id=b"\x01" * 32))
        b = parse_client_hello(build_client_hello(session_id=b"\x02" * 32))
        # The session id is per-connection and must not affect the fingerprint.
        assert a.ja3_hash == b.ja3_hash

    def test_different_cipher_order_gives_a_different_fingerprint(self):
        a = parse_client_hello(build_client_hello(ciphers=[0x1301, 0x1302]))
        b = parse_client_hello(build_client_hello(ciphers=[0x1302, 0x1301]))
        assert a.ja3_hash != b.ja3_hash

    def test_grease_values_are_stripped_and_flagged(self):
        grease = sorted(GREASE)[0]
        info = parse_client_hello(build_client_hello(
            ciphers=[grease, 0x1301, 0x1302]))
        assert info.has_grease == 1
        assert info.n_cipher_suites == 2
        assert str(grease) not in info.ja3

    def test_grease_stripping_matches_a_hello_without_it(self):
        grease = sorted(GREASE)[0]
        with_grease = parse_client_hello(
            build_client_hello(ciphers=[grease, 0x1301]))
        without = parse_client_hello(build_client_hello(ciphers=[0x1301]))
        assert with_grease.ja3_hash == without.ja3_hash

    def test_extracts_sni(self):
        info = parse_client_hello(build_client_hello(
            extensions=[sni_extension("cdn.example.com")]))
        assert info.sni == "cdn.example.com"
        assert info.has_sni == 1

    def test_extracts_alpn(self):
        info = parse_client_hello(build_client_hello(
            extensions=[alpn_extension(["h2", "http/1.1"])]))
        assert info.alpn == ["h2", "http/1.1"]
        assert info.to_features()["alpn_is_h2"] == 1

    @pytest.mark.parametrize("payload", [
        b"", b"\x16", b"\x17\x03\x01\x00\x05hello",       # not a handshake
        b"\x16\x03\x01\x00\x05" + b"\x02" + b"\x00" * 40,  # ServerHello, not Client
        b"\x16\x03\x01" + b"\xff" * 200,                   # garbage
    ])
    def test_rejects_non_client_hello_input(self, payload):
        assert parse_client_hello(payload) is None

    def test_truncated_hello_does_not_raise(self):
        full = build_client_hello(extensions=[sni_extension("a.example.com")])
        for cut in range(5, len(full)):
            parse_client_hello(full[:cut])      # must never raise


class TestServerHello:
    def build(self, version: int = 0x0303, cipher: int = 0x1301) -> bytes:
        body = struct.pack("!H", version) + b"\x00" * 32
        body += b"\x00"                          # empty session id
        body += struct.pack("!H", cipher) + b"\x00"
        body += struct.pack("!H", 0)             # no extensions
        handshake = b"\x02" + len(body).to_bytes(3, "big") + body
        return b"\x16\x03\x03" + struct.pack("!H", len(handshake)) + handshake

    def test_parses_ja3s(self):
        result = parse_server_hello(self.build())
        assert result is not None
        assert result["ja3s"] == "771,4865,"
        assert result["ja3s_hash"] == hashlib.md5(b"771,4865,").hexdigest()

    def test_different_cipher_gives_different_ja3s(self):
        a = parse_server_hello(self.build(cipher=0x1301))
        b = parse_server_hello(self.build(cipher=0x1302))
        assert a["ja3s_hash"] != b["ja3s_hash"]

    def test_rejects_a_client_hello(self):
        assert parse_server_hello(build_client_hello()) is None


class TestDerivedFeatures:
    def test_entropy_of_random_hostname_exceeds_a_repetitive_one(self):
        assert shannon_entropy("x7q2m9zp4kfd") > shannon_entropy("aaaaaaaaaaaa")

    def test_entropy_of_empty_string_is_zero(self):
        assert shannon_entropy("") == 0.0

    def test_bucket_is_stable_and_in_range(self):
        digest = hashlib.md5(b"anything").hexdigest()
        assert ja3_bucket(digest, 256) == ja3_bucket(digest, 256)
        assert 0 <= ja3_bucket(digest, 256) < 256

    def test_empty_fingerprint_maps_to_bucket_zero(self):
        assert ja3_bucket("", 256) == 0

    def test_sni_features_are_derived_consistently(self):
        info = parse_client_hello(build_client_hello(
            extensions=[sni_extension("a1b2c3.example.com")]))
        features = info.to_features()
        assert features["sni_length"] == len("a1b2c3.example.com")
        assert 0 < features["sni_digit_ratio"] < 1
        assert features["sni_entropy"] > 0
