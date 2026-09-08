"""Generate a small sample capture for demos and end-to-end testing.

    python scripts/make_sample_pcap.py
    python scripts/make_sample_pcap.py --out data/raw/demo.pcap --flows 60

Writes a pcap containing a mix of traffic patterns - normal HTTPS browsing, a
periodic C2 beacon, an upload-heavy exfiltration flow, and a port scan - each
with a real, parseable TLS ClientHello so the JA3 extraction path is exercised
for real rather than mocked.

This is a *demo fixture*, not evidence. It exists so `--source pcap` and the
dashboard can be shown working without needing packet-capture privileges or a
28 GB dataset download.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.utils.logging_utils import banner, get_logger   # noqa: E402

log = get_logger("scripts.pcap")

# TLS constants for handcrafting a ClientHello
BROWSER_CIPHERS = [0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F, 0xC02C, 0xC030,
                   0xCCA9, 0xCCA8, 0xC013, 0xC014, 0x009C, 0x009D, 0x002F, 0x0035]
MALWARE_CIPHERS = [0xC02F, 0xC030, 0x009C, 0x002F]
GREASE_VALUE = 0x0A0A


def _client_hello(hostname: str, ciphers: list[int], curves: list[int],
                  with_grease: bool, with_alpn: bool) -> bytes:
    """Build a real TLS ClientHello record."""
    extensions: list[tuple[int, bytes]] = []

    name = hostname.encode()
    entry = b"\x00" + struct.pack("!H", len(name)) + name
    extensions.append((0x0000, struct.pack("!H", len(entry)) + entry))

    curve_blob = b"".join(struct.pack("!H", c) for c in curves)
    extensions.append((0x000A, struct.pack("!H", len(curve_blob)) + curve_blob))
    extensions.append((0x000B, b"\x01\x00"))                 # ec_point_formats

    if with_alpn:
        alpn = b"\x02h2\x08http/1.1"
        extensions.append((0x0010, struct.pack("!H", len(alpn)) + alpn))
    if with_grease:
        extensions.append((GREASE_VALUE, b""))

    cipher_list = ([GREASE_VALUE] + ciphers) if with_grease else ciphers

    body = struct.pack("!H", 0x0303) + bytes(range(32))       # version + random
    body += b"\x20" + bytes(range(32))                        # 32-byte session id
    body += struct.pack("!H", len(cipher_list) * 2)
    body += b"".join(struct.pack("!H", c) for c in cipher_list)
    body += b"\x01\x00"                                       # null compression

    ext_blob = b"".join(struct.pack("!HH", t, len(v)) + v for t, v in extensions)
    body += struct.pack("!H", len(ext_blob)) + ext_blob

    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


def _server_hello(cipher: int = 0x1301) -> bytes:
    body = struct.pack("!H", 0x0303) + bytes(range(32))
    body += b"\x00"
    body += struct.pack("!H", cipher) + b"\x00"
    body += struct.pack("!H", 0)
    handshake = b"\x02" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x03" + struct.pack("!H", len(handshake)) + handshake


def _certificate(n_certs: int = 3) -> bytes:
    """A TLS Certificate message carrying `n_certs` placeholder certificates.

    Only the chain *length* is used as a feature, so the certificate bodies are
    filler bytes. Real TLS 1.2 traffic always carries this message; omitting it
    left every fixture flow with cert_chain_len = 0, which matches no real
    capture and made benign fixture flows look anomalous.
    """
    certs = b""
    for i in range(n_certs):
        body = b"\x30\x82\x01\x00" + bytes((i + 1) % 251 for _ in range(256))
        certs += len(body).to_bytes(3, "big") + body
    payload = len(certs).to_bytes(3, "big") + certs
    handshake = b"\x0b" + len(payload).to_bytes(3, "big") + payload
    return b"\x16\x03\x03" + struct.pack("!H", len(handshake)) + handshake


def build(out: Path, n_flows: int, seed: int) -> int:
    import random

    from scapy.all import IP, TCP, Ether, Raw, wrpcap

    rng = random.Random(seed)
    packets = []
    now = 1_700_000_000.0
    client = "192.168.1.50"

    profiles = [
        # (name, server, hostname, ciphers, grease, alpn, n_pkts,
        #  fwd_size, bwd_size, gap, jitter)
        # Server-to-client sizes sit at the MTU because bulk transfers fill
        # segments. A fixture whose packets never reach ~1460 bytes does not
        # resemble any real capture.
        ("browsing", "93.184.216.34", "www.example.com", BROWSER_CIPHERS,
         True, True, 40, 200, 1460, 0.04, 0.9),
        ("streaming", "151.101.1.140", "media.stream.tv", BROWSER_CIPHERS,
         True, True, 120, 140, 1460, 0.012, 0.4),
        # NOTE: the beacon interval is 8 s, not the 30-60 s a real implant
        # typically uses. A flow ends after `live.idle_timeout` seconds of
        # silence (15 s by default), so a 60 s beacon would be split into one
        # single-packet flow per check-in and its defining signal - metronomic
        # inter-arrival timing - would be destroyed before the model ever sees
        # it. Detecting slow beacons needs either a longer idle timeout or
        # cross-flow aggregation per (host, destination) channel; see
        # docs/METHODOLOGY.md.
        ("c2_beacon", "45.77.12.99", "a7f3k9q2m1x8.net", MALWARE_CIPHERS,
         False, False, 14, 210, 260, 8.0, 0.02),
        ("exfiltration", "185.220.101.7", "backup-sync.net", MALWARE_CIPHERS,
         False, False, 90, 1460, 100, 0.02, 0.5),
        ("port_scan", "10.0.0.200", "", [], False, False, 3, 60, 60, 0.005, 0.3),
    ]

    for index in range(n_flows):
        (name, server, hostname, ciphers, grease, alpn,
         n_pkts, fwd_size, bwd_size, gap, jitter) = profiles[index % len(profiles)]
        sport = 40000 + index
        dport = 443 if name != "port_scan" else rng.choice([22, 23, 445, 3389])
        ts = now + index * 3.0
        seq = ack = 1

        def frame(src, dst, sp, dp, flags, payload, timestamp):
            # Fixed MACs: scapy would otherwise try to ARP-resolve each address
            # and emit a warning per packet on a host with no capture driver.
            pkt = (Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02")
                   / IP(src=src, dst=dst)
                   / TCP(sport=sp, dport=dp, flags=flags, seq=seq, ack=ack,
                         window=64240))
            if payload:
                pkt = pkt / Raw(load=payload)
            pkt.time = timestamp
            return pkt

        # TCP handshake
        packets.append(frame(client, server, sport, dport, "S", b"", ts))
        packets.append(frame(server, client, dport, sport, "SA", b"", ts + 0.02))
        packets.append(frame(client, server, sport, dport, "A", b"", ts + 0.021))

        if name == "port_scan":
            packets.append(frame(server, client, dport, sport, "R", b"", ts + 0.03))
            continue

        # TLS handshake - a genuine ClientHello, so JA3 extraction is exercised
        hello = _client_hello(hostname, ciphers,
                              [0x001D, 0x0017, 0x0018] if grease else [0x0017],
                              grease, alpn)
        packets.append(frame(client, server, sport, dport, "PA", hello, ts + 0.03))
        packets.append(frame(server, client, dport, sport, "PA", _server_hello(),
                             ts + 0.06))
        # Browser-facing servers present a full chain; the C2 servers in this
        # fixture present a single (self-signed-looking) certificate.
        packets.append(frame(server, client, dport, sport, "PA",
                             _certificate(3 if grease else 1), ts + 0.07))

        # Encrypted application data
        offset = ts + 0.1
        for i in range(n_pkts):
            offset += gap * (1.0 + rng.uniform(-jitter, jitter))
            # Client requests are small and bursty; server responses fill
            # segments. A fixed 50/50 direction split would erase the
            # download/upload asymmetry that separates browsing from
            # exfiltration - the very signal the model keys on.
            client_share = 0.5 if name == "c2_beacon" else 0.35
            if rng.random() < client_share:
                size = max(60, min(1460, int(rng.gauss(fwd_size,
                                                       fwd_size * 0.45))))
                packets.append(frame(client, server, sport, dport, "PA",
                                     b"\x17\x03\x03" + b"\x00" * (size - 3),
                                     offset))
            else:
                size = max(60, min(1460, int(rng.gauss(bwd_size,
                                                       bwd_size * 0.45))))
                packets.append(frame(server, client, dport, sport, "PA",
                                     b"\x17\x03\x03" + b"\x00" * (size - 3),
                                     offset))

        packets.append(frame(client, server, sport, dport, "FA", b"", offset + 0.5))
        packets.append(frame(server, client, dport, sport, "FA", b"", offset + 0.52))

    packets.sort(key=lambda p: p.time)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrpcap(str(out), packets)
    return len(packets)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/raw/sample_traffic.pcap")
    parser.add_argument("--flows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = Path(__file__).resolve().parents[1] / out

    banner("Generating a sample capture")
    n_packets = build(out, args.flows, args.seed)
    log.info("%d packets across ~%d flows -> %s", n_packets, args.flows, out)

    # Verify the round trip through our own extraction path.
    from encids.features.flow_stats import flows_from_pcap

    records = flows_from_pcap(str(out))
    tls = [r for r in records if r.get("is_tls")]
    log.info("Read back: %d flows, %d with a parsed TLS handshake", len(records),
             len(tls))
    if tls:
        example = tls[0]
        log.info("Example: %s -> %s | JA3 %s | SNI %r",
                 example["src_ip"], example["dst_ip"],
                 example["ja3_hash"][:16] + "...", example.get("sni", ""))
    log.info("")
    log.info("Try it:  python scripts/live_monitor.py --source pcap --pcap %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
