"""Live traffic capture -> flow records.

Three sources, all producing the identical feature vectors the models were
trained on:

  * ``scapy``  - sniffs a live interface.  Needs Npcap on Windows, libpcap on
    Linux/macOS, and administrator/root privileges.
  * ``pcap``   - replays a capture file, optionally in real time.  This is the
    demo path when packet capture cannot be installed, and the reproducible
    path for a recorded demo video.
  * ``synthetic`` - emits generated flows on a timer.  Lets the dashboard be
    developed and demonstrated with no capture privileges at all.

Only packet *headers* and the cleartext TLS handshake are read.  Payload bytes
are never stored, and only the first 2 KB of any TCP segment is even looked at,
purely to find the ClientHello.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from ..features.flow_stats import FlowTable, Packet, iter_pcap_packets
from ..utils.logging_utils import get_logger

log = get_logger("live.capture")

FlowCallback = Callable[[list[dict[str, Any]]], None]


# ---------------------------------------------------------------------------
def list_interfaces() -> list[dict[str, str]]:
    """Enumerate capture interfaces, with a readable description."""
    try:
        from scapy.arch import get_if_list
        from scapy.config import conf
    except Exception as exc:
        log.warning("scapy unavailable (%s)", exc)
        return []

    out = []
    for name in get_if_list():
        description = name
        try:
            iface = conf.ifaces.dev_from_name(name)
            description = getattr(iface, "description", None) or name
        except Exception:
            pass
        out.append({"name": name, "description": description})
    return out


def capture_available() -> tuple[bool, str]:
    """Can we actually sniff on this machine?  Returns (ok, explanation).

    Enumerating interface names is NOT sufficient evidence. On Windows scapy
    lists ``\\Device\\NPF_{...}`` names straight from the registry even when the
    Npcap driver is absent, so a name-only check reports "capture available"
    and then every sniff fails with "winpcap is not installed". The presence of
    a working packet-capture provider has to be checked directly.
    """
    try:
        from scapy.arch import get_if_list
        from scapy.config import conf
    except Exception as exc:
        return False, f"scapy not importable: {exc}"

    try:
        interfaces = get_if_list()
    except Exception as exc:
        return False, f"no capture interfaces: {exc}"
    if not interfaces:
        return False, ("no capture interfaces found - install Npcap "
                       "(https://npcap.com) on Windows or libpcap on Linux/macOS")

    # Is there a libpcap/Npcap provider behind those names?
    try:
        from scapy.arch import libpcap  # noqa: F401

        if not getattr(conf, "use_pcap", False):
            raise ImportError("libpcap present but not selected")
    except Exception:
        import platform

        if platform.system() == "Windows":
            return False, ("Npcap is not installed - interface names exist but "
                           "no capture driver is behind them. Install "
                           "tools/npcap-1.88.exe as Administrator.")
        return False, ("no libpcap provider - install libpcap "
                       "(e.g. apt install libpcap0.8)")

    return True, f"{len(interfaces)} interface(s) available"


# ---------------------------------------------------------------------------
def _scapy_to_packet(pkt) -> Packet | None:
    """Convert a scapy packet to the internal :class:`Packet`."""
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    if pkt.haslayer(IP):
        ip = pkt[IP]
        src, dst = ip.src, ip.dst
        ip_hdr_len = ip.ihl * 4
        total_len = ip.len or len(pkt)
    elif pkt.haslayer(IPv6):
        ip = pkt[IPv6]
        src, dst = ip.src, ip.dst
        ip_hdr_len = 40
        total_len = ip.plen + 40
    else:
        return None

    ts = float(pkt.time)
    if pkt.haslayer(TCP):
        tcp = pkt[TCP]
        payload = bytes(tcp.payload)[:2048]
        return Packet(ts=ts, src_ip=src, dst_ip=dst, src_port=int(tcp.sport),
                      dst_port=int(tcp.dport), protocol=6, length=int(total_len),
                      header_len=ip_hdr_len + tcp.dataofs * 4,
                      flags=int(tcp.flags), window=int(tcp.window),
                      payload=payload)
    if pkt.haslayer(UDP):
        udp = pkt[UDP]
        return Packet(ts=ts, src_ip=src, dst_ip=dst, src_port=int(udp.sport),
                      dst_port=int(udp.dport), protocol=17, length=int(total_len),
                      header_len=ip_hdr_len + 8)
    return None


# ---------------------------------------------------------------------------
class LiveCapture:
    """Sniff an interface and emit completed flows through a callback."""

    def __init__(self, interface: str | None = None, bpf_filter: str = "",
                 idle_timeout: float = 15.0, active_timeout: float = 120.0,
                 flush_interval: float = 3.0):
        self.interface = interface
        self.bpf_filter = bpf_filter
        self.flush_interval = flush_interval
        self.table = FlowTable(idle_timeout, active_timeout)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._callback: FlowCallback | None = None
        self._last_flush = time.time()
        self.packets_seen = 0
        self.flows_emitted = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self, callback: FlowCallback) -> None:
        self._callback = callback
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="encids-capture")
        self._thread.start()
        log.info("Capture started on %s%s", self.interface or "<default>",
                 f" (filter: {self.bpf_filter})" if self.bpf_filter else "")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        # Do not discard flows that were still open when the user stopped.
        remaining = self.table.flush()
        if remaining and self._callback:
            self._emit(remaining)
        log.info("Capture stopped: %d packets, %d flows", self.packets_seen,
                 self.flows_emitted)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internals ---------------------------------------------------------
    def _emit(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.flows_emitted += len(records)
        if self._callback:
            self._callback(records)

    def _handle(self, scapy_pkt) -> None:
        pkt = _scapy_to_packet(scapy_pkt)
        if pkt is None:
            return
        self.packets_seen += 1
        self._emit(self.table.add(pkt))

        # Periodically surface long-running flows so the dashboard is not
        # silent while a big download is still in progress.
        now = time.time()
        if now - self._last_flush >= self.flush_interval:
            self._last_flush = now

    def _run(self) -> None:
        from scapy.sendrecv import sniff

        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter or None,
                prn=self._handle,
                store=False,
                stop_filter=lambda _: self._stop.is_set(),
            )
        except PermissionError:
            log.error("Permission denied - run as Administrator (Windows) or "
                      "with sudo / CAP_NET_RAW (Linux)")
        except Exception as exc:
            log.error("Capture failed: %s", exc)


# ---------------------------------------------------------------------------
def replay_pcap(path: str | Path, callback: FlowCallback,
                idle_timeout: float = 15.0, active_timeout: float = 120.0,
                speed: float = 0.0, batch: int = 20) -> int:
    """Replay a pcap through the same flow pipeline.

    ``speed`` 0 means as fast as possible; 1.0 replays in real time, 10.0 at
    ten times real time.
    """
    table = FlowTable(idle_timeout, active_timeout)
    pending: list[dict[str, Any]] = []
    total = 0
    previous_ts: float | None = None

    for pkt in iter_pcap_packets(str(path)):
        if speed > 0 and previous_ts is not None:
            delay = (pkt.ts - previous_ts) / speed
            if 0 < delay < 5:
                time.sleep(delay)
        previous_ts = pkt.ts

        pending.extend(table.add(pkt))
        if len(pending) >= batch:
            callback(pending)
            total += len(pending)
            pending = []

    pending.extend(table.flush())
    if pending:
        callback(pending)
        total += len(pending)
    log.info("Replayed %s -> %d flows", Path(path).name, total)
    return total


def synthetic_stream(callback: FlowCallback, interval: float = 2.0,
                     flows_per_tick: int = 6, attack_fraction: float = 0.15,
                     stop_event: threading.Event | None = None,
                     seed: int = 0) -> None:
    """Emit generated flows on a timer - a demo source needing no privileges."""
    from ..data import synthetic as syn

    stop_event = stop_event or threading.Event()
    tick = 0
    while not stop_event.is_set():
        df = syn.generate(n_flows=flows_per_tick, attack_fraction=attack_fraction,
                          seed=seed + tick)
        callback(df.to_dict(orient="records"))
        tick += 1
        stop_event.wait(interval)


def iter_flows_from_pcap(path: str | Path, idle_timeout: float = 15.0,
                         active_timeout: float = 120.0
                         ) -> Iterator[dict[str, Any]]:
    """Convenience generator over a pcap's completed flows."""
    table = FlowTable(idle_timeout, active_timeout)
    for pkt in iter_pcap_packets(str(path)):
        yield from table.add(pkt)
    yield from table.flush()
