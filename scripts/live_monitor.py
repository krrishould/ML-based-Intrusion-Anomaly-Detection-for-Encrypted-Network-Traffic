"""Run the live capture -> score pipeline from the terminal.

    python scripts/live_monitor.py --source synthetic --duration 30
    python scripts/live_monitor.py --source pcap --pcap data/raw/sample.pcap
    python scripts/live_monitor.py --source interface --duration 60   # needs admin
    python scripts/live_monitor.py --list-interfaces

Live capture requires Npcap (Windows) or libpcap (Linux/macOS) and
administrator/root privileges.  Without them, use --source synthetic or pcap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.config import ensure_dirs, load_config          # noqa: E402
from encids.live.capture import capture_available, list_interfaces  # noqa: E402
from encids.live.scorer import run_live                     # noqa: E402
from encids.utils.logging_utils import banner, get_logger   # noqa: E402

log = get_logger("scripts.live")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", default="auto",
                        choices=["auto", "interface", "pcap", "synthetic"])
    parser.add_argument("--pcap", default=None, help="capture file to replay")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--duration", type=float, default=60.0,
                        help="seconds to run (0 = until Ctrl-C)")
    parser.add_argument("--speed", type=float, default=0.0,
                        help="pcap replay speed multiplier (0 = max)")
    parser.add_argument("--list-interfaces", action="store_true")
    args = parser.parse_args()

    if args.list_interfaces:
        ok, why = capture_available()
        banner("Capture interfaces")
        log.info("capture available: %s (%s)", ok, why)
        for iface in list_interfaces():
            log.info("  %-40s %s", iface["name"], iface["description"])
        return 0

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    banner("Live monitoring")
    scorer = run_live(cfg, source=args.source, pcap=args.pcap,
                      duration=args.duration or None,
                      interface=args.interface, speed=args.speed)

    stats = scorer.stats()
    banner("Session summary")
    log.info("flows scored      %d", stats["total_flows"])
    log.info("alerts raised     %d (%.1f%%)", stats["total_alerts"],
             100 * stats["alert_rate"])
    log.info("throughput        %.1f flows/s", stats["flows_per_second"])
    log.info("scoring latency   %.3f ms/flow", stats["mean_latency_ms"])

    df = scorer.snapshot()
    alerts = df[df["alert"] == 1] if not df.empty else df
    if not alerts.empty:
        banner("Recent alerts")
        for _, row in alerts.head(10).iterrows():
            log.info("%s:%s -> %s:%s | %s | risk %.2f",
                     row["src_ip"], row["src_port"], row["dst_ip"],
                     row["dst_port"], row["reason"], float(row["risk"]))
            if row.get("explanation"):
                log.info("    %s", row["explanation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
