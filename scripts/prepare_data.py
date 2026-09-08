"""Build the unified flow-feature table from every enabled dataset.

    python scripts/prepare_data.py                # use whatever is downloaded
    python scripts/prepare_data.py --force        # rebuild from scratch
    python scripts/prepare_data.py --backend native   # skip NFStream
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.config import ensure_dirs, load_config          # noqa: E402
from encids.data.build_dataset import build                 # noqa: E402
from encids.features.build_features import summarise        # noqa: E402
from encids.utils.logging_utils import banner, get_logger   # noqa: E402

log = get_logger("scripts.prepare")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="rebuild even if a cached table exists")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "nfstream", "native"],
                        help="pcap-to-flow engine (default: auto)")
    parser.add_argument("--limit-pcaps", type=int, default=None,
                        help="only read the first N pcaps (quick smoke test)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    df = build(cfg, backend=args.backend, force=args.force,
               limit_pcaps=args.limit_pcaps)

    banner("Dataset summary")
    stats = summarise(df)
    log.info("flows           %d", stats["n_flows"])
    log.info("malicious       %d (%.1f%%)", stats["n_malicious"],
             100 * stats["malicious_rate"])
    log.info("TLS flows       %.1f%%", 100 * stats["tls_flow_rate"])
    log.info("sources         %s", stats["sources"] or "-")
    log.info("classes:")
    for name, count in sorted(stats["class_counts"].items(),
                              key=lambda kv: -kv[1]):
        log.info("   %-24s %7d", name, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
