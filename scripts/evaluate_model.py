"""Full evaluation: ablation, zero-day experiment, latency, figures.

    python scripts/evaluate_model.py                     # everything
    python scripts/evaluate_model.py --skip-ablation     # faster
    python scripts/evaluate_model.py --zero-day tor_darknet

Writes reports/metrics/evaluation.json, the ablation CSV, and the figures used
in the report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.config import ensure_dirs, load_config    # noqa: E402
from encids.evaluate import main as evaluate_main     # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--skip-ablation", action="store_true",
                        help="skip the flow-only vs flow+TLS comparison "
                             "(halves the runtime)")
    parser.add_argument("--zero-day", default="c2_beacon", metavar="FAMILY",
                        help="attack family to withhold from Stage 1; "
                             "pass '' to skip the experiment")
    parser.add_argument("--source", default=None,
                        help="evaluate on one dataset only (synthetic, ctu13). "
                             "Recommended: the TLS ablation is only meaningful "
                             "on a source that actually carries TLS metadata.")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="sub-sample to at most N flows (keeps class balance)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    evaluate_main(cfg, skip_ablation=args.skip_ablation,
                  zero_day_family=args.zero_day or "",
                  source=args.source, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
