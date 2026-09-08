"""Train the two-stage detector.

    python scripts/train_model.py                       # full dual-signal model
    python scripts/train_model.py --no-tls              # flow-only baseline
    python scripts/train_model.py --hold-out c2_beacon  # zero-day setup
    python scripts/train_model.py --stage1 random_forest --stage2 isolation_forest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encids.config import ensure_dirs, load_config          # noqa: E402
from encids.evaluate import evaluate_detector               # noqa: E402
from encids.train import train                              # noqa: E402
from encids.utils.logging_utils import banner, get_logger   # noqa: E402

log = get_logger("scripts.train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-tls", action="store_true",
                        help="train the flow-statistics-only baseline")
    parser.add_argument("--hold-out", default=None, metavar="FAMILY",
                        help="withhold this attack family from Stage 1 "
                             "(zero-day experiment)")
    parser.add_argument("--stage1", default=None,
                        choices=["random_forest", "xgboost"])
    parser.add_argument("--stage2", default=None,
                        choices=["isolation_forest", "autoencoder"])
    parser.add_argument("--source", default=None,
                        help="train on one dataset only (e.g. synthetic, ctu13); "
                             "recommended when sources differ in which features "
                             "they can provide")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="sub-sample to at most N flows, keeping the class "
                             "balance")
    parser.add_argument("--model-dir", default=None,
                        help="where to write the artefacts (default: models/)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.stage1:
        cfg["supervised"]["model"] = args.stage1
    if args.stage2:
        cfg["anomaly"]["model"] = args.stage2

    artefacts = train(
        cfg,
        use_tls=False if args.no_tls else None,
        hold_out_family=args.hold_out,
        model_dir=args.model_dir,
        source=args.source,
        max_rows=args.max_rows,
    )

    banner("Held-out test performance")
    results = evaluate_detector(artefacts.detector, artefacts.test_df)
    for stage in ("fused", "stage1_only", "stage2_only"):
        m = results[stage]
        log.info("%-12s acc=%.4f  f1=%.4f  recall=%.4f  precision=%.4f  "
                 "FPR=%.4f  AUC=%.4f", stage, m["accuracy"], m["f1"],
                 m["recall"], m["precision"], m["false_positive_rate"],
                 m.get("roc_auc", float("nan")))

    log.info("")
    log.info("Per-family detection rate (benign rows = false-positive rate):")
    for family, rate in sorted(results["per_family_detection_rate"].items(),
                               key=lambda kv: kv[1]):
        log.info("   %-24s %.3f", family, rate)

    log.info("")
    log.info("Next: python scripts/evaluate_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
