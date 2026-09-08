"""Launch the Streamlit monitoring dashboard.

    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 8502

Equivalent to ``streamlit run src/encids/dashboard/app.py``; this wrapper just
checks that a trained model exists first, so the failure mode is a clear
message rather than an empty page.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from encids.config import Paths, load_config          # noqa: E402
from encids.utils.logging_utils import get_logger     # noqa: E402

log = get_logger("scripts.dashboard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--headless", action="store_true",
                        help="do not open a browser window")
    args = parser.parse_args()

    paths = Paths.from_config(load_config())
    if not (paths.models / "stage1_supervised.joblib").exists():
        log.error("No trained model in %s", paths.models)
        log.error("Run this first:  python scripts/train_model.py")
        return 1

    app = ROOT / "src" / "encids" / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app),
               "--server.port", str(args.port)]
    if args.headless:
        command += ["--server.headless", "true"]

    log.info("Starting dashboard at http://localhost:%d", args.port)
    return subprocess.call(command, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
