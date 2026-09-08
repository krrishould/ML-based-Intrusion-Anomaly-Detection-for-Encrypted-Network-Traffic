"""Download the public datasets used by the project.

    python scripts/download_datasets.py --list
    python scripts/download_datasets.py --dataset ctu13            # ~290 MB
    python scripts/download_datasets.py --dataset ctu13 --full     # ~4.7 GB
    python scripts/download_datasets.py --check

Availability is not the same for all three datasets:

  * **CTU-13** is served directly by CTU/Stratosphere and downloads without
    registration.  This script fetches it.
  * **ISCX VPN-nonVPN 2016** and **CIC-Darknet2020** are behind the University
    of New Brunswick's registration form.  They cannot be fetched by script;
    the exact URLs, the form, and where to unpack the files are printed for you.

Downloads resume if interrupted (HTTP Range), and a file that is already
complete is skipped.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests  # noqa: E402

from encids.config import ensure_dirs, load_config          # noqa: E402
from encids.utils.logging_utils import banner, get_logger   # noqa: E402

log = get_logger("scripts.download")

CTU_BASE = "https://mcfp.felk.cvut.cz/publicDatasets"

# CTU-13 scenario -> (capture directory, netflow filename, size in MB).
# Ordered smallest-first so the default set is the cheap, diverse one.
CTU13_SCENARIOS: dict[int, tuple[str, str, int]] = {
    11: ("CTU-Malware-Capture-Botnet-52", "capture20110818-2.binetflow.2format", 25),
    9:  ("CTU-Malware-Capture-Botnet-48", "capture20110816-2.binetflow.2format", 27),
    5:  ("CTU-Malware-Capture-Botnet-46", "capture20110815-2.binetflow.2format", 31),
    12: ("CTU-Malware-Capture-Botnet-53", "capture20110819.binetflow.2format", 77),
    6:  ("CTU-Malware-Capture-Botnet-47", "capture20110816.binetflow.2format", 132),
    4:  ("CTU-Malware-Capture-Botnet-45", "capture20110815.binetflow.2format", 264),
    10: ("CTU-Malware-Capture-Botnet-51", "capture20110818.binetflow.2format", 308),
    2:  ("CTU-Malware-Capture-Botnet-43", "capture20110811.binetflow.2format", 424),
    13: ("CTU-Malware-Capture-Botnet-54", "capture20110815-3.binetflow.2format", 452),
    8:  ("CTU-Malware-Capture-Botnet-50", "capture20110817.binetflow.2format", 491),
    1:  ("CTU-Malware-Capture-Botnet-42", "capture20110810.binetflow.2format", 664),
    7:  ("CTU-Malware-Capture-Botnet-49", "capture20110816-3.binetflow.2format", 695),
    3:  ("CTU-Malware-Capture-Botnet-44", "capture20110812.binetflow.2format", 1105),
}

# Botnet families, so the default selection covers more than one malware family.
CTU13_DEFAULT = [11, 9, 5, 12, 6]     # Rbot, Murlo, Virut, NSIS.ay, Menti  (~290 MB)


@dataclass
class ManualDataset:
    """A dataset that a script is not permitted to fetch."""

    name: str
    portal: str
    direct: str
    target: str
    size: str
    note: str
    files: list[str] = field(default_factory=list)


MANUAL: dict[str, ManualDataset] = {
    "iscx_vpn2016": ManualDataset(
        name="ISCX VPN-nonVPN 2016",
        portal="https://www.unb.ca/cic/datasets/vpn.html",
        direct="http://cicresearch.ca/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/",
        target="data/raw/iscx_vpn2016/",
        size="~28 GB (pcaps) or ~50 MB (pre-extracted CSVs)",
        note=("Requires accepting UNB's terms on the portal page. Download the "
              "PCAPs folder for the full pipeline, or the CSVs folder if you "
              "only need flow features. Put the files anywhere under the "
              "target directory - subfolders are searched recursively."),
        files=["PCAPs/*.pcap", "CSVs/*.csv"],
    ),
    "cic_darknet2020": ManualDataset(
        name="CIC-Darknet2020",
        portal="https://www.unb.ca/cic/datasets/darknet2020.html",
        direct="http://cicresearch.ca/CICDataset/CICDarknet2020/Dataset/",
        target="data/raw/cic_darknet2020/",
        size="~1.2 GB (CSV)",
        note=("Requires accepting UNB's terms on the portal page. The file you "
              "want is Darknet.CSV (CICFlowMeter output, already labelled). "
              "The loader normalises CICFlowMeter column names automatically."),
        files=["Darknet.CSV"],
    ),
}


# ---------------------------------------------------------------------------
def human(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:,.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:,.1f} TB"


def download(url: str, destination: Path, timeout: int = 60,
             retries: int = 3) -> bool:
    """Download with resume support and a progress line.  Returns success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")

    try:
        head = requests.head(url, timeout=timeout, allow_redirects=True)
        total = int(head.headers.get("content-length", 0))
    except requests.RequestException as exc:
        log.error("Cannot reach %s (%s)", url, exc)
        return False

    if destination.exists() and total and destination.stat().st_size == total:
        log.info("%-46s already complete (%s)", destination.name, human(total))
        return True

    for attempt in range(1, retries + 1):
        existing = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        try:
            with requests.get(url, stream=True, timeout=timeout,
                              headers=headers) as response:
                if existing and response.status_code == 200:
                    existing = 0            # server ignored Range; restart
                elif response.status_code not in (200, 206):
                    log.error("HTTP %s for %s", response.status_code, url)
                    return False

                mode = "ab" if existing else "wb"
                downloaded = existing
                started = time.time()
                with open(part, mode) as fh:
                    for block in response.iter_content(chunk_size=1 << 20):
                        if not block:
                            continue
                        fh.write(block)
                        downloaded += len(block)
                        elapsed = max(time.time() - started, 1e-6)
                        rate = (downloaded - existing) / elapsed
                        pct = f"{100 * downloaded / total:5.1f}%" if total else "  ?  "
                        print(f"\r  {destination.name[:42]:42s} {pct} "
                              f"{human(downloaded)} @ {human(rate)}/s   ",
                              end="", flush=True)
            print()
            part.replace(destination)
            log.info("%-46s done (%s)", destination.name,
                     human(destination.stat().st_size))
            return True

        except (requests.RequestException, OSError) as exc:
            print()
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries,
                        destination.name, exc)
            time.sleep(2 * attempt)

    log.error("Giving up on %s", url)
    return False


# ---------------------------------------------------------------------------
def download_ctu13(root: Path, scenarios: list[int]) -> int:
    banner(f"CTU-13 - {len(scenarios)} scenario(s)")
    total_mb = sum(CTU13_SCENARIOS[s][2] for s in scenarios)
    log.info("Approximate total download: %d MB", total_mb)

    ok = 0
    for scenario in scenarios:
        if scenario not in CTU13_SCENARIOS:
            log.warning("No such CTU-13 scenario: %s", scenario)
            continue
        directory, filename, size_mb = CTU13_SCENARIOS[scenario]
        url = f"{CTU_BASE}/{directory}/{filename}"
        destination = root / f"scenario{scenario:02d}_{filename}"
        log.info("Scenario %-2d (%s, ~%d MB)", scenario, directory, size_mb)
        ok += download(url, destination)
    return ok


def print_manual(keys: list[str]) -> None:
    for key in keys:
        d = MANUAL[key]
        banner(f"{d.name} - manual download required")
        log.info("Why      : behind the University of New Brunswick's "
                 "registration/terms form, which a script must not bypass.")
        log.info("Portal   : %s", d.portal)
        log.info("Files    : %s", d.direct)
        log.info("Size     : %s", d.size)
        log.info("Save to  : %s", d.target)
        log.info("Wanted   : %s", ", ".join(d.files))
        log.info("Note     : %s", d.note)


def check(cfg) -> None:
    """Report what is present on disk and what the pipeline will use."""
    from encids.config import Paths

    paths = Paths.from_config(cfg)
    banner("Dataset status")
    rows = []
    for key in ("iscx_vpn2016", "cic_darknet2020", "ctu13"):
        directory = paths.raw / key
        files = [p for p in directory.rglob("*") if p.is_file()] \
            if directory.exists() else []
        size = sum(p.stat().st_size for p in files)
        rows.append((key, len(files), size, directory))

    for key, n_files, size, directory in rows:
        status = "READY" if n_files else "missing"
        log.info("%-18s %-8s %4d file(s)  %10s  %s", key, status, n_files,
                 human(size) if size else "-", directory)

    log.info("")
    log.info("The synthetic generator is always available, so the pipeline "
             "runs even with none of the above.")


# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="*",
                        choices=["ctu13", "iscx_vpn2016", "cic_darknet2020", "all"],
                        default=["all"])
    parser.add_argument("--scenarios", nargs="*", type=int, default=None,
                        help=f"CTU-13 scenarios to fetch (default: {CTU13_DEFAULT})")
    parser.add_argument("--full", action="store_true",
                        help="all 13 CTU-13 scenarios (~4.7 GB) instead of 5")
    parser.add_argument("--list", action="store_true",
                        help="show what is available and exit")
    parser.add_argument("--check", action="store_true",
                        help="report what is already on disk and exit")
    args = parser.parse_args()

    cfg = load_config()
    ensure_dirs(cfg)
    from encids.config import Paths
    paths = Paths.from_config(cfg)

    if args.check:
        check(cfg)
        return 0

    if args.list:
        banner("CTU-13 scenarios (direct download)")
        for scenario, (directory, filename, size_mb) in sorted(
                CTU13_SCENARIOS.items()):
            marker = "*" if scenario in CTU13_DEFAULT else " "
            log.info(" %s scenario %-2d  %5d MB  %s", marker, scenario, size_mb,
                     directory)
        log.info("   (* = downloaded by default)")
        print_manual(["iscx_vpn2016", "cic_darknet2020"])
        return 0

    selected = args.dataset
    if "all" in selected:
        selected = ["ctu13", "iscx_vpn2016", "cic_darknet2020"]

    if "ctu13" in selected:
        scenarios = args.scenarios or (list(CTU13_SCENARIOS) if args.full
                                       else CTU13_DEFAULT)
        download_ctu13(paths.raw / "ctu13", sorted(scenarios))

    manual = [k for k in ("iscx_vpn2016", "cic_darknet2020") if k in selected]
    if manual:
        print_manual(manual)

    log.info("")
    check(cfg)
    log.info("")
    log.info("Next: python scripts/prepare_data.py --force")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
