"""Build the unified feature table from whichever datasets are available.

Reads the ``datasets`` block of ``config.yaml``, runs each enabled loader, and
concatenates the results into one parquet file under ``data/processed``.

Datasets that are enabled but not yet downloaded are skipped with a warning
rather than an error, so the pipeline stays runnable while the multi-gigabyte
downloads are still in progress.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config, Paths, load_config
from ..features import schema
from ..features.build_features import assemble, summarise
from ..utils.logging_utils import banner, get_logger, timed
from . import synthetic
from .dataset_loaders import load_cic_darknet2020, load_ctu13, load_iscx_vpn2016

log = get_logger("data.build")

FEATURE_TABLE = "flow_features.parquet"


def build(cfg: Config | None = None, backend: str = "auto",
          force: bool = False, limit_pcaps: int | None = None) -> pd.DataFrame:
    """Assemble every enabled dataset into one feature table."""
    cfg = cfg or load_config()
    paths = Paths.from_config(cfg)
    out_path = paths.processed / FEATURE_TABLE

    if out_path.exists() and not force:
        log.info("Reusing cached feature table %s (--force to rebuild)", out_path)
        return pd.read_parquet(out_path)

    banner("Building the flow-feature table")
    frames: list[pd.DataFrame] = []
    ds = cfg["datasets"]

    # -- synthetic ---------------------------------------------------------
    if ds.get("synthetic", {}).get("enabled"):
        with timed("Generating synthetic flows", log):
            frames.append(synthetic.generate(
                n_flows=ds["synthetic"].get("n_flows", 60000),
                attack_fraction=ds["synthetic"].get("attack_fraction", 0.28),
                seed=cfg.get_path("project.seed", 42),
            ))

    # -- real datasets -----------------------------------------------------
    real_loaders = [
        ("iscx_vpn2016", lambda d: load_iscx_vpn2016(d, backend=backend,
                                                     limit=limit_pcaps)),
        ("cic_darknet2020", load_cic_darknet2020),
        ("ctu13", lambda d: load_ctu13(
            d, max_rows_per_file=ds.get("ctu13", {}).get("max_rows_per_file",
                                                         300_000))),
    ]
    for name, loader in real_loaders:
        entry = ds.get(name, {})
        if not entry.get("enabled"):
            continue
        directory = Path(entry.get("dir", f"data/raw/{name}"))
        if not directory.is_absolute():
            directory = paths.raw.parent.parent / directory
        if not directory.exists() or not any(directory.rglob("*")):
            log.warning("%s: %s is empty - run scripts/download_datasets.py",
                        name, directory)
            continue
        with timed(f"Loading {name}", log):
            frame = loader(directory)
        if not frame.empty:
            frame[schema.SOURCE_COLUMN] = name
            frames.append(frame)

    df = assemble(
        frames,
        min_packets=cfg.get_path("features.min_packets_per_flow", 2),
        ja3_buckets=cfg.get_path("features.ja3_hash_buckets", 256),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _to_parquet(df, out_path)
    log.info("Feature table -> %s (%d rows x %d cols)", out_path, *df.shape)

    stats = summarise(df)
    log.info("Malicious: %d/%d (%.1f%%) across %d classes",
             stats["n_malicious"], stats["n_flows"],
             100 * stats["malicious_rate"], stats["n_classes"])
    return df


def _to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet, coercing mixed-type object columns to string first."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str)
    out.to_parquet(path, index=False)


def load_feature_table(cfg: Config | None = None) -> pd.DataFrame:
    """Load the cached feature table, building it if it does not exist."""
    cfg = cfg or load_config()
    path = Paths.from_config(cfg).processed / FEATURE_TABLE
    if not path.exists():
        return build(cfg)
    return pd.read_parquet(path)
