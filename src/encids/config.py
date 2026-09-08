"""Configuration loading and path management.

Everything in the project reads its settings from ``config/config.yaml`` through
this module, so there is exactly one place to change a hyper-parameter.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/encids/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class Config(dict):
    """A dict with attribute access and dotted-path lookup."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get_path("supervised.xgboost.max_depth")``"""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def resolve(self, dotted: str) -> Path:
        """Resolve a configured relative path against the project root."""
        value = self.get_path(dotted)
        if value is None:
            raise KeyError(f"No path configured at '{dotted}'")
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


_CACHE: dict[str, Config] = {}


def load_config(path: str | os.PathLike | None = None) -> Config:
    """Load (and memoise) the YAML configuration."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    key = str(cfg_path.resolve())
    if key not in _CACHE:
        with open(cfg_path, "r", encoding="utf-8") as fh:
            _CACHE[key] = Config(yaml.safe_load(fh))
    return _CACHE[key]


def ensure_dirs(cfg: Config | None = None) -> None:
    """Create every directory referenced in the ``paths`` block."""
    cfg = cfg or load_config()
    for key in cfg["paths"]:
        cfg.resolve(f"paths.{key}").mkdir(parents=True, exist_ok=True)


def set_seed(seed: int | None = None) -> int:
    """Seed every RNG we use so results are reproducible across runs."""
    if seed is None:
        seed = load_config().get_path("project.seed", 42)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass
    return seed


@dataclass(frozen=True)
class Paths:
    """Convenience accessor for the standard project directories."""

    raw: Path
    interim: Path
    processed: Path
    live: Path
    models: Path
    reports: Path
    figures: Path
    metrics: Path

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "Paths":
        cfg = cfg or load_config()
        return cls(**{k: cfg.resolve(f"paths.{k}") for k in cfg["paths"]})
