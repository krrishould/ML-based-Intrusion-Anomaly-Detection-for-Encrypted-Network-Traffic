"""Consistent, readable logging for every entry-point script."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager

_CONFIGURED = False


def get_logger(name: str = "encids", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        # When stdout is redirected to a file on Windows it defaults to the
        # ANSI code page (cp1252), and a single non-Latin-1 character in a log
        # line then raises UnicodeEncodeError and kills a long training run.
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        root = logging.getLogger("encids")
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("encids") else f"encids.{name}")


@contextmanager
def timed(message: str, logger: logging.Logger | None = None):
    """``with timed("training"): ...`` -> logs elapsed wall-clock time."""
    log = logger or get_logger()
    log.info("%s ...", message)
    start = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s done in %.2fs", message, time.perf_counter() - start)


def banner(text: str, logger: logging.Logger | None = None) -> None:
    log = logger or get_logger()
    line = "=" * max(60, len(text) + 4)
    log.info(line)
    log.info(text)
    log.info(line)
