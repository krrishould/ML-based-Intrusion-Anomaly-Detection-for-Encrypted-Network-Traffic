"""Real-time scoring: capture -> features -> two stages -> alert store.

The scorer sits between the capture thread and the dashboard.  It owns the
trained detector, scores each batch of completed flows, attaches a SHAP-based
explanation to anything it flags, and appends the result to a rolling store
that the dashboard polls.

Explanations are computed for alerts only.  Explaining every benign flow would
dominate the per-flow latency budget and nobody would ever read them.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import Config, Paths, load_config
from ..features import schema
from ..utils.logging_utils import get_logger

log = get_logger("live.scorer")

DISPLAY_COLUMNS = [
    "timestamp", "flow_id", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "duration_ms", "total_packets", "total_bytes", "is_tls", "sni",
    "alert", "reason", "predicted_family", "p_malicious", "anomaly_score",
    "anomaly_sigma", "risk", "explanation", "top_features",
]


@dataclass
class LiveScorer:
    """Scores batches of live flows and keeps a rolling window of results."""

    detector: Any
    explainer: Any = None
    max_rows: int = 5000
    output_csv: Path | None = None
    ja3_buckets: int = 256
    explain_alerts: bool = True

    rows: deque = field(default_factory=lambda: deque(maxlen=5000))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    total_flows: int = 0
    total_alerts: int = 0
    started_at: float = field(default_factory=time.time)
    _latencies: deque = field(default_factory=lambda: deque(maxlen=500))

    def __post_init__(self) -> None:
        self.rows = deque(maxlen=self.max_rows)
        if self.output_csv:
            self.output_csv = Path(self.output_csv)
            self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # -- scoring -----------------------------------------------------------
    def score_batch(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Score one batch of completed flows; returns the annotated rows."""
        if not records:
            return pd.DataFrame()

        started = time.perf_counter()
        df = pd.DataFrame.from_records(records)
        # Fingerprint rarity is looked up in the frequency table frozen at
        # training time. Recomputing it from this batch would make every
        # fingerprint in a small live window look rare and silently destroy the
        # feature - see FeaturePipeline.apply_fingerprints.
        df = self.detector.pipeline.apply_fingerprints(
            df, n_buckets=self.ja3_buckets)

        X = self.detector.pipeline.transform(df)
        verdicts = self.detector.score_matrix(X, index=df.index)

        out = pd.concat([df.reset_index(drop=True),
                         verdicts.reset_index(drop=True)], axis=1)
        out["timestamp"] = pd.Timestamp.now().isoformat(timespec="seconds")
        out["explanation"] = ""
        out["top_features"] = ""

        if self.explain_alerts and self.explainer is not None:
            alert_idx = out.index[out["alert"] == 1].tolist()
            if alert_idx:
                try:
                    explanations = self.explainer.explain(
                        X[alert_idx], verdicts.iloc[alert_idx],
                        raw=df.iloc[alert_idx])
                    for pos, exp in zip(alert_idx, explanations):
                        out.at[pos, "explanation"] = exp["summary"]
                        out.at[pos, "top_features"] = json.dumps(
                            [{"f": c["label"], "w": round(c["weight"], 4),
                              "sigma": round(c["deviation_sigma"], 2)}
                             for c in exp["contributions"][:4]])
                except Exception as exc:
                    log.warning("Explanation failed for this batch: %s", exc)

        elapsed = time.perf_counter() - started
        self._latencies.append(elapsed / max(len(out), 1))

        with self._lock:
            self.total_flows += len(out)
            self.total_alerts += int(out["alert"].sum())
            for record in out.to_dict(orient="records"):
                self.rows.append(record)

        if self.output_csv is not None:
            self._append_csv(out)

        n_alerts = int(out["alert"].sum())
        if n_alerts:
            log.info("Batch: %d flows, %d alert(s) [%s]", len(out), n_alerts,
                     ", ".join(sorted(set(out.loc[out['alert'] == 1, 'reason']))))
        return out

    # -- output ------------------------------------------------------------
    def _append_csv(self, df: pd.DataFrame) -> None:
        columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
        header = not self.output_csv.exists()
        df[columns].to_csv(self.output_csv, mode="a", header=header, index=False)

    def snapshot(self) -> pd.DataFrame:
        """Current rolling window, newest first."""
        with self._lock:
            rows = list(self.rows)
        if not rows:
            return pd.DataFrame(columns=DISPLAY_COLUMNS)
        df = pd.DataFrame(rows)
        columns = [c for c in DISPLAY_COLUMNS if c in df.columns]
        return df[columns].iloc[::-1].reset_index(drop=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            flows, alerts = self.total_flows, self.total_alerts
        latency_ms = (1000 * sum(self._latencies) / len(self._latencies)
                      if self._latencies else 0.0)
        uptime = max(time.time() - self.started_at, 1e-6)
        return {
            "total_flows": flows,
            "total_alerts": alerts,
            "alert_rate": alerts / flows if flows else 0.0,
            "flows_per_second": flows / uptime,
            "mean_latency_ms": round(latency_ms, 3),
            "uptime_seconds": round(uptime, 1),
        }

    def reset(self) -> None:
        with self._lock:
            self.rows.clear()
            self.total_flows = self.total_alerts = 0
            self.started_at = time.time()


# ---------------------------------------------------------------------------
def build_scorer(cfg: Config | None = None, model_dir: str | Path | None = None
                 ) -> LiveScorer:
    """Load the trained detector and wrap it in a :class:`LiveScorer`."""
    from ..train import load_artefacts

    cfg = cfg or load_config()
    paths = Paths.from_config(cfg)
    detector, explainer = load_artefacts(cfg, model_dir)
    return LiveScorer(
        detector=detector,
        explainer=explainer,
        max_rows=cfg.get_path("live.max_buffered_flows", 5000),
        output_csv=paths.live / "live_flows.csv",
        ja3_buckets=cfg.get_path("features.ja3_hash_buckets", 256),
        explain_alerts=cfg.get_path("explain.enabled", True),
    )


def run_live(cfg: Config | None = None, source: str = "auto",
             pcap: str | Path | None = None, duration: float | None = None,
             interface: str | None = None, speed: float = 0.0) -> LiveScorer:
    """Run the capture -> score loop.

    ``source`` is one of ``auto``, ``interface``, ``pcap`` or ``synthetic``.
    ``auto`` sniffs if capture is available and falls back to synthetic.
    """
    from .capture import LiveCapture, capture_available, replay_pcap, synthetic_stream

    cfg = cfg or load_config()
    scorer = build_scorer(cfg)
    idle = cfg.get_path("live.idle_timeout", 15)
    active = cfg.get_path("live.active_timeout", 120)

    if source == "auto":
        ok, why = capture_available()
        if pcap:
            source = "pcap"
        elif ok:
            source = "interface"
        else:
            log.warning("Live capture unavailable (%s) - using the synthetic "
                        "source instead", why)
            source = "synthetic"

    log.info("Live pipeline source: %s", source)

    if source == "pcap":
        if not pcap:
            raise ValueError("source='pcap' requires a pcap path")
        replay_pcap(pcap, scorer.score_batch, idle, active, speed=speed)
        return scorer

    if source == "synthetic":
        stop = threading.Event()
        thread = threading.Thread(
            target=synthetic_stream,
            args=(scorer.score_batch,),
            kwargs={"interval": cfg.get_path("live.snapshot_interval", 3),
                    "stop_event": stop},
            daemon=True)
        thread.start()
        _wait(duration, stop)
        stop.set()
        return scorer

    capture = LiveCapture(
        interface=interface or cfg.get_path("live.interface"),
        bpf_filter=cfg.get_path("live.bpf_filter", ""),
        idle_timeout=idle, active_timeout=active,
    )
    capture.start(scorer.score_batch)
    try:
        _wait(duration, threading.Event())
    finally:
        capture.stop()
    return scorer


def _wait(duration: float | None, stop: threading.Event) -> None:
    """Block for ``duration`` seconds, or until interrupted."""
    try:
        if duration:
            stop.wait(duration)
        else:
            while not stop.is_set():
                stop.wait(1)
    except KeyboardInterrupt:
        log.info("Interrupted by user")
