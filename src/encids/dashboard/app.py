"""Streamlit monitoring dashboard.

Run with::

    streamlit run src/encids/dashboard/app.py

Shows live flows as they are captured and scored, which stage fired for each
alert, and the SHAP-derived reason.  The explanation panel is the point: an
analyst has to be able to see *why* a flow was flagged, not just that it was.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `src` importable when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from encids.config import Paths, load_config          # noqa: E402
from encids.live.capture import (                     # noqa: E402
    LiveCapture, capture_available, list_interfaces, replay_pcap, synthetic_stream,
)
from encids.live.scorer import build_scorer           # noqa: E402

CFG = load_config()
PATHS = Paths.from_config(CFG)

st.set_page_config(
    page_title=CFG.get_path("dashboard.title", "Encrypted-Traffic IDS"),
    page_icon="🛡️", layout="wide",
)

REASON_COLOURS = {
    "benign": "#4c9f70",
    "known-attack": "#c1442e",
    "anomalous": "#e08c3b",
    "known-attack+anomalous": "#8b2e8b",
}


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _state():
    ss = st.session_state
    ss.setdefault("scorer", None)
    ss.setdefault("capture", None)
    ss.setdefault("stop_event", None)
    ss.setdefault("running", False)
    ss.setdefault("source", "synthetic")
    ss.setdefault("error", "")
    return ss


@st.cache_resource(show_spinner="Loading trained models …")
def _load_scorer():
    return build_scorer(CFG)


def _models_exist() -> bool:
    return (PATHS.models / "stage1_supervised.joblib").exists()


# ---------------------------------------------------------------------------
# Control actions
# ---------------------------------------------------------------------------
def start_monitoring(source: str, interface: str | None, pcap: str | None,
                     speed: float) -> None:
    ss = _state()
    scorer = _load_scorer()
    scorer.reset()
    ss.scorer = scorer
    ss.error = ""

    idle = CFG.get_path("live.idle_timeout", 15)
    active = CFG.get_path("live.active_timeout", 120)

    if source == "Live interface":
        capture = LiveCapture(
            interface=interface or None,
            bpf_filter=CFG.get_path("live.bpf_filter", ""),
            idle_timeout=idle, active_timeout=active,
        )
        capture.start(scorer.score_batch)
        ss.capture = capture

    elif source == "Replay pcap":
        if not pcap or not Path(pcap).exists():
            ss.error = f"pcap not found: {pcap}"
            return
        stop = threading.Event()
        ss.stop_event = stop
        threading.Thread(
            target=replay_pcap,
            args=(pcap, scorer.score_batch, idle, active, speed),
            daemon=True,
        ).start()

    else:  # Simulated traffic
        stop = threading.Event()
        ss.stop_event = stop
        threading.Thread(
            target=synthetic_stream,
            args=(scorer.score_batch,),
            kwargs={"interval": 2.0, "flows_per_tick": 8, "stop_event": stop,
                    "seed": int(time.time()) % 10_000},
            daemon=True,
        ).start()

    ss.running = True
    ss.source = source


def stop_monitoring() -> None:
    ss = _state()
    if ss.capture is not None:
        ss.capture.stop()
        ss.capture = None
    if ss.stop_event is not None:
        ss.stop_event.set()
        ss.stop_event = None
    ss.running = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar() -> None:
    ss = _state()
    st.sidebar.title("🛡️ Controls")

    if not _models_exist():
        st.sidebar.error("No trained model found.\n\nRun:\n\n"
                         "`python scripts/train_model.py`")
        return

    can_capture, why = capture_available()
    options = ["Simulated traffic", "Replay pcap"]
    if can_capture:
        options.insert(0, "Live interface")

    source = st.sidebar.radio("Traffic source", options,
                              disabled=ss.running,
                              help="Live capture needs Npcap (Windows) or "
                                   "libpcap (Linux/macOS) plus admin rights.")

    interface = pcap = None
    speed = 0.0
    if source == "Live interface":
        interfaces = list_interfaces()
        labels = [f"{i['description']}" for i in interfaces]
        if labels:
            choice = st.sidebar.selectbox("Interface", labels, disabled=ss.running)
            interface = interfaces[labels.index(choice)]["name"]
    elif source == "Replay pcap":
        default_dir = PATHS.raw
        found = sorted(str(p) for ext in ("*.pcap", "*.pcapng")
                       for p in default_dir.rglob(ext))
        pcap = (st.sidebar.selectbox("Capture file", found, disabled=ss.running)
                if found else st.sidebar.text_input("Path to .pcap",
                                                    disabled=ss.running))
        speed = st.sidebar.slider("Replay speed (x real time, 0 = as fast as "
                                  "possible)", 0.0, 50.0, 0.0, 1.0,
                                  disabled=ss.running)
    else:
        st.sidebar.caption("Generates flows with the same statistical profiles "
                           "used for training. No capture privileges needed.")

    if not can_capture:
        st.sidebar.info(f"Live capture unavailable — {why}")

    col1, col2 = st.sidebar.columns(2)
    if col1.button("▶ Start", disabled=ss.running, use_container_width=True):
        start_monitoring(source, interface, pcap, speed)
        st.rerun()
    if col2.button("⏹ Stop", disabled=not ss.running, use_container_width=True):
        stop_monitoring()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Display")
    ss.show_alerts_only = st.sidebar.checkbox("Alerts only", value=False)
    ss.max_rows = st.sidebar.slider("Rows shown", 25, 500,
                                    CFG.get_path("dashboard.max_rows", 300), 25)
    ss.auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)

    st.sidebar.divider()
    meta_path = PATHS.models / "training_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        st.sidebar.caption(
            f"**Model**\n\n"
            f"Stage 1: `{meta.get('stage1_model')}`  \n"
            f"Stage 2: `{meta.get('stage2_model')}`  \n"
            f"Fusion: `{meta.get('fusion_rule')}`  \n"
            f"Features: {meta.get('n_features')} "
            f"(TLS: {'on' if meta.get('use_tls_metadata') else 'off'})"
        )


# ---------------------------------------------------------------------------
# Main panels
# ---------------------------------------------------------------------------
def header_metrics(stats: dict) -> None:
    cols = st.columns(5)
    cols[0].metric("Flows scored", f"{stats['total_flows']:,}")
    cols[1].metric("Alerts", f"{stats['total_alerts']:,}")
    cols[2].metric("Alert rate", f"{100 * stats['alert_rate']:.1f}%")
    cols[3].metric("Throughput", f"{stats['flows_per_second']:.1f} flows/s")
    cols[4].metric("Scoring latency", f"{stats['mean_latency_ms']:.2f} ms/flow")


def charts(df: pd.DataFrame) -> None:
    import plotly.express as px

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Why flows were flagged")
        counts = df["reason"].value_counts().reset_index()
        counts.columns = ["reason", "count"]
        fig = px.bar(counts, x="count", y="reason", orientation="h",
                     color="reason", color_discrete_map=REASON_COLOURS)
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Risk over time")
        plot_df = df.head(200).iloc[::-1].reset_index(drop=True)
        plot_df["n"] = plot_df.index
        fig = px.scatter(plot_df, x="n", y="risk", color="reason",
                         color_discrete_map=REASON_COLOURS,
                         hover_data=["predicted_family", "dst_ip", "sni"])
        fig.add_hline(y=0.5, line_dash="dash", line_color="grey")
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="flow (most recent on the right)",
                          yaxis_title="risk")
        st.plotly_chart(fig, use_container_width=True)


def alert_table(df: pd.DataFrame, max_rows: int) -> None:
    st.subheader("Flows")
    view = df.head(max_rows).copy()
    for col in ("p_malicious", "risk", "anomaly_sigma"):
        if col in view.columns:
            view[col] = view[col].astype(float).round(3)
    columns = [c for c in ("timestamp", "src_ip", "src_port", "dst_ip", "dst_port",
                           "sni", "total_packets", "total_bytes", "reason",
                           "predicted_family", "p_malicious", "anomaly_sigma",
                           "risk") if c in view.columns]

    st.dataframe(
        view[columns],
        use_container_width=True, height=380, hide_index=True,
        column_config={
            "risk": st.column_config.ProgressColumn(
                "risk", min_value=0.0, max_value=1.0, format="%.2f"),
            "sni": st.column_config.TextColumn("hostname (SNI)"),
        },
    )


def explanation_panel(df: pd.DataFrame) -> None:
    """The explainability view - one alert, expanded into its drivers."""
    alerts = df[df["alert"] == 1]
    if alerts.empty:
        st.info("No alerts yet. Explanations appear here as flows are flagged.")
        return

    st.subheader("Alert explanations")
    labels = [
        f"{r['timestamp']}  ·  {r['src_ip']}:{r['src_port']} → "
        f"{r['dst_ip']}:{r['dst_port']}  ·  {r['reason']}  ·  risk {r['risk']:.2f}"
        for _, r in alerts.head(30).iterrows()
    ]
    choice = st.selectbox("Select an alert", labels, label_visibility="collapsed")
    row = alerts.iloc[labels.index(choice)]

    left, right = st.columns([3, 2])
    with left:
        colour = REASON_COLOURS.get(str(row["reason"]), "#666")
        st.markdown(
            f"<div style='padding:14px 16px;border-left:5px solid {colour};"
            f"background:rgba(128,128,128,0.10);border-radius:6px'>"
            f"<b>{row['reason']}</b><br>{row.get('explanation') or '—'}</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        meta = pd.DataFrame({
            "field": ["source", "destination", "hostname (SNI)", "packets",
                      "bytes", "duration (ms)", "TLS", "predicted family",
                      "P(malicious)", "anomaly (σ from normal)"],
            "value": [
                f"{row['src_ip']}:{row['src_port']}",
                f"{row['dst_ip']}:{row['dst_port']}",
                row.get("sni") or "—",
                f"{row.get('total_packets', 0):,.0f}",
                f"{row.get('total_bytes', 0):,.0f}",
                f"{row.get('duration_ms', 0):,.0f}",
                "yes" if row.get("is_tls") else "no",
                row.get("predicted_family", "—"),
                f"{float(row.get('p_malicious', 0)):.3f}",
                f"{float(row.get('anomaly_sigma', 0)):.2f}",
            ],
        })
        st.dataframe(meta, hide_index=True, use_container_width=True)

    with right:
        st.caption("Feature contributions")
        raw = row.get("top_features")
        if raw:
            import plotly.express as px

            features = pd.DataFrame(json.loads(raw))
            features = features.iloc[::-1]
            fig = px.bar(features, x="w", y="f", orientation="h",
                         color=features["w"] > 0,
                         color_discrete_map={True: "#c1442e", False: "#3b6ea5"})
            fig.update_layout(showlegend=False, height=300,
                              margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="contribution to the decision",
                              yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("—")


# ---------------------------------------------------------------------------
def main() -> None:
    ss = _state()
    st.title(CFG.get_path("dashboard.title", "Encrypted-Traffic IDS"))
    st.caption("Two-stage detection on encrypted traffic — flow statistics + "
               "TLS handshake metadata, no payload decryption.")

    sidebar()

    if ss.error:
        st.error(ss.error)
    if not _models_exist():
        st.warning("Train the model first: `python scripts/train_model.py`")
        return
    if ss.scorer is None:
        st.info("Choose a traffic source in the sidebar and press **Start**.")
        return

    scorer = ss.scorer
    header_metrics(scorer.stats())

    df = scorer.snapshot()
    if df.empty:
        st.info("Waiting for the first completed flows … "
                "(a flow is emitted once it goes idle)")
    else:
        if getattr(ss, "show_alerts_only", False):
            df = df[df["alert"] == 1]
        if df.empty:
            st.success("No alerts in the current window.")
        else:
            charts(df)
            alert_table(df, getattr(ss, "max_rows", 300))
            st.divider()
            explanation_panel(df)

    if ss.running and getattr(ss, "auto_refresh", True):
        time.sleep(CFG.get_path("dashboard.refresh_seconds", 3))
        st.rerun()


main()
