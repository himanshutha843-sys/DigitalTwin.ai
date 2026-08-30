"""
dashboard/app.py
------------------
Streamlit control-room view for the DigitalTwin.ai prototype.

Three tabs, matching the pitch deck's "different stakeholders need
different views of the same twin":

  - Floor Supervisor : real-time station map + active drift alerts
  - Plant Manager    : ripple projections, inspection queue, weekly trend
  - Leadership       : rollout business case / impact summary

Run with:  streamlit run dashboard/app.py
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from simulate_line import build_station_list, simulate_telemetry, default_drift_scenario
from live_line_model import build_live_snapshot
from bottleneck_detection import detect_drift, project_ripple, summarize_alerts
from confidence_inference import infer_manual_station_state, combined_station_view
from defect_prediction import build_vehicle_features, train_defect_model, score_vehicles, inspection_queue

st.set_page_config(page_title="DigitalTwin.ai — Control Room", layout="wide", page_icon="🏭")

PURPLE = "#7c3aed"
PURPLE_LIGHT = "#c4b5fd"
AMBER = "#f59e0b"
RED = "#dc2626"
GREY = "#94a3b8"

STATUS_COLOR = {"nominal": "#16a34a", "watch": AMBER, "critical": RED, "no_data": GREY}
CONFIDENCE_COLOR = {"high": "#16a34a", "medium": AMBER, "low": GREY}


# ---------------------------------------------------------------------------
# Data pipeline (cached so the app doesn't re-simulate/re-train on every
# widget interaction -- only when the seed/capacity controls change)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Simulating line and running the twin's three layers...")
def run_pipeline(seed, capacity_per_shift):
    stations = build_station_list(seed=seed)
    telemetry = simulate_telemetry(stations, seed=seed, drift_events=default_drift_scenario())

    snapshot = build_live_snapshot(telemetry, stations)

    station_sequence = stations[["station_id", "sequence"]]
    drift_df = detect_drift(telemetry, station_sequence)
    ripple_df = project_ripple(drift_df, stations)
    alerts = summarize_alerts(drift_df, ripple_df, top_n=15)

    manual_inferred = infer_manual_station_state(drift_df, stations)
    combined_view = combined_station_view(drift_df, stations, manual_inferred)

    features = build_vehicle_features(telemetry)
    model, feature_cols, metrics = train_defect_model(features, random_state=seed)
    scored = score_vehicles(model, features, feature_cols)
    queue = inspection_queue(scored, capacity_per_shift=capacity_per_shift)

    return {
        "stations": stations, "telemetry": telemetry, "snapshot": snapshot,
        "drift_df": drift_df, "ripple_df": ripple_df, "alerts": alerts,
        "manual_inferred": manual_inferred, "combined_view": combined_view,
        "features": features, "metrics": metrics, "scored": scored, "queue": queue,
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.title("🏭 DigitalTwin.ai")
st.sidebar.caption("Predictive digital twin — vehicle assembly line")
st.sidebar.markdown("**Team Himanshu6030**")
seed = st.sidebar.slider("Simulation seed", 1, 100, 42, help="Re-roll the simulated line/production run")
capacity = st.sidebar.slider("Inspection capacity / shift", 5, 40, 15,
                              help="How many vehicles the inspection team can actually check per shift")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data shown is **simulated** (illustrative, not real enterprise data) — "
    "40 stations, 6 shifts, uneven sensor coverage, 3 injected drift events."
)

data = run_pipeline(seed, capacity)
stations, telemetry, snapshot = data["stations"], data["telemetry"], data["snapshot"]
drift_df, ripple_df, alerts = data["drift_df"], data["ripple_df"], data["alerts"]
manual_inferred, combined_view = data["manual_inferred"], data["combined_view"]
features, metrics, scored, queue = data["features"], data["metrics"], data["scored"], data["queue"]

st.title("Control Room")
st.caption("A live, honest picture of the line — not a perfect one.")

tab_floor, tab_manager, tab_leadership = st.tabs(
    ["👷 Floor Supervisor", "📋 Plant Manager", "📊 Leadership"]
)

# ---------------------------------------------------------------------------
# TAB 1 -- Floor Supervisor: real-time, in-the-moment signals
# ---------------------------------------------------------------------------
with tab_floor:
    st.subheader("Live line map")
    c1, c2, c3, c4 = st.columns(4)
    counts = snapshot.status.value_counts()
    c1.metric("Stations nominal", int(counts.get("nominal", 0)))
    c2.metric("Watch", int(counts.get("watch", 0)))
    c3.metric("Critical", int(counts.get("critical", 0)))
    c4.metric("No data (manual)", int(counts.get("no_data", 0)))

    fig = px.scatter(
        snapshot, x="sequence", y="zone", color="status",
        color_discrete_map=STATUS_COLOR, hover_name="station_id",
        hover_data={"latest_cycle_time_s": True, "baseline_cycle_time_s": True,
                    "sensor_coverage": True, "sequence": False},
        symbol="sensor_coverage",
        labels={"sequence": "Station sequence (body → paint → final assembly)"},
        title="Station status across the line (color = status, shape = sensor coverage)",
        height=350,
    )
    fig.update_traces(marker=dict(size=14))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Active drift alerts")
    if alerts.empty:
        st.success("No stations currently drifting above threshold.")
    else:
        for _, row in alerts.iterrows():
            severity = "🔴" if row.z_score > 4 else "🟠"
            with st.container(border=True):
                a, b, c = st.columns([2, 2, 2])
                a.markdown(f"{severity} **{row.station_id}** ({row.zone})")
                a.caption(f"z-score {row.z_score:.2f} · cycle {row.cycle_time_s:.1f}s vs baseline {row.rolling_mean:.1f}s")
                b.metric("Projected delay", f"{row.projected_delay_minutes:.1f} min")
                c.metric("Vehicles projected queued", f"{row.projected_vehicles_queued:.1f}")

    st.subheader("Confidence-aware coverage")
    st.caption("Manual/no-sensor stations are inferred from neighboring stations' timing and explicitly tagged low-confidence.")
    conf_summary = combined_view.groupby("confidence").station_id.nunique().reindex(
        ["high", "medium", "low"]).fillna(0).astype(int)
    fig2 = go.Figure(go.Bar(
        x=conf_summary.index, y=conf_summary.values,
        marker_color=[CONFIDENCE_COLOR[c] for c in conf_summary.index]
    ))
    fig2.update_layout(title="Stations by confidence level", height=280,
                        xaxis_title="confidence", yaxis_title="station count")
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 -- Plant Manager: weekly planning trends + targeted inspection
# ---------------------------------------------------------------------------
with tab_manager:
    st.subheader("Bottleneck drift over the run")
    if not drift_df[drift_df.drift_flag].empty:
        worst_station = drift_df[drift_df.drift_flag].station_id.value_counts().idxmax()
        sdf = drift_df[drift_df.station_id == worst_station].sort_values("unit_id")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=sdf.unit_id, y=sdf.cycle_time_s, name="cycle time",
                                   line=dict(color=PURPLE)))
        fig3.add_trace(go.Scatter(x=sdf.unit_id, y=sdf.rolling_mean, name="rolling baseline",
                                   line=dict(color=GREY, dash="dash")))
        flagged = sdf[sdf.drift_flag]
        fig3.add_trace(go.Scatter(x=flagged.unit_id, y=flagged.cycle_time_s, mode="markers",
                                   name="drift flagged", marker=dict(color=AMBER, size=6)))
        fig3.update_layout(title=f"Most-flagged station: {worst_station}", height=380,
                            xaxis_title="unit_id", yaxis_title="cycle time (s)")
        st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Projected downstream delay by station")
    if not ripple_df.empty:
        top_ripple = (ripple_df.groupby("station_id").projected_delay_minutes.max()
                      .sort_values(ascending=False).reset_index())
        fig4 = px.bar(top_ripple, x="projected_delay_minutes", y="station_id", orientation="h",
                      color_discrete_sequence=[PURPLE],
                      labels={"projected_delay_minutes": "minutes", "station_id": ""})
        fig4.update_layout(height=max(300, 20 * len(top_ripple)), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No ripple currently projected.")

    st.subheader(f"Ranked inspection queue (top {capacity} of {features.unit_id.nunique()} vehicles)")
    st.caption("Vehicles are ranked by defect-risk score from Layer 3, instead of uniform sampling.")
        display_queue = queue.rename(columns={
        "unit_id": "Unit", "defect_risk_score": "Risk score",
        "defect_flag": "Actually defective (sim ground truth)",
        "inspection_priority": "Priority"
    })

    def _risk_text_color(val):
        lo, hi = display_queue["Risk score"].min(), display_queue["Risk score"].max()
        norm = (val - lo) / (hi - lo + 1e-9)
        return "color: white" if norm > 0.55 else "color: black"

    styled_queue = (
        display_queue.style
        .background_gradient(subset=["Risk score"], cmap="Purples")
        .applymap(_risk_text_color, subset=["Risk score"])
    )

    st.dataframe(styled_queue, use_container_width=True, hide_index=True)
# ---------------------------------------------------------------------------
# TAB 3 -- Leadership: rollout business case / impact summary
# ---------------------------------------------------------------------------
with tab_leadership:
    st.subheader("Model validation")
    c1, c2, c3 = st.columns(3)
    auc = metrics["roc_auc"]
    c1.metric("Defect-risk model ROC-AUC", f"{auc:.3f}" if auc else "n/a",
              help="Measured on a held-out validation split, not the training data.")
    total_defects = int(features.defect_flag.sum())
    caught = int(queue.defect_flag.sum())
    baseline_expected = total_defects * (len(queue) / features.unit_id.nunique())
    c2.metric("Defects caught by targeted queue", f"{caught} / {total_defects}",
              help=f"Inspecting only {len(queue)} of {features.unit_id.nunique()} vehicles")
    lift = (caught / baseline_expected) if baseline_expected > 0 else float("nan")
    c3.metric("Lift over uniform random sampling", f"{lift:.1f}×" if baseline_expected > 0 else "n/a")

    st.subheader("Rollout roadmap")
    r1, r2, r3 = st.columns(3)
    with r1.container(border=True):
        st.markdown("**PHASE 1 — Shadow Mode**")
        st.caption("2–3 pilot stations. The twin watches and flags but makes no changes to inspection or line operations; output compared against what actually happened.")
    with r2.container(border=True):
        st.markdown("**PHASE 2 — Line-Wide Advisory**")
        st.caption("Bottleneck alerts and ripple projections go live across the full line, visible to plant supervisors — still a human decision every time.")
    with r3.container(border=True):
        st.markdown("**PHASE 3 — Targeted Routing**")
        st.caption("Defect-risk scores actively route which vehicles get pulled for inspection, replacing uniform sampling with targeted sampling.")

    st.subheader("What we'd expect to move")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Time to detect drift", "Minutes", help="Not a shift-end report")
    m2.metric("Downstream rework", "↓", help="Fewer vehicles queued behind a missed issue")
    m3.metric("Inspection targeting", "↑", help="More defects caught per inspection hour")
    m4.metric("Usable sensor coverage", "↑", help="More stations with a confidence-scored signal")
    st.caption("Directional goals for a pilot — not measured results.")

    st.subheader("Sensor coverage across the line")
    cov = stations.sensor_coverage.value_counts().reindex(["full", "partial", "manual"]).fillna(0)
    fig5 = go.Figure(go.Pie(labels=cov.index, values=cov.values,
                             marker_colors=[PURPLE, PURPLE_LIGHT, GREY], hole=0.5))
    fig5.update_layout(height=320, title="Stations by sensor coverage tier")
    st.plotly_chart(fig5, use_container_width=True)

    st.info(
        "**Risks we're watching:** false-positive alerts eroding floor-level trust in the twin, "
        "and classifier drift as the line changes — both are why Phase 1 stays advisory-only and "
        "every score ships with a confidence tag."
    )
