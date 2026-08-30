"""
app.py
------
Streamlit dashboard for the DigitalTwin.ai Round 2 prototype.

Run:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from twin_model import (
    compute_business_metrics,
    run_pipeline,
    score_defect_risk,
    station_snapshot_at,
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

STATUS_COLORS = {
    "nominal": "#227a5f",
    "watch": "#d18b00",
    "critical": "#b42318",
    "low_confidence": "#667085",
}
AREA_ORDER = ["Body", "Paint", "Final Assembly"]
LINE_BLUE = "#2563eb"
TEAL = "#0f766e"
AMBER = "#d18b00"
RED = "#b42318"
GRAY = "#667085"


st.set_page_config(
    page_title="DigitalTwin.ai Control Room",
    page_icon="DT",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        border: 1px solid #e4e7ec;
        border-radius: 8px;
        padding: 12px 14px;
        background: #ffffff;
    }
    div[data-testid="stMetric"] label {
        color: #475467;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Running the digital twin pipeline")
def load_pipeline(seed: int):
    return run_pipeline(data_dir=DATA_DIR, output_dir=None, threshold=0.35, random_state=seed)


def money(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def active_bottleneck_window(
    bottlenecks: pd.DataFrame,
    station_master: pd.DataFrame,
    current_index: int,
    lookback: int = 30,
) -> pd.DataFrame:
    window = bottlenecks[
        (bottlenecks["Vehicle_Global_Index"] >= current_index - lookback)
        & (bottlenecks["Vehicle_Global_Index"] <= current_index)
        & (bottlenecks["Bottleneck_Alert"])
    ]
    if window.empty:
        return pd.DataFrame()

    alerts = window.groupby("Station_ID").agg(
        Station_Number=("Station_Number", "first"),
        Process_Area=("Process_Area", "first"),
        Operation=("Operation", "first"),
        Sensor_Coverage=("Sensor_Coverage", "first"),
        Alert_Count=("Bottleneck_Alert", "sum"),
        Max_Z_Score=("Cycle_Z_Score", "max"),
        Max_Projected_Delay=("Projected_Delay_Minutes", "max"),
        Max_Queued_Vehicles=("Estimated_Queued_Vehicles", "max"),
        Last_Seen=("Timestamp", "max"),
    )
    alerts = alerts.reset_index()
    station_order = station_master[["Station_ID", "Station_Number"]]
    alerts = alerts.drop(columns=["Station_Number"]).merge(station_order, on="Station_ID", how="left")
    return alerts.sort_values(["Max_Z_Score", "Max_Projected_Delay"], ascending=False)


def current_wip_risks(
    scored: pd.DataFrame,
    current_index: int,
    horizon: int,
    threshold: float,
) -> pd.DataFrame:
    wip = scored[
        (scored["Vehicle_Global_Index"] >= current_index)
        & (scored["Vehicle_Global_Index"] <= current_index + horizon)
    ].copy()
    if wip.empty:
        return wip
    wip["Alert"] = wip["Defect_Probability"] >= threshold
    return wip.sort_values("Defect_Probability", ascending=False)


def build_daily_summary(
    vehicle_summary: pd.DataFrame,
    bottlenecks: pd.DataFrame,
) -> pd.DataFrame:
    daily = vehicle_summary.groupby("Production_Day").agg(
        Vehicles=("Vehicle_ID", "nunique"),
        Defects=("End_of_Line_Quality", lambda values: (values == "Fail").sum()),
    )
    daily["Defect_Rate"] = daily["Defects"] / daily["Vehicles"]

    alert_daily = bottlenecks[bottlenecks["Bottleneck_Alert"]].groupby("Production_Day").agg(
        Bottleneck_Alerts=("Bottleneck_Alert", "sum"),
        Projected_Delay_Minutes=("Projected_Delay_Minutes", "sum"),
    )
    daily = daily.join(alert_daily, how="left").fillna(0).reset_index()
    return daily


seed = st.sidebar.number_input("Model seed", min_value=1, max_value=999, value=42, step=1)
alert_threshold = st.sidebar.slider("Defect alert threshold", 0.10, 0.80, 0.35, 0.05)
wip_horizon = st.sidebar.slider("WIP horizon", 10, 80, 45, 5)

pipeline = load_pipeline(int(seed))
station_master = pipeline["station_master"]
vehicle_summary = pipeline["vehicle_summary"]
features = pipeline["features"]
model = pipeline["model"]
metrics = pipeline["metrics"]
bottlenecks = pipeline["bottleneck_events"]
station_health = pipeline["station_health"]
imputed_events = pipeline["imputed_events"]

scored = score_defect_risk(model, features, threshold=alert_threshold)
business = compute_business_metrics(imputed_events, vehicle_summary, scored)

max_vehicle_index = int(vehicle_summary["Vehicle_Global_Index"].max())
default_index = min(max_vehicle_index, int(max_vehicle_index * 0.55))
current_index = st.sidebar.slider(
    "Current WIP vehicle index",
    0,
    max_vehicle_index,
    default_index,
    5,
)

st.title("DigitalTwin.ai Control Room")
st.caption("Read-only predictive twin for a mixed-model vehicle assembly line")

tab_supervisor, tab_manager, tab_leadership = st.tabs(
    ["Supervisor View", "Plant Manager View", "Leadership View"]
)

with tab_supervisor:
    snapshot = station_snapshot_at(
        bottlenecks,
        station_master,
        current_vehicle_index=current_index,
        lookback_vehicles=30,
    )
    status_counts = snapshot["Station_Status"].value_counts()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Line stations", f"{len(snapshot)}")
    c2.metric("Nominal", int(status_counts.get("nominal", 0)))
    c3.metric("Watch", int(status_counts.get("watch", 0)))
    c4.metric("Critical", int(status_counts.get("critical", 0)))
    c5.metric("Low confidence", int(status_counts.get("low_confidence", 0)))

    map_df = snapshot.copy()
    map_df["Alert_Size"] = map_df["Window_Alert_Count"].clip(lower=0) + 1
    line_map = px.scatter(
        map_df,
        x="Station_Number",
        y="Process_Area",
        color="Station_Status",
        symbol="Sensor_Coverage",
        size="Alert_Size",
        hover_name="Station_ID",
        hover_data={
            "Operation": True,
            "Sensor_Coverage": True,
            "Cycle_Time": ":.1f",
            "Baseline_Cycle_Time": ":.1f",
            "Cycle_Z_Score": ":.2f",
            "Window_Alert_Count": True,
            "Alert_Size": False,
            "Station_Number": False,
        },
        color_discrete_map=STATUS_COLORS,
        category_orders={"Process_Area": AREA_ORDER},
        height=330,
    )
    line_map.update_traces(marker=dict(line=dict(width=0.6, color="#ffffff")))
    line_map.update_layout(
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Station sequence",
        yaxis_title="",
        legend_title="",
    )
    st.plotly_chart(line_map, use_container_width=True)

    alerts = active_bottleneck_window(bottlenecks, station_master, current_index)
    left, right = st.columns([1.05, 1])

    with left:
        st.subheader("Active Bottleneck Alerts")
        if alerts.empty:
            st.success("No active bottleneck alert in the current window.")
        else:
            display_alerts = alerts[
                [
                    "Station_ID",
                    "Station_Number",
                    "Process_Area",
                    "Operation",
                    "Sensor_Coverage",
                    "Alert_Count",
                    "Max_Z_Score",
                    "Max_Projected_Delay",
                    "Max_Queued_Vehicles",
                ]
            ].head(10)
            st.dataframe(
                display_alerts,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Max_Z_Score": st.column_config.NumberColumn("Max z", format="%.2f"),
                    "Max_Projected_Delay": st.column_config.NumberColumn("Delay min", format="%.1f"),
                    "Max_Queued_Vehicles": st.column_config.NumberColumn("Queued", format="%.1f"),
                },
            )

    with right:
        st.subheader("Current WIP Defect Warnings")
        wip = current_wip_risks(scored, current_index, wip_horizon, alert_threshold)
        high_risk = wip[wip["Alert"]].head(10)
        if high_risk.empty:
            st.info("No vehicle exceeds the active alert threshold.")
            high_risk = wip.head(5)
        st.dataframe(
            high_risk[
                [
                    "Vehicle_ID",
                    "Vehicle_Model",
                    "Shift",
                    "Risk_Percent",
                    "Likely_Driver",
                    "Telemetry_Confidence_Mean",
                ]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Risk_Percent": st.column_config.ProgressColumn(
                    "Risk",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "Telemetry_Confidence_Mean": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.2f",
                ),
            },
        )

with tab_manager:
    daily = build_daily_summary(vehicle_summary, bottlenecks)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Vehicles this week", int(daily["Vehicles"].sum()))
    c2.metric("EOL defects", int(daily["Defects"].sum()))
    c3.metric("Avg defect rate", percent(daily["Defects"].sum() / daily["Vehicles"].sum()))
    c4.metric("Bottleneck alerts", int(daily["Bottleneck_Alerts"].sum()))

    weekly_fig = go.Figure()
    weekly_fig.add_trace(
        go.Bar(
            x=daily["Production_Day"],
            y=daily["Bottleneck_Alerts"],
            name="Bottleneck alerts",
            marker_color=AMBER,
            yaxis="y",
        )
    )
    weekly_fig.add_trace(
        go.Scatter(
            x=daily["Production_Day"],
            y=daily["Defect_Rate"] * 100,
            name="Defect rate",
            mode="lines+markers",
            line=dict(color=RED, width=3),
            yaxis="y2",
        )
    )
    weekly_fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Production day",
        yaxis=dict(title="Bottleneck alerts"),
        yaxis2=dict(title="Defect rate %", overlaying="y", side="right"),
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(weekly_fig, use_container_width=True)

    health_left, health_right = st.columns([1, 1])

    with health_left:
        st.subheader("Station Health Trend")
        top_health = station_health.head(12).sort_values("Bottleneck_Alert_Count")
        health_fig = px.bar(
            top_health,
            x="Bottleneck_Alert_Count",
            y="Station_ID",
            color="Process_Area",
            orientation="h",
            hover_data=["Operation", "P95_Cycle_Time", "Missing_Telemetry_Rate"],
            color_discrete_sequence=[TEAL, LINE_BLUE, AMBER],
            height=420,
        )
        health_fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Alert count",
            yaxis_title="",
            legend_title="",
        )
        st.plotly_chart(health_fig, use_container_width=True)

    with health_right:
        st.subheader("MTBF Analytics")
        mtbf_table = station_health[
            [
                "Station_ID",
                "Process_Area",
                "Operation",
                "Bottleneck_Alert_Count",
                "Failure_Event_Count",
                "MTBF_Hours",
                "Avg_Telemetry_Confidence",
            ]
        ].head(15)
        st.dataframe(
            mtbf_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Bottleneck_Alert_Count": st.column_config.NumberColumn("Alerts", format="%d"),
                "Failure_Event_Count": st.column_config.NumberColumn("Events", format="%d"),
                "MTBF_Hours": st.column_config.NumberColumn("MTBF hrs", format="%.1f"),
                "Avg_Telemetry_Confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            },
        )

with tab_leadership:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("OEE roll-up", percent(business["oee"]))
    c2.metric("Availability", percent(business["availability"]))
    c3.metric("Performance", percent(business["performance"]))
    c4.metric("Quality", percent(business["quality"]))

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    s2.metric("Average precision", f"{metrics['average_precision']:.3f}")
    s3.metric("High-risk alerts", int(business["high_risk_alerts"]))
    s4.metric("True positive alerts", int(business["true_positive_alerts"]))

    savings_low = business["estimated_annual_savings_low"]
    savings_high = business["estimated_annual_savings_high"]
    assumed_pilot_cost = 350_000.0
    payback_months = (assumed_pilot_cost / max(savings_low, 1.0)) * 12.0

    business_left, business_right = st.columns([1, 1])

    with business_left:
        st.subheader("Estimated Cost Savings")
        savings_fig = go.Figure(
            go.Bar(
                x=["Conservative", "Upside"],
                y=[savings_low, savings_high],
                marker_color=[TEAL, LINE_BLUE],
                text=[money(savings_low), money(savings_high)],
                textposition="outside",
            )
        )
        savings_fig.update_layout(
            height=330,
            margin=dict(l=10, r=10, t=25, b=10),
            yaxis_title="Annual savings",
            xaxis_title="",
        )
        st.plotly_chart(savings_fig, use_container_width=True)
        st.metric("Estimated pilot payback", f"{payback_months:.1f} months")

    with business_right:
        st.subheader("Sensor Coverage Mix")
        coverage = station_master["Sensor_Coverage"].value_counts().reset_index()
        coverage.columns = ["Sensor_Coverage", "Stations"]
        coverage_fig = px.pie(
            coverage,
            values="Stations",
            names="Sensor_Coverage",
            hole=0.48,
            color_discrete_sequence=[TEAL, LINE_BLUE, AMBER, GRAY],
            height=330,
        )
        coverage_fig.update_layout(margin=dict(l=10, r=10, t=25, b=10), legend_title="")
        st.plotly_chart(coverage_fig, use_container_width=True)

    threshold_df = pd.DataFrame(metrics["threshold_table"])
    st.subheader("False-Alarm Threshold Tradeoff")
    threshold_fig = go.Figure()
    threshold_fig.add_trace(
        go.Scatter(
            x=threshold_df["threshold"],
            y=threshold_df["precision"] * 100,
            name="Precision",
            mode="lines+markers",
            line=dict(color=TEAL, width=3),
        )
    )
    threshold_fig.add_trace(
        go.Scatter(
            x=threshold_df["threshold"],
            y=threshold_df["recall"] * 100,
            name="Recall",
            mode="lines+markers",
            line=dict(color=LINE_BLUE, width=3),
        )
    )
    threshold_fig.add_trace(
        go.Scatter(
            x=threshold_df["threshold"],
            y=threshold_df["false_alarm_rate"] * 100,
            name="False-alarm rate",
            mode="lines+markers",
            line=dict(color=AMBER, width=3),
        )
    )
    threshold_fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Probability threshold",
        yaxis_title="Validation metric %",
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(threshold_fig, use_container_width=True)
