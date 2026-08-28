"""
live_line_model.py
--------------------
Layer 1 of the digital twin: the base line model. Represents each
station as a node with its designed cycle time, current queue/status,
and most recent reading -- the "real-time map of the line" the other
two layers build on top of.
"""

import pandas as pd


def build_live_snapshot(telemetry, stations_df, as_of_unit=None):
    """
    Return one row per station describing its state as of a given point
    in production (defaults to the latest unit simulated). This is the
    structure a control-room dashboard would poll on a short interval.
    """
    if as_of_unit is None:
        as_of_unit = telemetry.unit_id.max()

    window = telemetry[telemetry.unit_id <= as_of_unit]
    latest = (
        window.sort_values("unit_id")
        .groupby("station_id")
        .tail(1)
        [["station_id", "zone", "sensor_coverage", "unit_id", "cycle_time_s"]]
        .rename(columns={"cycle_time_s": "latest_cycle_time_s", "unit_id": "latest_unit_id"})
    )

    snapshot = stations_df.merge(latest, on=["station_id", "zone", "sensor_coverage"], how="left")
    snapshot["status"] = snapshot.apply(_classify_status, axis=1)
    return snapshot.sort_values("sequence")


def _classify_status(row):
    if pd.isna(row.get("latest_cycle_time_s")):
        return "no_data"
    delta = row.latest_cycle_time_s - row.baseline_cycle_time_s
    if delta > 15:
        return "critical"
    if delta > 6:
        return "watch"
    return "nominal"
