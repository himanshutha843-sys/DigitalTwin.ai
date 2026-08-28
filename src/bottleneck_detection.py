"""
bottleneck_detection.py
------------------------
Layer 2 of the digital twin: control-chart style drift detection against
a rolling baseline, plus a simple queueing projection to estimate how far
a detected slowdown will ripple downstream (in vehicles and minutes).
"""

import numpy as np
import pandas as pd


def compute_rolling_baseline(telemetry, station_id, window_units=150):
    """Rolling mean/std of cycle time for one station, used as the
    'normal' band a control chart compares live readings against."""
    df = telemetry[telemetry.station_id == station_id].sort_values("unit_id").copy()
    df["rolling_mean"] = df.cycle_time_s.rolling(window_units, min_periods=30).mean()
    df["rolling_std"] = df.cycle_time_s.rolling(window_units, min_periods=30).std()
    return df


def detect_drift(telemetry, station_sequence, sigma_threshold=3.0, window_units=150):
    """
    Control-chart drift detection: flag a unit as 'drifting' when its
    cycle time exceeds rolling_mean + sigma_threshold * rolling_std.

    Returns a per-station-per-unit dataframe with a `drift_flag` column
    and the current z-score, so alerts can be ranked by severity.
    """
    results = []
    for station_id in telemetry.station_id.unique():
        df = compute_rolling_baseline(telemetry, station_id, window_units)
        df["z_score"] = (df.cycle_time_s - df.rolling_mean) / df.rolling_std.replace(0, np.nan)
        df["drift_flag"] = df.z_score > sigma_threshold
        results.append(df)
    out = pd.concat(results, ignore_index=True)
    return out.merge(station_sequence, on="station_id", how="left")


def project_ripple(drift_df, stations_df, units_per_shift=120, avg_units_in_flight=6):
    """
    For each currently-drifting station, estimate how many downstream
    vehicles will queue behind it before the shift ends, and roughly
    how many minutes that represents -- the 'projected ripple window'
    referenced in the pitch deck.

    This is a deliberately simple queueing approximation: extra seconds
    per unit at the bottleneck station, multiplied by the number of
    units still to pass through it this shift, converted to a vehicle
    count using the line's average units-in-flight.
    """
    active = drift_df[drift_df.drift_flag].copy()
    if active.empty:
        return pd.DataFrame(columns=[
            "station_id", "unit_id", "excess_cycle_time_s",
            "projected_delay_minutes", "projected_vehicles_queued"
        ])

    active["excess_cycle_time_s"] = (active.cycle_time_s - active.rolling_mean).clip(lower=0)
    # remaining units in the current shift at time of detection
    active["unit_in_shift"] = ((active.unit_id - 1) % units_per_shift) + 1
    active["remaining_units_in_shift"] = units_per_shift - active.unit_in_shift

    active["projected_delay_minutes"] = (
        active.excess_cycle_time_s * active.remaining_units_in_shift / 60.0
    )
    active["projected_vehicles_queued"] = (
        (active.excess_cycle_time_s * avg_units_in_flight) / active.rolling_mean
    ).round(1)

    return active[[
        "station_id", "unit_id", "shift", "excess_cycle_time_s",
        "projected_delay_minutes", "projected_vehicles_queued"
    ]].sort_values("projected_delay_minutes", ascending=False)


def summarize_alerts(drift_df, ripple_df, top_n=10):
    """Produce a compact 'floor supervisor' style alert list: the most
    recent, most severe drift per station."""
    latest_drift = (
        drift_df[drift_df.drift_flag]
        .sort_values("unit_id")
        .groupby("station_id")
        .tail(1)
        [["station_id", "zone", "unit_id", "shift", "cycle_time_s", "rolling_mean", "z_score"]]
    )
    latest_drift = latest_drift.merge(
        ripple_df.groupby("station_id").agg(
            projected_delay_minutes=("projected_delay_minutes", "max"),
            projected_vehicles_queued=("projected_vehicles_queued", "max"),
        ).reset_index(),
        on="station_id", how="left"
    )
    return latest_drift.sort_values("z_score", ascending=False).head(top_n)
