"""
confidence_inference.py
------------------------
Handles the "confidence-aware fallback" from the pitch deck: for stations
with partial or manual (no live sensor) coverage, infer their likely state
from neighboring, well-instrumented stations' timing rather than fabricating
precision the twin doesn't have -- and explicitly tag the result's confidence
level so plant teams know how much weight to give it.
"""

import numpy as np
import pandas as pd

CONFIDENCE_BY_COVERAGE = {
    "full": 1.0,
    "partial": 0.6,
    "manual": None,  # computed via neighbor inference below
}


def infer_manual_station_state(drift_df, stations_df, neighbor_window=2):
    """
    For 'manual' (no-sensor) stations, approximate their drift status
    using the average z-score of the nearest instrumented neighbors
    (by line sequence, within +/- neighbor_window stations), discounted
    by distance and tagged as low-confidence.
    """
    seq_lookup = stations_df.set_index("station_id").sequence.to_dict()
    coverage_lookup = stations_df.set_index("station_id").sensor_coverage.to_dict()

    # per-unit z-scores for instrumented stations only
    instrumented = drift_df[drift_df.sensor_coverage.isin(["full", "partial"])]
    z_by_unit_station = instrumented.pivot_table(
        index="unit_id", columns="station_id", values="z_score", aggfunc="mean"
    )

    manual_stations = stations_df[stations_df.sensor_coverage == "manual"]
    inferred_rows = []

    for _, station in manual_stations.iterrows():
        seq = station.sequence
        neighbor_ids = [
            sid for sid, s in seq_lookup.items()
            if 0 < abs(s - seq) <= neighbor_window and coverage_lookup[sid] != "manual"
        ]
        neighbor_ids = [sid for sid in neighbor_ids if sid in z_by_unit_station.columns]
        if not neighbor_ids:
            continue

        inferred_z = z_by_unit_station[neighbor_ids].mean(axis=1)
        for unit_id, z in inferred_z.dropna().items():
            inferred_rows.append({
                "station_id": station.station_id,
                "zone": station.zone,
                "unit_id": unit_id,
                "inferred_z_score": z,
                "inferred_drift_flag": z > 3.0,
                "confidence": "low",
                "confidence_score": 0.3,  # inferred, never treated as measured
                "basis": f"neighbor stations: {', '.join(neighbor_ids)}",
            })

    return pd.DataFrame(inferred_rows)


def annotate_confidence(drift_df, stations_df):
    """Attach a confidence tag/score to every row so downstream views
    (dashboard, inspection queue) can display it alongside every alert
    or score, rather than presenting inferred and measured data alike."""
    coverage_lookup = stations_df.set_index("station_id").sensor_coverage.to_dict()
    df = drift_df.copy()
    df["confidence"] = df.station_id.map(coverage_lookup).map(
        {"full": "high", "partial": "medium", "manual": "low"}
    )
    df["confidence_score"] = df.station_id.map(coverage_lookup).map(CONFIDENCE_BY_COVERAGE)
    return df


def combined_station_view(drift_df, stations_df, manual_inferred_df):
    """Merge measured (full/partial) and inferred (manual) station
    states into one line-wide view, each row tagged with its confidence
    level -- this is what feeds the 'Confidence Inference' box in the
    twin engine and the control-room dashboard."""
    measured = annotate_confidence(drift_df, stations_df)[
        ["station_id", "zone", "unit_id", "cycle_time_s", "z_score",
         "drift_flag", "confidence", "confidence_score"]
    ].rename(columns={"drift_flag": "is_drifting"})

    if not manual_inferred_df.empty:
        inferred = manual_inferred_df.rename(columns={
            "inferred_z_score": "z_score",
            "inferred_drift_flag": "is_drifting",
        })
        inferred["cycle_time_s"] = np.nan
        inferred = inferred[
            ["station_id", "zone", "unit_id", "cycle_time_s", "z_score",
             "is_drifting", "confidence", "confidence_score"]
        ]
        return pd.concat([measured, inferred], ignore_index=True)

    return measured
