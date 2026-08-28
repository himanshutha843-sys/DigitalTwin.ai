"""
simulate_line.py
-----------------
Generates synthetic vehicle-assembly-line data for the DigitalTwin.ai prototype.

Design goals (mirrors the pitch deck's "Reference Parameters"):
  - 30-50 stations across body construction, paint, and final assembly
  - Uneven sensor coverage: most stations well-instrumented, a meaningful
    minority reliant on manual checks only (sparse/no telemetry)
  - Realistic-but-simplified process signals: cycle time, torque, weld
    current, queue length
  - Injected drift events (bottlenecks) and correlated defect outcomes,
    so the detection/prediction layers have real signal to find

This is illustrative/simulated data only, not real enterprise data,
as encouraged by the Round 2 brief.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
STATIONS_PER_ZONE = {"body": 15, "paint": 10, "final_assembly": 15}
SHIFTS_SIMULATED = 6          # multi-shift baseline window
UNITS_PER_SHIFT = 120         # vehicles produced per shift
MANUAL_STATION_FRACTION = 0.25  # fraction of stations with no/low sensor coverage


def build_station_list(seed=RNG_SEED):
    """Create the station roster with zone, baseline cycle time, and
    sensor-coverage tier ('full', 'partial', 'manual')."""
    rng = np.random.default_rng(seed)
    stations = []
    station_id = 0
    for zone, count in STATIONS_PER_ZONE.items():
        for i in range(count):
            station_id += 1
            baseline_cycle_time = rng.normal(60, 8)  # seconds
            coverage_roll = rng.random()
            if coverage_roll < MANUAL_STATION_FRACTION:
                coverage = "manual"       # no live sensors, checklist only
            elif coverage_roll < MANUAL_STATION_FRACTION + 0.15:
                coverage = "partial"      # some signals, e.g. cycle time only
            else:
                coverage = "full"         # cycle time + torque + weld current
            stations.append({
                "station_id": f"{zone[:2].upper()}-{station_id:03d}",
                "zone": zone,
                "sequence": station_id,
                "baseline_cycle_time_s": round(baseline_cycle_time, 2),
                "sensor_coverage": coverage,
            })
    return pd.DataFrame(stations)


def simulate_telemetry(stations_df, seed=RNG_SEED, drift_events=None):
    """
    Simulate per-unit, per-station telemetry across shifts.

    drift_events: optional list of dicts describing injected bottlenecks:
        {"station_id": "BO-005", "start_unit": 300, "duration_units": 80,
         "severity_s": 25}
    Each drift event gradually raises that station's cycle time for the
    given unit window, which is what the detection layer should catch.
    """
    rng = np.random.default_rng(seed + 1)
    total_units = SHIFTS_SIMULATED * UNITS_PER_SHIFT
    drift_events = drift_events or []

    records = []
    for _, station in stations_df.iterrows():
        active_drifts = [d for d in drift_events if d["station_id"] == station.station_id]

        for unit in range(1, total_units + 1):
            shift = (unit - 1) // UNITS_PER_SHIFT + 1
            cycle_time = rng.normal(station.baseline_cycle_time_s, 2.5)

            # apply any active drift window for this station
            drift_offset = 0.0
            for d in active_drifts:
                if d["start_unit"] <= unit < d["start_unit"] + d["duration_units"]:
                    progress = (unit - d["start_unit"]) / d["duration_units"]
                    drift_offset = d["severity_s"] * min(1.0, progress * 2)  # ramps up then holds
            cycle_time += drift_offset

            # process signals only exist where sensor coverage allows
            torque = weld_current = np.nan
            if station.sensor_coverage in ("full",):
                torque = rng.normal(45, 3) + (drift_offset * 0.4)
                weld_current = rng.normal(210, 12) + (drift_offset * 1.1)
            elif station.sensor_coverage == "partial":
                torque = rng.normal(45, 3) + (drift_offset * 0.4)
                # weld_current stays NaN -- partial coverage

            # Defect probability: low baseline noise per station, rising
            # sharply with drift severity and torque/weld-current stress.
            # Kept deliberately low at baseline so that, aggregated across
            # ~40 stations per vehicle, genuine drift-driven signal isn't
            # drowned out by station-level noise -- otherwise a classifier
            # can look good on the training split while carrying no real
            # signal (test AUC near 0.5), which would misrepresent what
            # the model can actually do.
            defect_logit = -7.0 + 0.28 * drift_offset
            if not np.isnan(torque):
                defect_logit += 0.09 * abs(torque - 45)
            if not np.isnan(weld_current):
                defect_logit += 0.015 * abs(weld_current - 210)
            defect_prob = 1 / (1 + np.exp(-defect_logit))
            defect = rng.random() < defect_prob

            records.append({
                "unit_id": unit,
                "shift": shift,
                "station_id": station.station_id,
                "zone": station.zone,
                "sensor_coverage": station.sensor_coverage,
                "cycle_time_s": round(cycle_time, 2),
                "torque_nm": None if np.isnan(torque) else round(torque, 2),
                "weld_current_a": None if np.isnan(weld_current) else round(weld_current, 2),
                "defect_flag": bool(defect),
            })

    return pd.DataFrame(records)


def default_drift_scenario():
    """A representative set of injected bottlenecks used for the demo."""
    return [
        {"station_id": "BO-005", "start_unit": 250, "duration_units": 90, "severity_s": 22},
        {"station_id": "FI-032", "start_unit": 500, "duration_units": 60, "severity_s": 30},
        {"station_id": "PA-020", "start_unit": 610, "duration_units": 100, "severity_s": 15},
    ]


if __name__ == "__main__":
    stations = build_station_list()
    telemetry = simulate_telemetry(stations, drift_events=default_drift_scenario())
    stations.to_csv("stations.csv", index=False)
    telemetry.to_csv("telemetry.csv", index=False)
    print(f"Simulated {len(stations)} stations, {len(telemetry)} telemetry rows.")
    print(stations.sensor_coverage.value_counts())
