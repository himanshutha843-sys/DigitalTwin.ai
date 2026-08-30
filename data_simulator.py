"""
data_simulator.py
-----------------
Synthetic data generator for DigitalTwin.ai.

The generated data represents a mixed-model vehicle assembly line with 45
stations across body, paint, and final assembly. It intentionally includes
uneven sensor coverage, null telemetry for legacy/manual stations, delayed
quality detection, and a multi-causal defect pattern:

    higher temperature at Station 12 + high vibration at Station 15
    -> defect detected later at Station 45.

Outputs:
    data/station_master.csv
    data/synthetic_station_events.csv
    data/vehicle_quality_summary.csv

Example:
    python data_simulator.py
    python data_simulator.py --days 7 --units-per-shift 80 --seed 99
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SEED = 42
DEFAULT_STATIONS = 45
DEFAULT_DAYS = 6
DEFAULT_SHIFTS_PER_DAY = 2
DEFAULT_UNITS_PER_SHIFT = 75
DEFAULT_OUTPUT_DIR = Path("data")



@dataclass(frozen=True)
class SiteProfile:
    name: str
    station_count: int
    coverage_probs_body: list[float]
    coverage_probs_paint: list[float]
    coverage_probs_final: list[float]
    base_defect_logit: float
    seed: int

SITE_PROFILES = {
    "plant_a_high_instrumentation": SiteProfile(
        name="plant_a_high_instrumentation",
        station_count=45,
        coverage_probs_body=[0.56, 0.22, 0.15, 0.07],
        coverage_probs_paint=[0.48, 0.24, 0.20, 0.08],
        coverage_probs_final=[0.50, 0.25, 0.17, 0.08],
        base_defect_logit=-4.15,
        seed=42,
    ),
    "plant_b_mixed": SiteProfile(
        name="plant_b_mixed",
        station_count=45,
        coverage_probs_body=[0.30, 0.30, 0.25, 0.15],
        coverage_probs_paint=[0.30, 0.30, 0.30, 0.10],
        coverage_probs_final=[0.30, 0.30, 0.30, 0.10],
        base_defect_logit=-3.80,
        seed=101,
    ),
    "plant_c_legacy_heavy": SiteProfile(
        name="plant_c_legacy_heavy",
        station_count=45, # Keep at 45 to not break ST-45 quality gate logic without rewriting more
        coverage_probs_body=[0.10, 0.25, 0.45, 0.20],
        coverage_probs_paint=[0.10, 0.25, 0.45, 0.20],
        coverage_probs_final=[0.10, 0.25, 0.45, 0.20],
        base_defect_logit=-3.40,
        seed=202,
    )
}

@dataclass(frozen=True)
class ShiftWindow:
    name: str
    start_hour: int
    start_minute: int


SHIFT_WINDOWS = (
    ShiftWindow("A", 6, 0),
    ShiftWindow("B", 14, 30),
    ShiftWindow("C", 23, 0),
)


MODEL_COMPLEXITY = {
    "Compact": 0.94,
    "Sedan": 1.00,
    "SUV": 1.08,
    "Truck": 1.14,
}


def station_area(station_number: int, total_stations: int) -> str:
    """Map sequence position to the major assembly area."""
    body_end = round(total_stations * 0.36)
    paint_end = round(total_stations * 0.62)
    if station_number <= body_end:
        return "Body"
    if station_number <= paint_end:
        return "Paint"
    return "Final Assembly"


def station_operation(area: str, station_number: int) -> str:
    """Return a readable operation label for station metadata."""
    operations = {
        "Body": [
            "Underbody weld",
            "Side-frame marriage",
            "Roof laser weld",
            "Door fit check",
            "Dimensional clamp",
            "Seam sealing prep",
        ],
        "Paint": [
            "E-coat inspection",
            "Primer booth",
            "Oven cure",
            "Basecoat booth",
            "Clearcoat booth",
            "Paint quality gate",
        ],
        "Final Assembly": [
            "Powertrain marriage",
            "Battery connect",
            "Interior fit",
            "Wheel torque",
            "ADAS calibration",
            "End-of-line quality",
        ],
    }
    return operations[area][(station_number - 1) % len(operations[area])]


def build_station_master(
    station_count: int = DEFAULT_STATIONS,
    seed: int = DEFAULT_SEED,
    site_profile: SiteProfile | None = None,
) -> pd.DataFrame:
    """Create station metadata with deliberately uneven sensor coverage."""
    if site_profile is not None:
        station_count = site_profile.station_count
        seed = site_profile.seed
        
    if station_count < 30 or station_count > 50:
        raise ValueError("station_count must be between 30 and 50.")
    if station_count < 45:
        raise ValueError("station_count must be at least 45 for Station 45 quality detection.")

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []

    for station_number in range(1, station_count + 1):
        area = station_area(station_number, station_count)

        if area == "Body":
            cycle_base = rng.normal(64, 5)
            coverage_probs = site_profile.coverage_probs_body if site_profile else [0.56, 0.22, 0.15, 0.07]
        elif area == "Paint":
            cycle_base = rng.normal(88, 7)
            coverage_probs = site_profile.coverage_probs_paint if site_profile else [0.48, 0.24, 0.20, 0.08]
        else:
            cycle_base = rng.normal(72, 6)
            coverage_probs = site_profile.coverage_probs_final if site_profile else [0.50, 0.25, 0.17, 0.08]

        coverage = rng.choice(
            ["modern", "partial", "legacy", "manual_check"],
            p=coverage_probs,
        )

        # Force the key causal stations and quality gate to have the signals
        # needed by the Round 2 scenario.
        if station_number in (12, 15):
            coverage = "modern"
        if station_number == 45:
            coverage = "modern"

        records.append(
            {
                "Station_ID": f"ST-{station_number:02d}",
                "Station_Number": station_number,
                "Process_Area": area,
                "Operation": station_operation(area, station_number),
                "Sensor_Coverage": coverage,
                "Baseline_Cycle_Time": round(float(cycle_base), 2),
                "Is_Quality_Gate": station_number == 45,
                "PLC_Integration_Mode": "read_only",
                "Maintenance_Window_Required_For_Changes": True,
            }
        )

    return pd.DataFrame(records)


def shift_start(base_date: datetime, day_index: int, shift_index: int) -> datetime:
    """Return shift start timestamp for the requested day and shift."""
    shift = SHIFT_WINDOWS[shift_index]
    return (
        base_date
        + timedelta(days=day_index, hours=shift.start_hour, minutes=shift.start_minute)
    )


def triangular_event_strength(
    unit_global_index: int,
    start: int,
    peak: int,
    end: int,
    amplitude: float,
) -> float:
    """Smooth degradation event used to create realistic drift windows."""
    if unit_global_index < start or unit_global_index > end:
        return 0.0
    if unit_global_index <= peak:
        return amplitude * (unit_global_index - start) / max(1, peak - start)
    return amplitude * (end - unit_global_index) / max(1, end - peak)


def generate_vehicle_contexts(
    days: int,
    shifts_per_day: int,
    units_per_shift: int,
    seed: int,
    site_profile: SiteProfile | None = None,
) -> pd.DataFrame:
    if site_profile is not None:
        seed = site_profile.seed
    """Build one row per vehicle with model mix and latent quality outcome."""
    rng = np.random.default_rng(seed + 101)
    total_vehicles = days * shifts_per_day * units_per_shift
    base_date = datetime(2026, 8, 30)

    operators_by_shift = {
        "A": [f"OP-A{i:02d}" for i in range(1, 9)],
        "B": [f"OP-B{i:02d}" for i in range(1, 9)],
        "C": [f"OP-C{i:02d}" for i in range(1, 7)],
    }

    records: list[dict[str, object]] = []

    for global_index in range(total_vehicles):
        day_index = global_index // (shifts_per_day * units_per_shift)
        within_day = global_index % (shifts_per_day * units_per_shift)
        shift_index = within_day // units_per_shift
        unit_in_shift = within_day % units_per_shift
        shift = SHIFT_WINDOWS[shift_index].name
        model = rng.choice(
            list(MODEL_COMPLEXITY.keys()),
            p=[0.30, 0.30, 0.28, 0.12],
        )

        # Main scenario: a slow thermal drift at Station 12 overlaps with
        # vibration growth at Station 15. The quality symptom is delayed until
        # Station 45, which mirrors real attribution pain.
        heat_drift = triangular_event_strength(
            global_index,
            start=int(total_vehicles * 0.30),
            peak=int(total_vehicles * 0.48),
            end=int(total_vehicles * 0.66),
            amplitude=5.8,
        )
        vibration_drift = triangular_event_strength(
            global_index,
            start=int(total_vehicles * 0.36),
            peak=int(total_vehicles * 0.54),
            end=int(total_vehicles * 0.72),
            amplitude=2.1,
        )

        ambient_shift_heat = 0.8 if shift == "B" else 0.0
        s12_temperature = 39.5 + ambient_shift_heat + heat_drift + rng.normal(0, 0.75)
        s15_vibration = 2.1 + vibration_drift + 0.13 * heat_drift + rng.normal(0, 0.20)

        heat_excess = max(0.0, s12_temperature - 42.0)
        vibration_excess = max(0.0, s15_vibration - 3.1)
        interaction = heat_excess * vibration_excess

        random_process_noise = rng.normal(0, 0.18)
        model_penalty = {"Compact": -0.08, "Sedan": 0.0, "SUV": 0.14, "Truck": 0.22}[model]

        # A small baseline quality risk plus a strong interaction term. This
        # creates defects the model can learn without making failures too common.
        base_logit = site_profile.base_defect_logit if site_profile else -4.15
        defect_logit = base_logit + 0.82 * interaction + model_penalty + random_process_noise
        eol_fail_probability = 1.0 / (1.0 + np.exp(-defect_logit))
        failed = rng.random() < eol_fail_probability

        other_cause = rng.random() < 0.009
        if other_cause:
            failed = True

        if failed and interaction > 1.4:
            root_cause = "S12_TEMP_X_S15_VIBRATION"
        elif failed and other_cause:
            root_cause = rng.choice(
                [
                    "PAINT_ENVIRONMENT_VARIATION",
                    "FINAL_TORQUE_VARIATION",
                    "MANUAL_FIT_CHECK_VARIATION",
                ]
            )
        elif failed:
            root_cause = "LOW_CONFIDENCE_MULTI_CAUSAL"
        else:
            root_cause = "NONE"

        records.append(
            {
                "Vehicle_ID": f"VH-{global_index + 1:05d}",
                "Vehicle_Global_Index": global_index,
                "Production_Day": day_index + 1,
                "Shift": shift,
                "Unit_In_Shift": unit_in_shift + 1,
                "Vehicle_Model": model,
                "Model_Complexity_Factor": MODEL_COMPLEXITY[model],
                "Primary_Operator_ID": rng.choice(operators_by_shift[shift]),
                "Shift_Start_Timestamp": shift_start(base_date, day_index, shift_index),
                "Station_12_Temperature_Signal": round(float(s12_temperature), 3),
                "Station_15_Vibration_Signal": round(float(s15_vibration), 3),
                "Latent_Defect_Probability": round(float(eol_fail_probability), 5),
                "End_of_Line_Quality": "Fail" if failed else "Pass",
                "Root_Cause_Label": root_cause,
                "Delayed_Defect_Detection_Station": "ST-45" if failed else "",
            }
        )

    return pd.DataFrame(records)


def telemetry_confidence(sensor_coverage: str) -> float:
    """Base confidence score by station instrumentation level."""
    return {
        "modern": 0.94,
        "partial": 0.68,
        "legacy": 0.44,
        "manual_check": 0.30,
    }[sensor_coverage]


def apply_sensor_gaps(
    rng: np.random.Generator,
    sensor_coverage: str,
    torque: float | None,
    temperature: float | None,
    vibration: float | None,
) -> tuple[float | None, float | None, float | None, str]:
    """Null out telemetry according to the station's coverage tier."""
    gap_reason = "none"

    if sensor_coverage == "manual_check":
        return None, None, None, "manual_station_no_live_telemetry"

    if sensor_coverage == "legacy":
        # Legacy stations often provide cycle time and status, but not rich
        # process telemetry. Rare temporary sensors are represented as sparse
        # temperature readings.
        sparse_temperature = temperature if rng.random() < 0.12 else None
        return None, sparse_temperature, None, "legacy_station_sparse_telemetry"

    if sensor_coverage == "partial":
        keep_torque = rng.random() < 0.70
        keep_temperature = rng.random() < 0.55
        keep_vibration = rng.random() < 0.45
        if not (keep_torque and keep_temperature and keep_vibration):
            gap_reason = "partial_sensor_package"
        return (
            torque if keep_torque else None,
            temperature if keep_temperature else None,
            vibration if keep_vibration else None,
            gap_reason,
        )

    # Modern stations still have occasional dropouts.
    dropout = rng.random(3) < 0.025
    if dropout.any():
        gap_reason = "modern_sensor_dropout"
    return (
        None if dropout[0] else torque,
        None if dropout[1] else temperature,
        None if dropout[2] else vibration,
        gap_reason,
    )


def simulate_station_events(
    station_master: pd.DataFrame,
    vehicles: pd.DataFrame,
    units_per_shift: int,
    seed: int,
    site_profile: SiteProfile | None = None,
) -> pd.DataFrame:
    if site_profile is not None:
        seed = site_profile.seed
    """Generate long-form per-vehicle, per-station production events."""
    rng = np.random.default_rng(seed + 202)
    records: list[dict[str, object]] = []

    for vehicle in vehicles.itertuples(index=False):
        launch_time = vehicle.Shift_Start_Timestamp + timedelta(
            seconds=int((vehicle.Unit_In_Shift - 1) * 72)
        )
        model_factor = float(vehicle.Model_Complexity_Factor)
        s12_temperature_signal = float(vehicle.Station_12_Temperature_Signal)
        s15_vibration_signal = float(vehicle.Station_15_Vibration_Signal)
        failed = vehicle.End_of_Line_Quality == "Fail"

        cumulative_seconds = 0.0

        for station in station_master.itertuples(index=False):
            station_number = int(station.Station_Number)
            station_id = str(station.Station_ID)
            area = str(station.Process_Area)
            coverage = str(station.Sensor_Coverage)
            baseline_cycle = float(station.Baseline_Cycle_Time)

            model_cycle_adjustment = (model_factor - 1.0) * 8.0
            cycle_noise = rng.normal(0, 2.8)
            cycle_time = baseline_cycle + model_cycle_adjustment + cycle_noise

            # Process values before sensor coverage gaps are applied.
            if area == "Body":
                torque = rng.normal(46.0, 3.2)
                temperature = rng.normal(38.5, 1.1)
                vibration = rng.normal(2.0, 0.28)
            elif area == "Paint":
                torque = rng.normal(20.0, 1.8)
                temperature = rng.normal(46.0, 1.8)
                vibration = rng.normal(1.5, 0.22)
            else:
                torque = rng.normal(39.0, 4.0)
                temperature = rng.normal(34.0, 1.2)
                vibration = rng.normal(1.8, 0.25)

            if station_number == 12:
                temperature = s12_temperature_signal
                torque += max(0.0, temperature - 42.0) * 0.28
                cycle_time += max(0.0, temperature - 42.0) * 0.75

            if station_number == 15:
                vibration = s15_vibration_signal
                cycle_time += max(0.0, vibration - 3.1) * 5.5
                torque += max(0.0, vibration - 3.1) * 0.85

            # Ripple effect: downstream stations carry a small cycle-time burden
            # after the upstream thermal/vibration interaction.
            if 16 <= station_number <= 32:
                interaction = max(0.0, s12_temperature_signal - 42.0) * max(
                    0.0, s15_vibration_signal - 3.1
                )
                cycle_time += min(7.0, interaction * 0.75)

            if station_id == "ST-45" and failed:
                cycle_time += rng.normal(38.0, 8.0)

            cycle_time = max(30.0, cycle_time)
            cumulative_seconds += cycle_time
            timestamp = launch_time + timedelta(seconds=int(cumulative_seconds))

            torque, temperature, vibration, gap_reason = apply_sensor_gaps(
                rng,
                coverage,
                round(float(torque), 3),
                round(float(temperature), 3),
                round(float(vibration), 3),
            )

            confidence = telemetry_confidence(coverage)
            if gap_reason != "none":
                confidence -= 0.08
            confidence = max(0.10, round(confidence, 2))

            telemetry_payload = {
                "torque_nm": torque,
                "temperature_c": temperature,
                "vibration_mm_s": vibration,
            }

            defect_detected_here = station_id == "ST-45" and failed
            delayed_pattern_vehicle = vehicle.Root_Cause_Label == "S12_TEMP_X_S15_VIBRATION"

            records.append(
                {
                    "Vehicle_ID": vehicle.Vehicle_ID,
                    "Station_ID": station_id,
                    "Station_Number": station_number,
                    "Process_Area": area,
                    "Timestamp": timestamp.isoformat(sep=" "),
                    "Shift": vehicle.Shift,
                    "Vehicle_Model": vehicle.Vehicle_Model,
                    "Operator_ID": vehicle.Primary_Operator_ID,
                    "Sensor_Coverage": coverage,
                    "Sensor_Telemetry": json.dumps(telemetry_payload, sort_keys=True),
                    "Torque_Nm": torque,
                    "Temperature_C": temperature,
                    "Vibration_MM_S": vibration,
                    "Cycle_Time": round(float(cycle_time), 3),
                    "Telemetry_Confidence": confidence,
                    "Sensor_Gap_Reason": gap_reason,
                    "End_of_Line_Quality": vehicle.End_of_Line_Quality,
                    "Defect_Detected": defect_detected_here,
                    "Defect_Detected_Station_ID": "ST-45" if defect_detected_here else "",
                    "Delayed_Defect_Flag": defect_detected_here and delayed_pattern_vehicle,
                    "Quality_Detection_Lag_Stations": 33 if defect_detected_here and delayed_pattern_vehicle else 0,
                    "Root_Cause_Label": vehicle.Root_Cause_Label,
                    "Read_Only_OT_Integration": True,
                    "Maintenance_Window_Flag": False,
                }
            )

    return pd.DataFrame(records)


def write_outputs(
    station_master: pd.DataFrame,
    station_events: pd.DataFrame,
    vehicle_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Persist generated data to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    station_master.to_csv(output_dir / "station_master.csv", index=False)
    station_events.to_csv(output_dir / "synthetic_station_events.csv", index=False)
    vehicle_summary.to_csv(output_dir / "vehicle_quality_summary.csv", index=False)


def summarize(station_master: pd.DataFrame, station_events: pd.DataFrame, vehicles: pd.DataFrame) -> str:
    """Return a concise generation summary for console output."""
    quality_counts = vehicles["End_of_Line_Quality"].value_counts().to_dict()
    delayed_count = int(
        station_events.loc[
            station_events["Delayed_Defect_Flag"] == True,  # noqa: E712
            "Vehicle_ID",
        ].nunique()
    )
    null_rates = (
        station_events[["Torque_Nm", "Temperature_C", "Vibration_MM_S"]]
        .isna()
        .mean()
        .mul(100)
        .round(1)
        .to_dict()
    )
    coverage_mix = {
        str(key): int(value)
        for key, value in station_master["Sensor_Coverage"].value_counts().to_dict().items()
    }

    return "\n".join(
        [
            "Synthetic DigitalTwin.ai dataset generated.",
            f"Stations: {len(station_master)}",
            f"Vehicles: {len(vehicles)}",
            f"Station-event rows: {len(station_events)}",
            f"End-of-line quality: {quality_counts}",
            f"Delayed ST-12/ST-15 -> ST-45 defects: {delayed_count}",
            f"Sensor coverage mix: {coverage_mix}",
            f"Telemetry null rates (%): {null_rates}",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DigitalTwin.ai synthetic line data.")
    parser.add_argument("--stations", type=int, default=DEFAULT_STATIONS, help="Number of stations, 45 to 50 recommended.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Number of production days to simulate.")
    parser.add_argument("--shifts-per-day", type=int, default=DEFAULT_SHIFTS_PER_DAY, choices=[1, 2, 3])
    parser.add_argument("--units-per-shift", type=int, default=DEFAULT_UNITS_PER_SHIFT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    station_master = build_station_master(args.stations, args.seed)
    vehicles = generate_vehicle_contexts(
        days=args.days,
        shifts_per_day=args.shifts_per_day,
        units_per_shift=args.units_per_shift,
        seed=args.seed,
    )
    station_events = simulate_station_events(
        station_master=station_master,
        vehicles=vehicles,
        units_per_shift=args.units_per_shift,
        seed=args.seed,
    )
    write_outputs(station_master, station_events, vehicles, args.output_dir)
    print(summarize(station_master, station_events, vehicles))
    print(f"Files written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
