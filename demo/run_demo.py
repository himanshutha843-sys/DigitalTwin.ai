"""
run_demo.py
------------
End-to-end demo of the DigitalTwin.ai prototype's core predictive
mechanism, on simulated data:

  1. Simulate a 40-station line across 6 shifts with injected drift events
  2. Layer 1 -- build a live line snapshot (station map)
  3. Layer 2 -- detect bottleneck drift + project downstream ripple
  4. Confidence layer -- infer state at manual/no-sensor stations
  5. Layer 3 -- train a defect-risk classifier and rank vehicles for inspection
  6. Print the three stakeholder views (floor supervisor / plant manager /
     leadership) and save a summary chart

Run with:  python demo/run_demo.py
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "data"))

from simulate_line import build_station_list, simulate_telemetry, default_drift_scenario
from live_line_model import build_live_snapshot
from bottleneck_detection import detect_drift, project_ripple, summarize_alerts
from confidence_inference import infer_manual_station_state, combined_station_view
from defect_prediction import build_vehicle_features, train_defect_model, score_vehicles, inspection_queue

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 70)
    print("STEP 1 -- Simulating the line")
    print("=" * 70)
    stations = build_station_list()
    telemetry = simulate_telemetry(stations, drift_events=default_drift_scenario())
    print(f"{len(stations)} stations | {len(telemetry)} telemetry rows | "
          f"coverage breakdown:\n{stations.sensor_coverage.value_counts().to_string()}\n")

    print("=" * 70)
    print("LAYER 1 -- Live line snapshot (floor supervisor view)")
    print("=" * 70)
    snapshot = build_live_snapshot(telemetry, stations)
    print(snapshot[["station_id", "zone", "sensor_coverage", "status",
                     "latest_cycle_time_s", "baseline_cycle_time_s"]].to_string(index=False))
    print()

    print("=" * 70)
    print("LAYER 2 -- Bottleneck detection + ripple projection")
    print("=" * 70)
    station_sequence = stations[["station_id", "sequence"]]
    drift_df = detect_drift(telemetry, station_sequence)
    ripple_df = project_ripple(drift_df, stations)
    alerts = summarize_alerts(drift_df, ripple_df)
    print("Active/most severe drift alerts:")
    print(alerts.to_string(index=False) if not alerts.empty else "No drift currently above threshold.")
    print()

    print("=" * 70)
    print("CONFIDENCE LAYER -- inferring manual/no-sensor stations")
    print("=" * 70)
    manual_inferred = infer_manual_station_state(drift_df, stations)
    combined_view = combined_station_view(drift_df, stations, manual_inferred)
    print(f"Manual stations with inferred state: {manual_inferred.station_id.nunique()}")
    if not manual_inferred.empty:
        print(manual_inferred.groupby("station_id").inferred_drift_flag.any()
              .reset_index().rename(columns={"inferred_drift_flag": "any_inferred_drift"})
              .to_string(index=False))
    print()

    print("=" * 70)
    print("LAYER 3 -- Defect risk prediction + inspection queue")
    print("=" * 70)
    features = build_vehicle_features(telemetry)
    model, feature_cols, metrics = train_defect_model(features)
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "ROC-AUC unavailable")
    print("Feature importance:", metrics["feature_importance"])
    scored = score_vehicles(model, features, feature_cols)
    queue = inspection_queue(scored, capacity_per_shift=15)
    print("\nTop of ranked inspection queue (plant manager / floor view):")
    print(queue.to_string(index=False))
    print()

    print("=" * 70)
    print("LEADERSHIP VIEW -- rollout impact summary")
    print("=" * 70)
    total_defects = int(features.defect_flag.sum())
    caught_in_queue = int(queue.defect_flag.sum())
    print(f"Simulated defects across run: {total_defects}")
    print(f"Defects captured within top-{len(queue)} targeted inspection queue: {caught_in_queue} "
          f"({caught_in_queue / max(total_defects, 1):.0%} of all defects, "
          f"inspecting only {len(queue)}/{features.unit_id.nunique()} vehicles)")
    print()

    _save_summary_chart(drift_df, ripple_df, scored, queue)
    combined_view.to_csv(os.path.join(OUTPUT_DIR, "combined_station_view.csv"), index=False)
    scored.to_csv(os.path.join(OUTPUT_DIR, "defect_risk_scores.csv"), index=False)
    print(f"Saved outputs to {OUTPUT_DIR}/")


def _save_summary_chart(drift_df, ripple_df, scored, queue):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1. cycle time drift for the most-flagged station
    if not drift_df[drift_df.drift_flag].empty:
        worst_station = drift_df[drift_df.drift_flag].station_id.value_counts().idxmax()
        sdf = drift_df[drift_df.station_id == worst_station].sort_values("unit_id")
        axes[0].plot(sdf.unit_id, sdf.cycle_time_s, label="cycle time", color="#7c3aed")
        axes[0].plot(sdf.unit_id, sdf.rolling_mean, label="rolling baseline", color="#94a3b8", linestyle="--")
        axes[0].fill_between(sdf.unit_id, sdf.rolling_mean, sdf.cycle_time_s,
                              where=sdf.drift_flag, color="#f59e0b", alpha=0.3, label="drift flagged")
        axes[0].set_title(f"Bottleneck detection: {worst_station}")
        axes[0].set_xlabel("unit_id")
        axes[0].set_ylabel("cycle time (s)")
        axes[0].legend(fontsize=8)

    # 2. projected ripple
    if not ripple_df.empty:
        top_ripple = ripple_df.groupby("station_id").projected_delay_minutes.max().sort_values(ascending=False)
        axes[1].barh(top_ripple.index, top_ripple.values, color="#a855f7")
        axes[1].set_title("Projected downstream delay by station")
        axes[1].set_xlabel("minutes")

    # 3. defect risk distribution + queue cutoff
    axes[2].hist(scored.defect_risk_score, bins=30, color="#c4b5fd")
    if not queue.empty:
        cutoff = queue.defect_risk_score.min()
        axes[2].axvline(cutoff, color="#dc2626", linestyle="--", label="inspection cutoff")
        axes[2].legend(fontsize=8)
    axes[2].set_title("Defect risk score distribution")
    axes[2].set_xlabel("risk score")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "control_room_summary.png")
    plt.savefig(out_path, dpi=140)
    print(f"Saved chart: {out_path}")


if __name__ == "__main__":
    main()
