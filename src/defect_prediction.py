"""
defect_prediction.py
----------------------
Layer 3 of the digital twin: a lightweight classifier that correlates
upstream process parameters (torque, weld current, cycle-time variance)
with historical defect/rework outcomes, to surface an early risk score
per vehicle -- so inspection can be targeted rather than uniform.

Only stations with at least partial sensor coverage feed features here;
manual/no-sensor stations are excluded from the feature set (their state
is handled separately by confidence_inference.py) so the model never
fabricates precision it doesn't have.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


def build_vehicle_features(telemetry):
    """
    Aggregate per-unit, per-station telemetry into one feature row per
    vehicle (unit_id): mean/max torque and weld current across
    instrumented stations, and cycle-time variance across the whole
    line, since a defect's origin station is often unknown in advance.
    """
    instrumented = telemetry[telemetry.sensor_coverage.isin(["full", "partial"])]

    agg = instrumented.groupby("unit_id").agg(
        mean_torque=("torque_nm", "mean"),
        max_torque=("torque_nm", "max"),
        mean_weld_current=("weld_current_a", "mean"),
        max_weld_current=("weld_current_a", "max"),
        cycle_time_std=("cycle_time_s", "std"),
        cycle_time_max=("cycle_time_s", "max"),
    ).reset_index()

    label = telemetry.groupby("unit_id").defect_flag.any().reset_index()
    features = agg.merge(label, on="unit_id", how="inner")
    return features.fillna(features.median(numeric_only=True))


def train_defect_model(features, random_state=42):
    """Train/validate a gradient boosted classifier on the vehicle
    feature table. Reports ROC-AUC as the validation check referenced
    in the deck ('how you'd validate them before trusting their output')."""
    feature_cols = [c for c in features.columns if c not in ("unit_id", "defect_flag")]
    X = features[feature_cols]
    y = features.defect_flag.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    val_scores = model.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": roc_auc_score(y_test, val_scores) if y_test.nunique() > 1 else None,
        "report": classification_report(y_test, model.predict(X_test), zero_division=0),
        "feature_importance": dict(zip(feature_cols, model.feature_importances_.round(3))),
    }
    return model, feature_cols, metrics


def score_vehicles(model, features, feature_cols):
    """Score every vehicle and return a ranked inspection queue --
    the highest-risk units first, exactly the output the pitch deck
    promises for Layer 3."""
    scored = features.copy()
    scored["defect_risk_score"] = model.predict_proba(features[feature_cols])[:, 1]
    return scored[["unit_id", "defect_risk_score", "defect_flag"]].sort_values(
        "defect_risk_score", ascending=False
    ).reset_index(drop=True)


def inspection_queue(scored_vehicles, capacity_per_shift, units_per_shift=120):
    """
    Convert continuous risk scores into an actionable inspection queue:
    given inspection capacity (vehicles that can actually be checked per
    shift), return the top-risk vehicles up to that capacity instead of
    uniform sampling -- the Phase 3 'Targeted Routing' behaviour.
    """
    top = scored_vehicles.head(capacity_per_shift).copy()
    top["inspection_priority"] = range(1, len(top) + 1)
    return top
