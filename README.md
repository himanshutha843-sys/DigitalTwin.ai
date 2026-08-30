# DigitalTwin.ai

DigitalTwin.ai is a read-only predictive digital twin for mixed-model vehicle assembly lines. It turns uneven plant-floor data into early bottleneck alerts, delayed-defect attribution, and role-specific decisions for supervisors, plant managers, and leadership.

The prototype targets a 30 to 50 station vehicle line across body, paint, and final assembly. It is built around real operating constraints: legacy equipment, partial instrumentation, manual checks, delayed end-of-line quality signals, and strict OT rules that prevent PLC changes outside maintenance windows.

## Project Hook

Most factory dashboards show what already happened. DigitalTwin.ai scores what is becoming risky now, even when the data is incomplete.

## Architecture Overview

```text
data_simulator.py
    |
    |-- data/station_master.csv
    |-- data/synthetic_station_events.csv
    |-- data/vehicle_quality_summary.csv
    v
twin_model.py
    |
    |-- telemetry imputation for legacy/manual stations
    |-- Random Forest defect-risk model
    |-- SPC + Isolation Forest bottleneck detection
    |-- OEE, MTBF, and savings metrics
    v
model_outputs/
    |
    |-- defect_risk_scores.csv
    |-- bottleneck_alerts.csv
    |-- station_health.csv
    |-- model_metrics.json
    v
app.py
    |
    |-- Supervisor View
    |-- Plant Manager View
    |-- Leadership View
```

## What the Prototype Demonstrates

- A 45-station assembly line with modern, partial, legacy, and manual-check stations.
- Intentional telemetry gaps and nulls for realistic uneven sensor coverage.
- A delayed multi-causal defect pattern: elevated Station 12 temperature plus Station 15 vibration creates defects detected later at Station 45.
- Imputation that uses station medians, process-area medians, global fallback, and missingness flags.
- Defect probability scores with adjustable alert thresholds for false-alarm control.
- Bottleneck detection using statistical process control and Isolation Forest anomaly detection.
- Supervisor, plant manager, and leadership views from the same underlying digital twin.

## Repository Structure

```text
.
|-- BUSINESS_PROPOSAL.md          # Phase 1 business proposal
|-- data_simulator.py             # Phase 2 synthetic data generator
|-- twin_model.py                 # Phase 3 predictive engine
|-- app.py                        # Phase 4 Streamlit dashboard
|-- README.md                     # Phase 5 project guide
|-- requirements.txt
|-- data/
|   |-- station_master.csv
|   |-- synthetic_station_events.csv
|   |-- vehicle_quality_summary.csv
|   `-- simulate_line.py          # earlier simulator retained for reference
|-- dashboard/
|   `-- app.py                    # earlier dashboard retained for reference
|-- demo/
|   `-- run_demo.py
`-- src/
    |-- bottleneck_detection.py
    |-- confidence_inference.py
    |-- defect_prediction.py
    `-- live_line_model.py
```

## Local Setup

From the project directory:

```powershell
cd C:\Users\deepa\OneDrive\Desktop\digital_twin\DigitalTwin.ai
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If Windows does not expose `python`, use the Python launcher:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the Prototype

Generate the synthetic plant-floor data:

```powershell
python data_simulator.py
```

Run the predictive engine:

```powershell
python twin_model.py
```

Launch the dashboard:

```powershell
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dashboard Views

### Supervisor View

- Real-time station status across the line.
- Active bottleneck alerts with projected delay and queued vehicles.
- High-probability defect warnings for current WIP.

### Plant Manager View

- Weekly aggregation of bottleneck alerts and defect rate.
- Station health trends by alert volume and process area.
- MTBF-style analytics based on repeated bottleneck event clusters.

### Leadership View

- OEE roll-up for availability, performance, and quality.
- Model validation metrics including ROC-AUC and average precision.
- Estimated annual cost savings from targeted defect prevention.
- False-alarm threshold tradeoff for alert governance.

## Business Proposal

See [BUSINESS_PROPOSAL.md](BUSINESS_PROPOSAL.md) for the full business case, rollout plan, target-user value propositions, data-gap strategy, and risk mitigations.

## Demo Video

[Link to Demo Video]

## Notes for Judges

This is a simulated-data prototype. It is intentionally designed as a read-only advisory layer: no PLC writes, no control-loop changes, and no production routing automation without human approval. The goal is to show how a practical digital twin can create value even before a plant reaches perfect instrumentation maturity.

