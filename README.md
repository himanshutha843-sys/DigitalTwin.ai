# DigitalTwin.ai — Predictive Digital Twin for Vehicle Assembly Lines
**Team Himanshu6030** · Himanshu Thakur (BSBE, IIT Guwahati) · Deepak Yadav (ECE, IIT Guwahati)

> A live, honest picture of the line — not a perfect one.

---

## 🎯 Problem

On a mixed-model vehicle assembly line (30–50 stations across body construction, paint, and final assembly), a local bottleneck rarely stays local. A six-minute drift at one mid-line station can push 30–40 vehicles into the downstream queue before anyone notices — because inspection happens at fixed checkpoints, not continuously, and MES/PLC systems track throughput station-by-station without ever correlating it with the defects that show up later.

Two structural gaps make this worse:
- **Uneven sensor coverage** — a majority of stations are well-instrumented; a meaningful minority rely on manual checks.
- **Reactive detection** — plant teams only see the problem after a shift-end quality report, by which point the fix is rework, not prevention.

## 💡 Proposed Solution

A plant-floor digital twin built in **three layers**, fed by existing MES/PLC signals and manual entry where sensors don't exist:

| Layer | Data In | Method | Output |
|---|---|---|---|
| **1. Live Line Model** | MES/PLC tags, manual entry forms, station heartbeat pings | Each station modeled as a node with cycle time, queue length, and status, refreshed on a short polling interval against a rolling multi-shift baseline | Real-time map of where work-in-progress is piling up |
| **2. Bottleneck Detection** | Per-station cycle-time stream, queue growth | Control-chart-style thresholds flag drift from the rolling baseline; a queueing projection estimates downstream ripple in vehicles and minutes | Early drift alert + projected ripple window |
| **3. Defect Prediction** | Torque, weld current, cycle-time variance, historical defect/rework logs | Lightweight classifier trained on labeled defect outcomes scores each vehicle's risk from its upstream process signature | Ranked inspection queue |

**Confidence-aware fallback:** where sensor coverage is thin, the twin infers a station's state from neighboring stations' timing instead of guessing blind — and explicitly tags that inference as low-confidence, so plant teams know how much weight to give it.

We deliberately stopped at three layers — each answers a question a supervisor actually asks (where's the work piling up, what's about to become a problem, which vehicles are worth a second look). A fourth layer, e.g. full physics simulation, would add modeling cost without changing floor-level decisions.

## 🧪 Working Prototype

The prototype demonstrates the core predictive mechanism on **simulated production data** (illustrative, not real enterprise data), since live PLC/MES access wasn't available for this round.

- 40-station line (body/paint/final assembly), 6 shifts, 720 vehicles, with 3 injected drift/bottleneck events and realistic uneven sensor coverage (~63% full, ~14% partial, ~23% manual/no-sensor)
- **Layer 1 — Live Line Model:** per-station snapshot with status classification (`nominal` / `watch` / `critical` / `no_data`)
- **Layer 2 — Bottleneck Detection:** rolling-baseline control-chart thresholds (z-score > 3σ) flag drift, with a queueing-based ripple projection into downstream minutes and vehicles queued
- **Confidence layer:** manual/no-sensor stations get their state inferred from neighboring instrumented stations' timing, explicitly tagged `low` confidence — never presented as measured data
- **Layer 3 — Defect Prediction:** a gradient-boosted classifier trained on upstream torque/weld-current/cycle-time-variance features, validated on a held-out split (**ROC-AUC ≈ 0.83**), producing a ranked inspection queue
- Result: targeting the top 15 highest-risk vehicles per shift (~2% of production) catches **~15% of all defects** — vs. the ~2% you'd expect from uniform random sampling at that same inspection capacity

### Tech Stack
Python · pandas / numpy · scikit-learn (GradientBoostingClassifier) · matplotlib

### Repository Structure
```
├── data/
│   └── simulate_line.py          # station roster + telemetry simulator, drift injection
├── src/
│   ├── live_line_model.py        # Layer 1: live station snapshot
│   ├── bottleneck_detection.py   # Layer 2: control-chart drift + ripple projection
│   ├── confidence_inference.py   # confidence-aware fallback for manual stations
│   └── defect_prediction.py      # Layer 3: classifier + ranked inspection queue
├── demo/
│   ├── run_demo.py               # end-to-end orchestration + stakeholder views
│   └── output/                   # generated CSVs + summary chart
├── requirements.txt
└── README.md
```

### How to Run
```bash
pip install -r requirements.txt
python demo/run_demo.py
```
This prints the floor-supervisor, plant-manager, and leadership views to the console and saves `demo/output/control_room_summary.png` (bottleneck detection, ripple projection, and defect-risk distribution) plus the underlying CSVs.

### Interactive Dashboard
For the demo video / live walkthrough, run the Streamlit control-room dashboard:
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```
Three tabs, matching the stakeholders in the pitch deck:
- **👷 Floor Supervisor** — live station-status map (color = drift status, shape = sensor coverage), active drift alerts with projected delay/vehicles queued, and a confidence-coverage breakdown
- **📋 Plant Manager** — cycle-time-vs-baseline trend for the most-flagged station, projected downstream delay by station, and the ranked inspection queue
- **📊 Leadership** — model validation (held-out ROC-AUC, lift over uniform sampling), the 3-phase rollout roadmap, expected impact metrics, and sensor-coverage mix

Sidebar controls let you re-roll the simulation seed and adjust inspection capacity per shift to show how the targeted queue adapts.

## 📈 Rollout Plan

| Phase | Scope |
|---|---|
| **1 — Shadow Mode** | 2–3 pilot stations. Twin watches and flags but makes no changes to inspection or line operations; output compared against what actually happened. |
| **2 — Line-Wide Advisory** | Bottleneck alerts and ripple projections go live across the full line, visible to plant supervisors — still a human decision every time. |
| **3 — Targeted Routing** | Defect-risk scores actively route which vehicles get pulled for inspection, replacing uniform sampling with targeted sampling. |

**What we'd expect to move:** time to detect drift (minutes, not a shift-end report), downstream rework, inspection targeting, usable sensor coverage.

**Risks we're watching:** false-positive alerts eroding floor-level trust in the twin, and classifier drift as the line changes — both are why Phase 1 stays advisory-only and every score ships with a confidence tag.

## 🎥 Demo Video
*(link to be added)*

## 📄 Business Proposal
See `/docs/business-proposal.md` *(or link to the full document once created)* for problem framing, business case, target users, and risk mitigations in full.

---
*Built for DigitalTwin.ai Round 2.*
