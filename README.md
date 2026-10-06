# RBQM Studio — Local Reference Implementation

Risk Based Quality Management(RBQM) is a proactive, data driven framework used in 
clinical trials and pharmaceutical development to manage quality across the entire study lifecycle.

Instead of applying uniform monitoring or waiting until the end of a study to catch errors, 
RBQM focuses resources on the processes and data points that are critical-to-quality (CTQ) 
and most likely to affect patient safety or trial integrity. 

A runnable, locally-testable implementation of the ML powered clinical data
& risk surveillance platform: role-based views for **Data Managers, CRAs,
Central Monitors, Medical Monitors, and CPMs**, backed by real ML modules
(anomaly detection, KRI/QTL engines, AE signal detection, an AI
reconciliation assistant) running against a synthetic multi-site trial
dataset.

This is a scoped down, locally runnable version of the full architecture —
SQLite stands in for Delta Lake/S3, scikit-learn models stand in for
SageMaker, and a rules+similarity reconciliation assistant stands in for the
LLM+pgvector retrieval assistant described in the design doc. The module
boundaries mirror the production architecture 1:1, so swapping in AWS
services later is a matter of replacing implementations behind the same
interfaces, not a redesign.

## What's inside

```
rbqm_studio/
├── src/rbqm/
│   ├── config.py            # KRI weights, QTL boundaries, study parameters
│   ├── db.py                 # SQLAlchemy models (Site, Subject, AE, ConMed, Lab, PD, Query, ePRO...)
│   ├── data_generator.py     # synthetic multi-site trial data w/ seeded anomalies & signals
│   ├── features.py           # subject-level & site-level feature engineering
│   ├── ml/
│   │   ├── anomaly_detection.py   # Isolation Forest cross-domain anomaly worklist
│   │   ├── kri_engine.py          # composite Low/Medium/High site risk scoring + trajectory forecast
│   │   ├── qtl_engine.py          # Quality Tolerance Limit boundary monitoring
│   │   ├── reconciliation.py      # AI reconciliation assistant (central lab vs EDC) + feedback log
│   │   └── ae_signal.py           # AE cluster/disproportionality detection + ConMed interaction flags
│   ├── api/                  # FastAPI service, one router per persona
│   └── dashboard/            # Streamlit UI, one view per persona
├── tests/                    # pytest suite (45 tests) covering data, features, and every ML module + API
└── scripts/                  # seed_db.py, run_api.sh, run_dashboard.sh
```

## Setup (macOS)

Requires Python 3.9+ (check with `python3 --version`). Python 3.9 is EOL
soon and several dependencies (FastAPI, Streamlit, Pydantic) will drop
support for it in future releases, so `brew install python@3.12` and
recreating the venv against that is worth doing eventually — but 3.9 works
today.

```bash
cd rbqm_studio
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Generate the synthetic dataset

```bash
python scripts/seed_db.py
```

This creates `data/rbqm.db` — a 12-site, 240-subject synthetic trial with a
handful of **deliberately seeded signals** so you can verify the ML modules
actually catch them:

| Seeded signal | Site | What should catch it |
|---|---|---|
| AE under-reporting | SITE-002 | Central Statistical Monitoring outlier detection |
| High protocol deviation rate | SITE-005 | KRI Dashboard (High risk tier), anomaly worklist |
| High subject dropout / low ePRO compliance | SITE-009 | Predicted premature termination forecast |
| Late SAE reporting (>15 days) | SITE-011 | QTL breach, SAE timeliness console |
| Lab value transcription errors (10x, unit mismatch) | scattered, ~7% of labs | AI Reconciliation Assistant |
| Warfarin + Bleeding AE co-occurrence | 1 explicit subject + random scatter | ConMed interaction flagging |

## Run the tests

```bash
python -m pytest -v
```

45 tests across data generation, feature engineering, each ML module, and
every API endpoint — including assertions that the seeded signals above are
actually recovered by the corresponding engine (not just "code runs without
error").

## Run the API

```bash
./scripts/run_api.sh
# or: uvicorn rbqm.api.main:app --reload --port 8000 (with PYTHONPATH=src)
```

Interactive docs: **http://127.0.0.1:8000/docs**

Key endpoints:
- `GET /api/dm/anomalies` — AI-ranked anomaly worklist
- `GET /api/dm/reconciliation` — AI reconciliation assistant suggestions
- `GET /api/cra/sites/{site_id}` — site-level KRI + query/PD summary
- `GET /api/central-monitor/kri-dashboard` — full KRI dashboard payload
- `GET /api/central-monitor/qtl` — QTL boundary status
- `GET /api/medical-monitor/ae-clusters` — AE signal detection
- `GET /api/medical-monitor/conmed-interactions` — safety interaction flags
- `GET /api/cpm/portfolio-summary` — executive risk rollup + trajectory forecast

## Run the interactive dashboard

```bash
./scripts/run_dashboard.sh
# or: streamlit run src/rbqm/dashboard/app.py (with PYTHONPATH=src)
```

Opens at **http://localhost:8501**. Use the sidebar to switch between the
five persona views (KRI Dashboard, Data Manager Studio, CRA Site Workbench,
Safety Console, QTL Console).

## Resetting the data

Data is deterministic (seeded with `RANDOM_SEED=42` in `config.py`), so
re-running `python scripts/seed_db.py` wipes and regenerates an identical
dataset — useful after changing generator logic or KRI/QTL thresholds in
`config.py`.

## Notes on scope

This is a **local reference implementation**, not the production system:
- No auth/RBAC (all five persona views are accessible to anyone running it)
- No 21 CFR Part 11 audit trail / e-signature persistence (the `FeedbackLog`
  in `reconciliation.py` shows where that would hook in)
- AI reconciliation assistant uses rule-based hypothesis generation rather
  than an LLM call — the interface (`ReconciliationSuggestion`) is designed
  so an LLM+retrieval implementation is a drop-in replacement
- AE clustering is a simplified disproportionality check, not a validated
  pharmacovigilance signal-detection method (PRR/EBGM) — clearly labeled as
  such in code comments
