"""Central configuration: file paths and study-level thresholds.

Keeping every tunable in one place is what lets a central statistician
configure KRI weights / QTL boundaries without touching the ML or API code.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "rbqm.db"
DB_URL = f"sqlite:///{DB_PATH}"

# ---------------------------------------------------------------------------
# Synthetic study parameters (used by the data generator)
# ---------------------------------------------------------------------------
N_SITES = 12
N_SUBJECTS = 240
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# KRI engine: weights for the composite site risk score.
# Each KRI is z-scored across sites, then combined with these weights.
# ---------------------------------------------------------------------------
KRI_WEIGHTS = {
    "query_rate": 0.15,           # open queries per subject
    "query_aging_days": 0.15,     # median days a query stays open
    "pd_rate": 0.20,              # protocol deviations per subject
    "major_pd_rate": 0.15,        # major protocol deviations per subject
    "ae_underreporting_z": 0.15,  # AE rate vs portfolio peer group (negative = underreporting)
    "sae_late_reporting_rate": 0.10,  # % of SAEs reported outside regulatory clock
    "screen_failure_rate": 0.05,
    "dropout_rate": 0.05,
}

# Composite score cut points -> risk tier
KRI_RISK_CUTOFFS = {
    "low_max": 0.33,     # composite percentile below this -> Low
    "medium_max": 0.66,  # below this -> Medium, else High
}

# ---------------------------------------------------------------------------
# QTL definitions: metric -> (target, tolerance_limit, direction)
# direction "above" means breach if metric rises above the limit.
# ---------------------------------------------------------------------------
QTL_DEFINITIONS = {
    "major_pd_rate": {
        "label": "Major Protocol Deviation Rate",
        "target": 0.05,
        "limit": 0.10,
        "direction": "above",
        "approach_band": 0.85,  # fraction of limit that triggers "approaching" state
    },
    "sae_late_reporting_rate": {
        "label": "SAE Late Reporting Rate (>15 days)",
        "target": 0.02,
        "limit": 0.08,
        "direction": "above",
        "approach_band": 0.85,
    },
    "epro_compliance_rate": {
        "label": "ePRO Compliance Rate",
        "target": 0.90,
        "limit": 0.75,
        "direction": "below",
        "approach_band": 0.90,
    },
}

# Regulatory reporting clocks (days) used by the safety module
SAE_REPORTING_CLOCK_DAYS = 15
SAE_FATAL_LIFE_THREATENING_CLOCK_DAYS = 7

# Anomaly detection
ANOMALY_CONTAMINATION = 0.06  # expected fraction of anomalous subjects
