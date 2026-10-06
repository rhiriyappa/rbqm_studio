"""Cross-domain subject-level anomaly detection.

Powers the Data Manager Studio's "AI-ranked anomaly worklist": an Isolation
Forest scores every subject on a feature vector spanning AE, PD, query, lab,
and ePRO domains, so an anomaly can be caught even if no single domain looks
extreme in isolation (e.g. a subject with mildly elevated labs AND a rising
query count AND a missed visit -- individually unremarkable, jointly odd).

No statistics literacy is required to consume the output: every flagged
subject gets a plain-language explanation built from which features drove
the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from rbqm.config import ANOMALY_CONTAMINATION, RANDOM_SEED

FEATURE_COLUMNS = [
    "ae_count",
    "sae_count",
    "pd_count",
    "major_pd_count",
    "open_query_count",
    "lab_oor_rate",
    "epro_compliance_rate",
]


def detect_subject_anomalies(subject_features: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Returns the ranked anomaly worklist (highest-impact issues first)."""
    if subject_features.empty:
        return subject_features

    df = subject_features.copy()
    X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_SEED,
    )
    model.fit(X_scaled)

    # decision_function: higher = more normal. Flip sign so higher = more anomalous.
    raw_scores = -model.decision_function(X_scaled)
    is_outlier = model.predict(X_scaled) == -1

    df["anomaly_score"] = raw_scores
    df["is_anomaly"] = is_outlier
    df["confidence_pct"] = (
        100 * (df["anomaly_score"] - df["anomaly_score"].min())
        / max(df["anomaly_score"].max() - df["anomaly_score"].min(), 1e-9)
    ).round(1)

    z = (X_scaled - X_scaled.mean(axis=0)) / (X_scaled.std(axis=0) + 1e-9)
    z_df = pd.DataFrame(z, columns=FEATURE_COLUMNS, index=df.index)
    df["explanation"] = [
        _explain_row(z_df.loc[i]) for i in df.index
    ]

    ranked = df.sort_values("anomaly_score", ascending=False)
    return ranked[
        [
            "subject_id",
            "site_id",
            "status",
            "anomaly_score",
            "confidence_pct",
            "is_anomaly",
            "explanation",
        ]
        + FEATURE_COLUMNS
    ].head(top_n)


def _explain_row(z_row: pd.Series, n_reasons: int = 2) -> str:
    """Builds a plain-language explanation from the top contributing features."""
    top = z_row.abs().sort_values(ascending=False).head(n_reasons)
    labels = {
        "ae_count": "adverse event count",
        "sae_count": "serious adverse event count",
        "pd_count": "protocol deviation count",
        "major_pd_count": "major protocol deviation count",
        "open_query_count": "open query count",
        "lab_oor_rate": "share of out-of-range labs",
        "epro_compliance_rate": "ePRO compliance",
    }
    reasons = []
    for feature, _ in top.items():
        direction = "higher than" if z_row[feature] > 0 else "lower than"
        reasons.append(f"{labels[feature]} is {direction} peer average")
    return "Flagged because: " + "; ".join(reasons) + "."
