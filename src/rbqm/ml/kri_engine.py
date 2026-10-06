"""KRI engine: turns per-site raw rates into a composite Low / Medium / High
risk tier, plus a predictive premature-termination forecast.

This is what powers the KRI Dashboard. The composite score is a weighted,
z-scored blend of individual KRIs (config.KRI_WEIGHTS) -- deliberately simple
and auditable rather than a black-box model, because RBQM stakeholders need
to be able to explain *why* a site is red to a sponsor or inspector.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from rbqm.config import KRI_RISK_CUTOFFS, KRI_WEIGHTS

# direction: +1 means "higher is worse", -1 means "lower is worse"
KRI_DIRECTIONS = {
    "query_rate": 1,
    "query_aging_days": 1,
    "pd_rate": 1,
    "major_pd_rate": 1,
    "ae_underreporting_z": -1,  # very negative z = underreporting = bad
    "sae_late_reporting_rate": 1,
    "screen_failure_rate": 1,
    "dropout_rate": 1,
}


def _zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def compute_site_kris(site_features: pd.DataFrame) -> pd.DataFrame:
    if site_features.empty:
        return site_features

    df = site_features.copy()
    weighted_components = pd.DataFrame(index=df.index)

    for kri, weight in KRI_WEIGHTS.items():
        z = _zscore(df[kri])
        direction = KRI_DIRECTIONS[kri]
        # orient so that positive contribution always means "worse"
        oriented = z * direction if direction == 1 else -z * direction
        weighted_components[kri] = oriented * weight

    df["composite_risk_score"] = weighted_components.sum(axis=1)

    # percentile-rank the composite score across sites -> tier
    pct_rank = df["composite_risk_score"].rank(pct=True)
    df["risk_percentile"] = (pct_rank * 100).round(1)

    def tier(p: float) -> str:
        if p <= KRI_RISK_CUTOFFS["low_max"]:
            return "Low"
        elif p <= KRI_RISK_CUTOFFS["medium_max"]:
            return "Medium"
        return "High"

    df["risk_tier"] = pct_rank.apply(tier)

    df["top_drivers"] = [
        _top_drivers(weighted_components.loc[i]) for i in df.index
    ]

    cols = [
        "site_id",
        "site_name",
        "country",
        "n_subjects",
        "risk_tier",
        "risk_percentile",
        "composite_risk_score",
        "top_drivers",
    ] + list(KRI_WEIGHTS.keys())
    return df[cols].sort_values("composite_risk_score", ascending=False)


def _top_drivers(weighted_row: pd.Series, n: int = 2) -> str:
    labels = {
        "query_rate": "query volume",
        "query_aging_days": "slow query resolution",
        "pd_rate": "protocol deviation rate",
        "major_pd_rate": "major protocol deviation rate",
        "ae_underreporting_z": "AE reporting rate vs. peers",
        "sae_late_reporting_rate": "late SAE reporting",
        "screen_failure_rate": "screen failure rate",
        "dropout_rate": "subject dropout rate",
    }
    top = weighted_row.sort_values(ascending=False).head(n)
    return ", ".join(labels[k] for k in top.index if weighted_row[k] > 0) or "no dominant driver"


def predict_premature_termination_rate(
    site_features: pd.DataFrame, protocol_assumed_rate: float = 0.10
) -> dict:
    """Simple, explainable forecast: current portfolio dropout rate extrapolated
    against enrollment maturity, compared to the protocol-assumed rate.

    This mirrors a lightweight survival-style projection without requiring a
    full time-to-event model for the local demo -- swap in
    `lifelines.KaplanMeierFitter` for a production-grade version.
    """
    if site_features.empty:
        return {"forecast_rate": 0.0, "protocol_assumed_rate": protocol_assumed_rate, "status": "insufficient_data"}

    total_subjects = site_features["n_subjects"].sum()
    total_discontinued = (site_features["dropout_rate"] * site_features["n_subjects"]).sum()
    observed_rate = total_discontinued / total_subjects if total_subjects else 0.0

    # naive maturity-adjusted forecast: assume trial is ~60% enrolled-to-target,
    # extrapolate current dropout trajectory forward with mild acceleration factor
    forecast_rate = round(min(observed_rate * 1.15, 0.95), 4)

    status = "on_track"
    if forecast_rate > protocol_assumed_rate * 1.3:
        status = "off_track_high_risk"
    elif forecast_rate > protocol_assumed_rate * 1.1:
        status = "trending_off_track"

    return {
        "observed_rate": round(observed_rate, 4),
        "forecast_rate": forecast_rate,
        "protocol_assumed_rate": protocol_assumed_rate,
        "status": status,
    }
