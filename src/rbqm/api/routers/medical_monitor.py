from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from rbqm.api.deps import get_feature_tables
from rbqm.config import SAE_REPORTING_CLOCK_DAYS
from rbqm.ml.ae_signal import detect_ae_clusters, flag_conmed_interactions

router = APIRouter(prefix="/api/medical-monitor", tags=["Safety Surveillance Console"])


@router.get("/ae-clusters")
def ae_clusters(min_site_events: int = 2):
    tables, _, _ = get_feature_tables()
    clusters = detect_ae_clusters(tables["aes"], tables["subjects"], min_site_events=min_site_events)
    return clusters.to_dict(orient="records")


@router.get("/conmed-interactions")
def conmed_interactions():
    tables, _, _ = get_feature_tables()
    flags = flag_conmed_interactions(tables["aes"], tables["conmeds"])
    return flags.to_dict(orient="records")


@router.get("/sae-timeliness")
def sae_timeliness():
    """SAEs at risk of, or already past, the regulatory reporting clock."""
    tables, _, _ = get_feature_tables()
    aes = tables["aes"]
    sae = aes[aes["serious"]].copy()
    if sae.empty:
        return []
    sae["report_lag_days"] = (
        pd.to_datetime(sae["reported_date"]) - pd.to_datetime(sae["onset_date"])
    ).dt.days
    sae["clock_status"] = sae["report_lag_days"].apply(
        lambda d: "breach" if d > SAE_REPORTING_CLOCK_DAYS else ("at_risk" if d > SAE_REPORTING_CLOCK_DAYS - 3 else "on_time")
    )
    sae = sae.sort_values("report_lag_days", ascending=False)
    return sae[
        ["ae_id", "subject_id", "site_id", "meddra_pt", "onset_date", "reported_date", "report_lag_days", "clock_status"]
    ].to_dict(orient="records")
