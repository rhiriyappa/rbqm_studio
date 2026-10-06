from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException

from rbqm.api.deps import get_feature_tables
from rbqm.ml.kri_engine import compute_site_kris

router = APIRouter(prefix="/api/cra", tags=["CRA Site Workbench"])


@router.get("/sites/{site_id}")
def site_summary(site_id: str):
    """Site-level KRI trend + open query / PD summary for visit prep."""
    tables, _, site_features = get_feature_tables()
    if site_id not in set(site_features["site_id"]):
        raise HTTPException(status_code=404, detail=f"Unknown site_id '{site_id}'")

    kris = compute_site_kris(site_features)
    site_row = kris[kris["site_id"] == site_id].to_dict(orient="records")[0]

    queries = tables["queries"]
    site_queries = queries[queries["site_id"] == site_id]
    open_queries = site_queries[site_queries["status"] == "open"]

    pds = tables["pds"]
    site_pds = pds[pds["site_id"] == site_id]

    return {
        "site_id": site_id,
        "kri_summary": site_row,
        "open_query_count": int(len(open_queries)),
        "open_queries_by_domain": open_queries.groupby("domain").size().to_dict(),
        "protocol_deviation_count": int(len(site_pds)),
        "major_pd_count": int(site_pds["major"].sum()) if not site_pds.empty else 0,
    }


@router.get("/queries/aging")
def query_aging(site_id: Optional[str] = None):
    """Open query list sorted by age, for CRA prioritization."""
    tables, _, _ = get_feature_tables()
    queries = tables["queries"].copy()
    if site_id:
        queries = queries[queries["site_id"] == site_id]
    open_q = queries[queries["status"] == "open"].copy()
    if open_q.empty:
        return []
    import pandas as pd

    open_q["age_days"] = (pd.Timestamp("2026-07-01") - pd.to_datetime(open_q["opened_date"])).dt.days
    open_q = open_q.sort_values("age_days", ascending=False)
    return open_q[["query_id", "subject_id", "site_id", "domain", "opened_date", "age_days"]].to_dict(
        orient="records"
    )
