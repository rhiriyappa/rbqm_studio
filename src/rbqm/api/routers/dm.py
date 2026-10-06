from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from rbqm.api.deps import get_feature_tables
from rbqm.ml.anomaly_detection import detect_subject_anomalies
from rbqm.ml.reconciliation import find_lab_discrepancies

router = APIRouter(prefix="/api/dm", tags=["Data Manager Studio"])


@router.get("/anomalies")
def anomalies(top_n: int = Query(20, ge=1, le=100)):
    """AI-ranked anomaly worklist across all clinical domains."""
    _, subject_features, _ = get_feature_tables()
    result = detect_subject_anomalies(subject_features, top_n=top_n)
    return result.to_dict(orient="records")


@router.get("/reconciliation")
def reconciliation(subject_id: Optional[str] = None):
    """AI Reconciliation Assistant: central lab vs. EDC-transcribed discrepancies."""
    tables, _, _ = get_feature_tables()
    suggestions = find_lab_discrepancies(tables["labs"], subject_id=subject_id)
    return [s.__dict__ for s in suggestions]


@router.get("/data-profile/{domain}")
def data_profile(domain: str):
    """Visual data profile (distribution + missingness) for a given domain,
    consumed by the front end to render the profile card -- no stats needed.
    """
    tables, _, _ = get_feature_tables()
    if domain not in tables:
        return {"error": f"unknown domain '{domain}'", "available_domains": list(tables.keys())}

    df = tables[domain]
    if df.empty:
        return {"domain": domain, "row_count": 0, "columns": []}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    profile = {
        "domain": domain,
        "row_count": len(df),
        "missingness": {col: float(df[col].isna().mean()) for col in df.columns},
        "numeric_summary": {
            col: {
                "mean": float(df[col].mean()) if not df[col].isna().all() else None,
                "std": float(df[col].std()) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            }
            for col in numeric_cols
        },
    }
    return profile
