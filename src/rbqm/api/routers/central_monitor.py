from __future__ import annotations

from fastapi import APIRouter

from rbqm.api.deps import get_feature_tables
from rbqm.ml.ae_signal import detect_ae_clusters
from rbqm.ml.kri_engine import compute_site_kris
from rbqm.ml.qtl_engine import evaluate_qtls, evaluate_study_qtls

router = APIRouter(prefix="/api/central-monitor", tags=["Centralized Statistical Monitoring"])


@router.get("/kri-dashboard")
def kri_dashboard():
    """Full KRI Dashboard payload: composite risk tier per site."""
    _, _, site_features = get_feature_tables()
    kris = compute_site_kris(site_features)
    tier_counts = kris["risk_tier"].value_counts().to_dict()
    return {"sites": kris.to_dict(orient="records"), "tier_counts": tier_counts}


@router.get("/qtl")
def qtl_status():
    """QTL boundary status per site plus study-level rollup."""
    _, _, site_features = get_feature_tables()
    site_qtls = evaluate_qtls(site_features)
    study_qtls = evaluate_study_qtls(site_features)
    return {
        "study_level": study_qtls.to_dict(orient="records"),
        "site_level": site_qtls.to_dict(orient="records"),
    }


@router.get("/statistical-outliers")
def statistical_outliers(min_site_events: int = 2):
    """Cross-site statistical monitoring signal: AE reporting rate outliers."""
    tables, _, _ = get_feature_tables()
    clusters = detect_ae_clusters(tables["aes"], tables["subjects"], min_site_events=min_site_events)
    return clusters.to_dict(orient="records")
