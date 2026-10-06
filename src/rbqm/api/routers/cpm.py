from __future__ import annotations

from fastapi import APIRouter

from rbqm.api.deps import get_feature_tables
from rbqm.ml.kri_engine import compute_site_kris, predict_premature_termination_rate
from rbqm.ml.qtl_engine import evaluate_study_qtls

router = APIRouter(prefix="/api/cpm", tags=["Executive Risk Dashboard"])


@router.get("/portfolio-summary")
def portfolio_summary():
    tables, _, site_features = get_feature_tables()
    kris = compute_site_kris(site_features)
    study_qtls = evaluate_study_qtls(site_features)
    termination_forecast = predict_premature_termination_rate(site_features)

    subjects = tables["subjects"]
    status_counts = subjects["status"].value_counts().to_dict()

    return {
        "n_sites": int(len(site_features)),
        "n_subjects": int(len(subjects)),
        "subject_status_counts": status_counts,
        "site_risk_tier_counts": kris["risk_tier"].value_counts().to_dict(),
        "premature_termination_forecast": termination_forecast,
        "study_qtls": study_qtls.to_dict(orient="records"),
        "highest_risk_sites": kris.head(5)[["site_id", "site_name", "risk_tier", "composite_risk_score", "top_drivers"]].to_dict(
            orient="records"
        ),
    }
