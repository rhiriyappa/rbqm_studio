import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from fastapi.testclient import TestClient

from rbqm.config import DB_PATH
from rbqm.data_generator import generate
from rbqm.db import get_engine, get_session, init_db


@pytest.fixture(scope="session", autouse=True)
def seed_file_db():
    """The API layer reads from the file-based SQLite DB (config.DB_URL), so
    for API tests we seed that file directly (separate from the in-memory
    fixtures used by the unit tests in conftest.py)."""
    if DB_PATH.exists():
        DB_PATH.unlink()
    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    generate(session, seed=42)
    session.close()
    yield
    # leave the seeded DB in place after tests so `streamlit run` / `uvicorn`
    # have data to show immediately after `pytest` is run


@pytest.fixture(scope="session")
def client():
    from rbqm.api.main import app

    return TestClient(app)


def test_health(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_dm_anomalies(client):
    resp = client.get("/api/dm/anomalies?top_n=10")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 10
    assert "explanation" in body[0]


def test_dm_reconciliation(client):
    resp = client.get("/api/dm/reconciliation")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if body:
        assert "confidence_pct" in body[0]


def test_dm_data_profile(client):
    resp = client.get("/api/dm/data-profile/aes")
    assert resp.status_code == 200
    body = resp.json()
    assert body["domain"] == "aes"
    assert body["row_count"] > 0


def test_dm_data_profile_unknown_domain(client):
    resp = client.get("/api/dm/data-profile/not_a_real_domain")
    assert resp.status_code == 200
    assert "error" in resp.json()


def test_cra_site_summary(client):
    resp = client.get("/api/cra/sites/SITE-001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["site_id"] == "SITE-001"
    assert "kri_summary" in body


def test_cra_site_summary_unknown_site(client):
    resp = client.get("/api/cra/sites/SITE-999")
    assert resp.status_code == 404


def test_cra_query_aging(client):
    resp = client.get("/api/cra/queries/aging")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    if len(body) > 1:
        ages = [r["age_days"] for r in body]
        assert ages == sorted(ages, reverse=True)


def test_central_monitor_kri_dashboard(client):
    resp = client.get("/api/central-monitor/kri-dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert "sites" in body and "tier_counts" in body


def test_central_monitor_qtl(client):
    resp = client.get("/api/central-monitor/qtl")
    assert resp.status_code == 200
    body = resp.json()
    assert "study_level" in body and "site_level" in body


def test_central_monitor_statistical_outliers(client):
    resp = client.get("/api/central-monitor/statistical-outliers")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_medical_monitor_ae_clusters(client):
    resp = client.get("/api/medical-monitor/ae-clusters")
    assert resp.status_code == 200


def test_medical_monitor_conmed_interactions(client):
    resp = client.get("/api/medical-monitor/conmed-interactions")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert any(r["drug_name"] == "Warfarin" for r in body)


def test_medical_monitor_sae_timeliness(client):
    resp = client.get("/api/medical-monitor/sae-timeliness")
    assert resp.status_code == 200
    body = resp.json()
    if body:
        assert body[0]["clock_status"] in {"on_time", "at_risk", "breach"}


def test_cpm_portfolio_summary(client):
    resp = client.get("/api/cpm/portfolio-summary")
    assert resp.status_code == 200
    body = resp.json()
    for key in ["n_sites", "n_subjects", "site_risk_tier_counts", "premature_termination_forecast", "highest_risk_sites"]:
        assert key in body
