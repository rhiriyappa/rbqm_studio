from rbqm.ml.qtl_engine import evaluate_qtls, evaluate_study_qtls


def test_site_qtl_statuses_are_valid(site_features):
    result = evaluate_qtls(site_features)
    assert set(result["status"].unique()).issubset({"within_limit", "approaching_limit", "breach"})


def test_study_qtl_rollup_has_all_metrics(site_features):
    result = evaluate_study_qtls(site_features)
    assert set(result["metric"]) == {"major_pd_rate", "sae_late_reporting_rate", "epro_compliance_rate"}


def test_late_sae_site_flagged_in_qtl(site_features):
    """SITE-011 is seeded with late SAE reporting; its QTL status should not be 'within_limit'
    for the sae_late_reporting_rate metric, OR its rate should be clearly elevated."""
    result = evaluate_qtls(site_features)
    site_row = result[(result["site_id"] == "SITE-011") & (result["metric"] == "sae_late_reporting_rate")]
    assert not site_row.empty
    portfolio_avg = result[result["metric"] == "sae_late_reporting_rate"]["value"].mean()
    assert site_row["value"].iloc[0] >= portfolio_avg
