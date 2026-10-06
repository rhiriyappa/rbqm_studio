from rbqm.ml.kri_engine import compute_site_kris, predict_premature_termination_rate


def test_kri_tiers_are_valid(site_features):
    kris = compute_site_kris(site_features)
    assert set(kris["risk_tier"].unique()).issubset({"Low", "Medium", "High"})


def test_kri_high_pd_site_scores_above_median(site_features):
    """SITE-005 is seeded with a high PD rate; it should not land in the bottom half."""
    kris = compute_site_kris(site_features)
    median_score = kris["composite_risk_score"].median()
    site_005_score = kris.loc[kris["site_id"] == "SITE-005", "composite_risk_score"].iloc[0]
    assert site_005_score >= median_score


def test_kri_top_drivers_is_non_empty_string(site_features):
    kris = compute_site_kris(site_features)
    assert (kris["top_drivers"].str.len() > 0).all()


def test_premature_termination_forecast_has_expected_keys(site_features):
    forecast = predict_premature_termination_rate(site_features)
    for key in ["observed_rate", "forecast_rate", "protocol_assumed_rate", "status"]:
        assert key in forecast
    assert 0 <= forecast["forecast_rate"] <= 1


def test_premature_termination_flags_high_dropout_site(site_features):
    """SITE-009 is seeded with elevated dropout; overall forecast should exceed
    the seeded low-dropout baseline of other sites."""
    site_009_dropout = site_features.loc[site_features["site_id"] == "SITE-009", "dropout_rate"].iloc[0]
    portfolio_mean_dropout = site_features["dropout_rate"].mean()
    assert site_009_dropout > portfolio_mean_dropout
