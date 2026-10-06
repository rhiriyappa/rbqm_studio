def test_subject_features_shape(subject_features, tables):
    assert len(subject_features) == len(tables["subjects"])
    for col in ["ae_count", "pd_count", "open_query_count", "lab_oor_rate", "epro_compliance_rate"]:
        assert col in subject_features.columns


def test_subject_features_no_nans_in_key_columns(subject_features):
    key_cols = ["ae_count", "pd_count", "query_count", "lab_oor_rate"]
    assert not subject_features[key_cols].isna().any().any()


def test_site_features_shape(site_features, tables):
    assert len(site_features) == len(tables["sites"])
    for col in ["pd_rate", "query_rate", "ae_rate", "dropout_rate", "major_pd_rate"]:
        assert col in site_features.columns


def test_site_features_rates_are_bounded(site_features):
    # rates should be non-negative; dropout/screen-failure are true proportions (<=1)
    assert (site_features["pd_rate"] >= 0).all()
    assert (site_features["dropout_rate"] <= 1).all()
    assert (site_features["screen_failure_rate"] <= 1).all()


def test_high_pd_site_has_elevated_pd_rate(site_features):
    """The generator seeds SITE-005 as a high-protocol-deviation site."""
    portfolio_mean = site_features["pd_rate"].mean()
    seeded_site_rate = site_features.loc[site_features["site_id"] == "SITE-005", "pd_rate"].iloc[0]
    assert seeded_site_rate > portfolio_mean
