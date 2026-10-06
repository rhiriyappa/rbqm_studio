from rbqm.ml.ae_signal import detect_ae_clusters, flag_conmed_interactions


def test_ae_clusters_have_valid_rate_ratios(tables):
    clusters = detect_ae_clusters(tables["aes"], tables["subjects"])
    if not clusters.empty:
        assert (clusters["rate_ratio"] >= 2.0).all()
        assert clusters["signal"].all()


def test_conmed_interaction_flags_warfarin_bleeding_case(tables):
    """Verifies the explicitly seeded Warfarin + Bleeding case is caught."""
    flags = flag_conmed_interactions(tables["aes"], tables["conmeds"])
    assert not flags.empty
    assert (flags["drug_name"] == "Warfarin").any()
    assert (flags["meddra_pt"] == "Bleeding").any()


def test_conmed_interaction_flags_have_rationale(tables):
    flags = flag_conmed_interactions(tables["aes"], tables["conmeds"])
    if not flags.empty:
        assert (flags["rationale"].str.len() > 0).all()
