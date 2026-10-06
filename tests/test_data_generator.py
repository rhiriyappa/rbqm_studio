from rbqm.config import N_SITES, N_SUBJECTS


def test_sites_generated(tables):
    assert len(tables["sites"]) == N_SITES


def test_subjects_generated(tables):
    # subjects_per_site * N_SITES (integer division), so allow for that
    expected = (N_SUBJECTS // N_SITES) * N_SITES
    assert len(tables["subjects"]) == expected


def test_subject_status_values_are_valid(tables):
    valid_statuses = {"active", "completed", "discontinued", "screen_failed"}
    assert set(tables["subjects"]["status"].unique()).issubset(valid_statuses)


def test_visits_follow_schedule(tables):
    visits_per_subject = tables["visits"].groupby("subject_id").size()
    assert (visits_per_subject == 7).all()  # 7 visits in VISIT_SCHEDULE


def test_labs_have_both_sources(tables):
    sources = set(tables["labs"]["source"].unique())
    assert sources == {"central_lab", "edc_transcribed"}


def test_seeded_interaction_case_exists(tables):
    """Verifies the explicit Warfarin + Bleeding interaction case was planted."""
    conmeds = tables["conmeds"]
    aes = tables["aes"]
    warfarin_subjects = set(conmeds[conmeds["drug_name"] == "Warfarin"]["subject_id"])
    bleeding_subjects = set(aes[aes["meddra_pt"] == "Bleeding"]["subject_id"])
    assert len(warfarin_subjects & bleeding_subjects) >= 1
