from rbqm.ml.anomaly_detection import FEATURE_COLUMNS, detect_subject_anomalies


def test_anomaly_worklist_returns_ranked_subjects(subject_features):
    result = detect_subject_anomalies(subject_features, top_n=15)
    assert len(result) == 15
    # ranked descending by anomaly_score
    scores = result["anomaly_score"].tolist()
    assert scores == sorted(scores, reverse=True)


def test_anomaly_worklist_has_explanations(subject_features):
    result = detect_subject_anomalies(subject_features, top_n=10)
    assert (result["explanation"].str.startswith("Flagged because")).all()


def test_anomaly_worklist_includes_feature_columns(subject_features):
    result = detect_subject_anomalies(subject_features, top_n=5)
    for col in FEATURE_COLUMNS:
        assert col in result.columns


def test_top_n_respects_available_subjects(subject_features):
    result = detect_subject_anomalies(subject_features, top_n=1000)
    assert len(result) == len(subject_features)
