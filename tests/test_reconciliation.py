from rbqm.ml.reconciliation import FeedbackLog, find_lab_discrepancies


def test_find_lab_discrepancies_returns_suggestions(tables):
    suggestions = find_lab_discrepancies(tables["labs"])
    assert len(suggestions) > 0  # generator seeds ~7% discrepancy rate


def test_suggestions_have_confidence_and_rationale(tables):
    suggestions = find_lab_discrepancies(tables["labs"])
    for s in suggestions[:10]:
        assert 0 <= s.confidence_pct <= 100
        assert len(s.rationale) > 0
        assert s.source_a == "central_lab"
        assert s.source_b == "edc_transcribed"


def test_suggestions_sorted_by_confidence_desc(tables):
    suggestions = find_lab_discrepancies(tables["labs"])
    confidences = [s.confidence_pct for s in suggestions]
    assert confidences == sorted(confidences, reverse=True)


def test_feedback_log_records_human_in_the_loop_decisions():
    log = FeedbackLog()
    log.record("RECON-TEST01", user="dm_jsmith", decision="accept")
    log.record("RECON-TEST02", user="dm_jsmith", decision="reject", reason_code="source_doc_mismatch")
    df = log.as_dataframe()
    assert len(df) == 2
    assert set(df["decision"]) == {"accept", "reject"}
