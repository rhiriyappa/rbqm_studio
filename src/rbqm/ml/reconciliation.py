"""AI Reconciliation Assistant.

Finds discrepancies between two data sources for the same subject/test/date
(e.g. central lab feed vs. EDC-transcribed value) and proposes a resolution
with a confidence score and a human-readable rationale. The Data Manager
always makes the final call (`accept` / `edit` / `reject`); this module never
writes back to the database on its own -- see `record_feedback` for the
human-in-the-loop logging that would drive periodic model retraining.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

MATERIAL_DIFF_PCT = 0.15  # flag if values differ by more than 15%


@dataclass
class ReconciliationSuggestion:
    case_id: str
    subject_id: str
    test_name: str
    collection_date: str
    source_a: str
    value_a: float
    source_b: str
    value_b: float
    suggested_resolution: str
    confidence_pct: float
    rationale: str


def _guess_explanation(value_a: float, value_b: float) -> tuple[str, float]:
    """Rule-based hypothesis generation -- stands in for the LLM+retrieval
    reconciliation assistant described in the architecture doc. In production
    this call would go to an LLM with retrieval over historically-resolved
    cases (pgvector) instead of hardcoded ratio checks.
    """
    if value_a == 0 or value_b == 0:
        return "One value is zero -- likely a missing/placeholder entry. Recommend source verification.", 55.0

    ratio = value_b / value_a
    if abs(ratio - 10) < 0.5:
        return "EDC value appears 10x the central lab value -- likely a decimal/unit entry error in EDC.", 91.0
    if abs(ratio - 0.1) < 0.05:
        return "EDC value appears 1/10th the central lab value -- likely a decimal/unit entry error in EDC.", 91.0
    if abs(ratio - 2.2046) < 0.15:
        return "Ratio consistent with a kg-to-lb unit mismatch. Recommend confirming unit of record.", 78.0
    pct_diff = abs(value_a - value_b) / max(abs(value_a), 1e-9)
    if pct_diff > 0.5:
        return "Large unexplained divergence between sources -- recommend transcription re-check against source document.", 40.0
    return "Values differ within a plausible transcription-error range -- recommend confirming against source document.", 60.0


def find_lab_discrepancies(labs: pd.DataFrame, subject_id: Optional[str] = None) -> list[ReconciliationSuggestion]:
    """Compares central_lab vs edc_transcribed values for the same
    subject/test/collection_date pair and proposes resolutions."""
    if labs.empty:
        return []

    df = labs.copy()
    if subject_id:
        df = df[df["subject_id"] == subject_id]

    pivot_key = ["subject_id", "test_name", "collection_date"]
    central = df[df["source"] == "central_lab"][pivot_key + ["value", "unit"]].rename(
        columns={"value": "value_central", "unit": "unit_central"}
    )
    edc = df[df["source"] == "edc_transcribed"][pivot_key + ["value", "unit"]].rename(
        columns={"value": "value_edc", "unit": "unit_edc"}
    )
    merged = central.merge(edc, on=pivot_key, how="inner")
    if merged.empty:
        return []

    merged["pct_diff"] = (merged["value_edc"] - merged["value_central"]).abs() / merged["value_central"].abs().replace(0, 1e-9)
    discrepant = merged[merged["pct_diff"] > MATERIAL_DIFF_PCT]

    suggestions: list[ReconciliationSuggestion] = []
    for _, row in discrepant.iterrows():
        rationale, confidence = _guess_explanation(row["value_central"], row["value_edc"])
        suggestions.append(
            ReconciliationSuggestion(
                case_id=f"RECON-{uuid.uuid4().hex[:8].upper()}",
                subject_id=row["subject_id"],
                test_name=row["test_name"],
                collection_date=str(row["collection_date"]),
                source_a="central_lab",
                value_a=row["value_central"],
                source_b="edc_transcribed",
                value_b=row["value_edc"],
                suggested_resolution=f"Use central_lab value ({row['value_central']}) as source of truth pending site confirmation.",
                confidence_pct=confidence,
                rationale=rationale,
            )
        )
    return sorted(suggestions, key=lambda s: -s.confidence_pct)


@dataclass
class FeedbackLog:
    """In-memory feedback store for the local demo. In production this is an
    audit-trailed table (21 CFR Part 11 e-signature on every decision) that
    feeds the scheduled retraining pipeline described in the design doc.
    """
    entries: list[dict] = field(default_factory=list)

    def record(self, case_id: str, user: str, decision: str, reason_code: Optional[str] = None) -> dict:
        assert decision in {"accept", "edit", "reject"}
        entry = {
            "case_id": case_id,
            "user": user,
            "decision": decision,
            "reason_code": reason_code,
        }
        self.entries.append(entry)
        return entry

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.entries)
