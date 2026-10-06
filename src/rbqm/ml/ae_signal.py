"""Safety Surveillance Console analytics: AE clustering/disproportionality
across sites, and concomitant-medication interaction flagging.

`detect_ae_clusters` is a simplified disproportionality analysis (in the
spirit of a PRR/EBGM signal detection method used in pharmacovigilance):
for each MedDRA PT, compare a site's local rate against the portfolio rate
and flag statistically notable clusters.

`flag_conmed_interactions` is a rules-engine check against a small known
drug-class/AE interaction knowledge base -- in production this would call
out to a licensed drug-interaction database (e.g. Multum, Lexicomp) via API.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Minimal illustrative interaction knowledge base: (WHODrug class) -> set of
# MedDRA PTs that warrant a clinical review if co-occurring in the same subject.
INTERACTION_KB = {
    "Anticoagulants": {"Bleeding", "Hypotension"},
    "NSAIDs": {"Bleeding", "Hepatic enzyme increased"},
    "Antidiabetics": {"Hypotension", "Dizziness"},
}


def detect_ae_clusters(aes: pd.DataFrame, subjects: pd.DataFrame, min_site_events: int = 2) -> pd.DataFrame:
    """Flags (site, MedDRA PT) combinations reported at a materially higher
    rate than the rest of the portfolio."""
    if aes.empty:
        return pd.DataFrame(
            columns=["site_id", "meddra_pt", "soc", "site_events", "site_rate", "portfolio_rate", "rate_ratio", "signal"]
        )

    n_subjects_by_site = subjects.groupby("site_id").size().rename("n_subjects")
    total_subjects = n_subjects_by_site.sum()

    site_pt_counts = aes.groupby(["site_id", "meddra_pt", "soc"]).size().rename("site_events").reset_index()
    portfolio_pt_counts = aes.groupby("meddra_pt").size().rename("portfolio_events").reset_index()

    merged = site_pt_counts.merge(portfolio_pt_counts, on="meddra_pt")
    merged = merged.merge(n_subjects_by_site, on="site_id")

    merged["site_rate"] = merged["site_events"] / merged["n_subjects"]
    merged["portfolio_rate"] = (merged["portfolio_events"] - merged["site_events"]) / max(
        total_subjects - 1, 1
    )
    merged["portfolio_rate"] = merged["portfolio_rate"].replace(0, 1e-4)
    merged["rate_ratio"] = merged["site_rate"] / merged["portfolio_rate"]

    merged["signal"] = (merged["site_events"] >= min_site_events) & (merged["rate_ratio"] >= 2.0)

    result = merged[merged["signal"]].sort_values("rate_ratio", ascending=False)
    return result[
        ["site_id", "meddra_pt", "soc", "site_events", "site_rate", "portfolio_rate", "rate_ratio", "signal"]
    ]


def flag_conmed_interactions(aes: pd.DataFrame, conmeds: pd.DataFrame) -> pd.DataFrame:
    """Joins AEs to concurrent ConMeds for the same subject and flags any
    combination present in the interaction knowledge base."""
    if aes.empty or conmeds.empty:
        return pd.DataFrame(
            columns=["subject_id", "ae_id", "meddra_pt", "conmed_id", "drug_name", "whodrug_class", "rationale"]
        )

    merged = aes.merge(conmeds, on="subject_id", suffixes=("_ae", "_cm"))
    # keep only ConMeds that were ongoing at/around AE onset
    merged["start_date"] = pd.to_datetime(merged["start_date"])
    merged["onset_date"] = pd.to_datetime(merged["onset_date"])
    merged["end_date"] = pd.to_datetime(merged["end_date"])
    concurrent = merged[
        (merged["start_date"] <= merged["onset_date"])
        & (merged["end_date"].isna() | (merged["end_date"] >= merged["onset_date"]))
    ]

    flags = []
    for _, row in concurrent.iterrows():
        watch_terms = INTERACTION_KB.get(row["whodrug_class"], set())
        if row["meddra_pt"] in watch_terms:
            flags.append(
                {
                    "subject_id": row["subject_id"],
                    "ae_id": row["ae_id"],
                    "meddra_pt": row["meddra_pt"],
                    "conmed_id": row["conmed_id"],
                    "drug_name": row["drug_name"],
                    "whodrug_class": row["whodrug_class"],
                    "rationale": (
                        f"Subject reported '{row['meddra_pt']}' while on {row['drug_name']} "
                        f"({row['whodrug_class']}), a known class of interest for this event type."
                    ),
                }
            )
    return pd.DataFrame.from_records(flags)
