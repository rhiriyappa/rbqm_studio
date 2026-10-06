"""Quality Tolerance Limit (QTL) monitoring.

Evaluates each site (and the overall study) against the sponsor-defined QTL
boundaries in config.QTL_DEFINITIONS, returning a plain-language status:
"within_limit" / "approaching_limit" / "breach".
"""
from __future__ import annotations

import pandas as pd

from rbqm.config import QTL_DEFINITIONS


def _status_for_value(value: float, definition: dict) -> str:
    target = definition["target"]
    limit = definition["limit"]
    direction = definition["direction"]
    approach_band = definition["approach_band"]

    if direction == "above":
        if value >= limit:
            return "breach"
        if value >= limit * approach_band:
            return "approaching_limit"
        return "within_limit"
    else:  # "below" -- breach when value falls below the limit
        if value <= limit:
            return "breach"
        if value <= limit / approach_band:
            return "approaching_limit"
        return "within_limit"


def evaluate_qtls(site_features: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: one row per (site, QTL metric)."""
    if site_features.empty:
        return pd.DataFrame(
            columns=["site_id", "site_name", "metric", "label", "value", "target", "limit", "status"]
        )

    records = []
    for _, row in site_features.iterrows():
        for metric, definition in QTL_DEFINITIONS.items():
            if metric not in row or pd.isna(row[metric]):
                continue
            value = float(row[metric])
            status = _status_for_value(value, definition)
            records.append(
                {
                    "site_id": row["site_id"],
                    "site_name": row.get("site_name", row["site_id"]),
                    "metric": metric,
                    "label": definition["label"],
                    "value": round(value, 4),
                    "target": definition["target"],
                    "limit": definition["limit"],
                    "status": status,
                }
            )
    return pd.DataFrame.from_records(records)


def evaluate_study_qtls(site_features: pd.DataFrame) -> pd.DataFrame:
    """Study-level QTL rollup, weighted by subject count per site."""
    if site_features.empty:
        return pd.DataFrame(columns=["metric", "label", "value", "target", "limit", "status"])

    records = []
    total_subjects = site_features["n_subjects"].sum() or 1
    for metric, definition in QTL_DEFINITIONS.items():
        if metric not in site_features.columns:
            continue
        weighted_value = (site_features[metric] * site_features["n_subjects"]).sum() / total_subjects
        status = _status_for_value(weighted_value, definition)
        records.append(
            {
                "metric": metric,
                "label": definition["label"],
                "value": round(weighted_value, 4),
                "target": definition["target"],
                "limit": definition["limit"],
                "status": status,
            }
        )
    return pd.DataFrame.from_records(records)
