"""Feature engineering shared by every ML module.

Everything downstream (anomaly detection, KRI engine, QTL engine, AE signal
detection) reads from these two tables rather than querying SQL directly, so
the "canonical Subject / Site feature model" described in the architecture
doc has one implementation.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from rbqm.db import AdverseEvent, ConMed, EPROEntry, Lab, ProtocolDeviation, Query, Site, Subject, Visit


def _to_df(session: Session, model) -> pd.DataFrame:
    rows = session.execute(select(model)).scalars().all()
    records = [
        {c.name: getattr(row, c.name) for c in model.__table__.columns} for row in rows
    ]
    return pd.DataFrame.from_records(records)


def load_tables(session: Session) -> dict[str, pd.DataFrame]:
    return {
        "sites": _to_df(session, Site),
        "subjects": _to_df(session, Subject),
        "visits": _to_df(session, Visit),
        "aes": _to_df(session, AdverseEvent),
        "conmeds": _to_df(session, ConMed),
        "labs": _to_df(session, Lab),
        "pds": _to_df(session, ProtocolDeviation),
        "queries": _to_df(session, Query),
        "epro": _to_df(session, EPROEntry),
    }


def build_subject_features(tables: dict[str, pd.DataFrame], as_of: Optional[date] = None) -> pd.DataFrame:
    """One row per subject with counts/rates used by anomaly detection."""
    subjects = tables["subjects"].copy()
    if subjects.empty:
        return subjects

    ae_counts = tables["aes"].groupby("subject_id").size().rename("ae_count")
    sae_counts = (
        tables["aes"][tables["aes"]["serious"]].groupby("subject_id").size().rename("sae_count")
    )
    pd_counts = tables["pds"].groupby("subject_id").size().rename("pd_count")
    major_pd_counts = (
        tables["pds"][tables["pds"]["major"]].groupby("subject_id").size().rename("major_pd_count")
    )
    query_counts = tables["queries"].groupby("subject_id").size().rename("query_count")
    open_query_counts = (
        tables["queries"][tables["queries"]["status"] == "open"]
        .groupby("subject_id")
        .size()
        .rename("open_query_count")
    )
    conmed_counts = tables["conmeds"].groupby("subject_id").size().rename("conmed_count")

    labs = tables["labs"]
    lab_out_of_range = labs.assign(
        out_of_range=lambda d: (d["value"] < d["low_range"]) | (d["value"] > d["high_range"])
    )
    lab_oor_counts = lab_out_of_range.groupby("subject_id")["out_of_range"].sum().rename("lab_oor_count")
    lab_total_counts = lab_out_of_range.groupby("subject_id").size().rename("lab_total_count")

    epro = tables["epro"]
    epro_rate = epro.groupby("subject_id")["completed"].mean().rename("epro_compliance_rate")

    df = subjects.set_index("subject_id")
    for series in [
        ae_counts,
        sae_counts,
        pd_counts,
        major_pd_counts,
        query_counts,
        open_query_counts,
        conmed_counts,
        lab_oor_counts,
        lab_total_counts,
        epro_rate,
    ]:
        df = df.join(series)

    df = df.fillna(0)
    df["lab_oor_rate"] = np.where(df["lab_total_count"] > 0, df["lab_oor_count"] / df["lab_total_count"], 0.0)
    df["epro_compliance_rate"] = df["epro_compliance_rate"].fillna(0.0)
    df = df.reset_index()
    return df


def build_site_features(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per site with the rates the KRI/QTL engines operate on."""
    subjects = tables["subjects"].copy()
    if subjects.empty:
        return subjects

    n_subjects = subjects.groupby("site_id").size().rename("n_subjects")
    n_discontinued = (
        subjects[subjects["status"] == "discontinued"].groupby("site_id").size().rename("n_discontinued")
    )
    n_screen_failed = (
        subjects[subjects["status"] == "screen_failed"].groupby("site_id").size().rename("n_screen_failed")
    )

    pds = tables["pds"]
    pd_count = pds.groupby("site_id").size().rename("pd_count")
    major_pd_count = pds[pds["major"]].groupby("site_id").size().rename("major_pd_count")

    queries = tables["queries"]
    query_count = queries.groupby("site_id").size().rename("query_count")
    open_query_count = (
        queries[queries["status"] == "open"].groupby("site_id").size().rename("open_query_count")
    )
    closed = queries[queries["status"] == "closed"].copy()
    closed["resolution_days"] = (
        pd.to_datetime(closed["closed_date"]) - pd.to_datetime(closed["opened_date"])
    ).dt.days
    query_aging = closed.groupby("site_id")["resolution_days"].median().rename("query_aging_days")

    aes = tables["aes"]
    ae_count = aes.groupby("site_id").size().rename("ae_count")
    aes_with_lag = aes.copy()
    aes_with_lag["report_lag_days"] = (
        pd.to_datetime(aes_with_lag["reported_date"]) - pd.to_datetime(aes_with_lag["onset_date"])
    ).dt.days
    sae = aes_with_lag[aes_with_lag["serious"]]
    sae_count = sae.groupby("site_id").size().rename("sae_count")
    sae_late_count = (
        sae[sae["report_lag_days"] > 15].groupby("site_id").size().rename("sae_late_count")
    )

    epro = tables["epro"].merge(subjects[["subject_id", "site_id"]], on="subject_id", how="left")
    epro_rate = epro.groupby("site_id")["completed"].mean().rename("epro_compliance_rate")

    df = tables["sites"].set_index("site_id")
    for series in [
        n_subjects,
        n_discontinued,
        n_screen_failed,
        pd_count,
        major_pd_count,
        query_count,
        open_query_count,
        query_aging,
        ae_count,
        sae_count,
        sae_late_count,
        epro_rate,
    ]:
        df = df.join(series)
    df = df.fillna(0)
    df = df.reset_index()

    df["n_subjects"] = df["n_subjects"].replace(0, np.nan)
    df["pd_rate"] = df["pd_count"] / df["n_subjects"]
    df["major_pd_rate"] = df["major_pd_count"] / df["n_subjects"]
    df["query_rate"] = df["query_count"] / df["n_subjects"]
    df["ae_rate"] = df["ae_count"] / df["n_subjects"]
    df["dropout_rate"] = df["n_discontinued"] / df["n_subjects"]
    df["screen_failure_rate"] = df["n_screen_failed"] / df["n_subjects"]
    df["sae_late_reporting_rate"] = np.where(df["sae_count"] > 0, df["sae_late_count"] / df["sae_count"], 0.0)
    df["query_aging_days"] = df["query_aging_days"].fillna(df["query_aging_days"].median())
    df["n_subjects"] = df["n_subjects"].fillna(0)

    # AE under/over-reporting z-score vs. portfolio peer mean (used by CSM)
    portfolio_mean = df["ae_rate"].mean()
    portfolio_std = df["ae_rate"].std(ddof=0) or 1.0
    df["ae_underreporting_z"] = (df["ae_rate"] - portfolio_mean) / portfolio_std

    return df
