"""RBQM Studio -- local interactive dashboard.

Run with:
    streamlit run src/rbqm/dashboard/app.py

This talks directly to the SQLite DB + ML modules (not through the API) to
keep the local demo to a single process. The FastAPI service in rbqm.api is
the same logic exposed as REST for integration with other tools.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running via `streamlit run` without installing the package
SRC_DIR = Path(__file__).resolve().parents[2]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import plotly.express as px
import streamlit as st

from rbqm.db import get_engine, get_session
from rbqm.features import build_site_features, build_subject_features, load_tables
from rbqm.ml.ae_signal import detect_ae_clusters, flag_conmed_interactions
from rbqm.ml.anomaly_detection import detect_subject_anomalies
from rbqm.ml.kri_engine import compute_site_kris, predict_premature_termination_rate
from rbqm.ml.qtl_engine import evaluate_qtls, evaluate_study_qtls
from rbqm.ml.reconciliation import find_lab_discrepancies

st.set_page_config(page_title="RBQM Studio", layout="wide", page_icon="🧬")

TIER_COLORS = {"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"}


@st.cache_data(show_spinner=False)
def load_all():
    engine = get_engine()
    session = get_session(engine)
    tables = load_tables(session)
    session.close()
    subject_features = build_subject_features(tables)
    site_features = build_site_features(tables)
    return tables, subject_features, site_features


def risk_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#999")
    return f"<span style='background-color:{color};color:white;padding:2px 10px;border-radius:10px;font-size:0.85em'>{tier}</span>"


tables, subject_features, site_features = load_all()
kris = compute_site_kris(site_features)

st.sidebar.title("🧬 RBQM Studio")
st.sidebar.caption("Local reference implementation")
role = st.sidebar.radio(
    "View as",
    ["Central Monitor / CPM — KRI Dashboard", "Data Manager Studio", "CRA Site Workbench", "Medical Monitor — Safety Console", "QTL Console"],
)
st.sidebar.markdown("---")
st.sidebar.metric("Sites", len(site_features))
st.sidebar.metric("Subjects", len(subject_features))
st.sidebar.markdown("---")
st.sidebar.caption("Data resets via `python -m rbqm.data_generator`")

# ---------------------------------------------------------------------------
if role.startswith("Central Monitor"):
    st.title("📊 KRI Dashboard")
    st.caption("Real-time site risk tiering and trial trajectory forecast — no statistics background required.")

    c1, c2, c3, c4 = st.columns(4)
    tier_counts = kris["risk_tier"].value_counts()
    c1.metric("🟢 Low risk sites", int(tier_counts.get("Low", 0)))
    c2.metric("🟡 Medium risk sites", int(tier_counts.get("Medium", 0)))
    c3.metric("🔴 High risk sites", int(tier_counts.get("High", 0)))
    forecast = predict_premature_termination_rate(site_features)
    c4.metric(
        "Forecast premature termination",
        f"{forecast['forecast_rate']*100:.1f}%",
        delta=f"vs. {forecast['protocol_assumed_rate']*100:.0f}% assumed",
        delta_color="inverse",
    )
    if forecast["status"] != "on_track":
        st.warning(
            f"⚠️ Trial trajectory status: **{forecast['status'].replace('_', ' ').title()}** — "
            f"observed dropout rate ({forecast['observed_rate']*100:.1f}%) is trending above the protocol assumption."
        )

    st.subheader("Site Risk Heat Map")
    fig = px.bar(
        kris,
        x="site_id",
        y="composite_risk_score",
        color="risk_tier",
        color_discrete_map=TIER_COLORS,
        hover_data=["site_name", "top_drivers", "n_subjects"],
        labels={"composite_risk_score": "Composite Risk Score", "site_id": "Site"},
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Site Detail")
    display_cols = ["site_id", "site_name", "n_subjects", "risk_tier", "top_drivers", "pd_rate", "query_rate", "sae_late_reporting_rate", "dropout_rate"]
    st.dataframe(kris[display_cols], use_container_width=True, hide_index=True)

    st.subheader("Statistical Outliers (Central Statistical Monitoring)")
    outliers = detect_ae_clusters(tables["aes"], tables["subjects"])
    if outliers.empty:
        st.info("No cross-site AE reporting-rate outliers detected in current data.")
    else:
        st.dataframe(outliers, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
elif role == "Data Manager Studio":
    st.title("🗂️ Data Manager Studio")
    st.caption("Unified review workspace: AI-ranked anomalies, reconciliation assistant, and visual data profiles.")

    tab1, tab2, tab3 = st.tabs(["AI Anomaly Worklist", "AI Reconciliation Assistant", "Visual Data Profiles"])

    with tab1:
        st.markdown("Subjects ranked by cross-domain anomaly score (Isolation Forest across AE/PD/query/lab/ePRO features).")
        anomalies = detect_subject_anomalies(subject_features, top_n=25)
        st.dataframe(
            anomalies[["subject_id", "site_id", "status", "confidence_pct", "explanation"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        st.markdown("Central lab vs. EDC-transcribed discrepancies with AI-suggested resolution. **Data Manager makes the final call.**")
        suggestions = find_lab_discrepancies(tables["labs"])
        if not suggestions:
            st.info("No material discrepancies detected.")
        else:
            for s in suggestions[:15]:
                with st.expander(f"{s.case_id} — {s.subject_id} / {s.test_name} (confidence {s.confidence_pct:.0f}%)"):
                    col1, col2 = st.columns(2)
                    col1.metric("Central Lab", s.value_a)
                    col2.metric("EDC Transcribed", s.value_b)
                    st.write(f"**AI rationale:** {s.rationale}")
                    st.write(f"**Suggested resolution:** {s.suggested_resolution}")
                    b1, b2, b3 = st.columns(3)
                    b1.button("✅ Accept", key=f"accept_{s.case_id}")
                    b2.button("✏️ Edit", key=f"edit_{s.case_id}")
                    b3.button("❌ Reject", key=f"reject_{s.case_id}")

    with tab3:
        domain = st.selectbox("Domain", ["aes", "pds", "labs", "queries", "conmeds", "epro"])
        df = tables[domain]
        st.write(f"**{len(df)} records** in `{domain}`")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            col = st.selectbox("Profile column", numeric_cols)
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)
        missing = df.isna().mean().sort_values(ascending=False)
        st.bar_chart(missing[missing > 0])

# ---------------------------------------------------------------------------
elif role == "CRA Site Workbench":
    st.title("📍 CRA Site Workbench")
    site_id = st.selectbox("Select site", sorted(site_features["site_id"].unique()))
    row = kris[kris["site_id"] == site_id].iloc[0]

    st.markdown(f"### {row['site_name']} ({site_id})")
    st.markdown(risk_badge(row["risk_tier"]), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Subjects", int(row["n_subjects"]))
    c2.metric("Protocol Deviation Rate", f"{row['pd_rate']:.2f}")
    c3.metric("Query Rate", f"{row['query_rate']:.2f}")

    st.subheader("Open Queries by Domain")
    site_queries = tables["queries"]
    open_q = site_queries[(site_queries["site_id"] == site_id) & (site_queries["status"] == "open")]
    if open_q.empty:
        st.info("No open queries.")
    else:
        st.bar_chart(open_q.groupby("domain").size())

    st.subheader("Protocol Deviations")
    site_pds = tables["pds"][tables["pds"]["site_id"] == site_id]
    st.dataframe(site_pds[["pd_id", "subject_id", "category", "major", "deviation_date"]], use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
elif role == "Medical Monitor — Safety Console":
    st.title("🩺 Safety Surveillance Console")

    tab1, tab2, tab3 = st.tabs(["AE Signal Clusters", "ConMed Interaction Flags", "SAE Reporting Timeliness"])

    with tab1:
        st.markdown("Site-level AE reporting rate vs. portfolio peer rate (disproportionality-style signal detection).")
        clusters = detect_ae_clusters(tables["aes"], tables["subjects"])
        if clusters.empty:
            st.info("No AE clusters flagged.")
        else:
            st.dataframe(clusters, use_container_width=True, hide_index=True)
            fig = px.bar(clusters, x="meddra_pt", y="rate_ratio", color="site_id", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("Adverse events that co-occur with a concomitant medication class of clinical interest.")
        flags = flag_conmed_interactions(tables["aes"], tables["conmeds"])
        if flags.empty:
            st.info("No interaction flags in current data.")
        else:
            st.dataframe(flags, use_container_width=True, hide_index=True)

    with tab3:
        aes = tables["aes"]
        sae = aes[aes["serious"]].copy()
        sae["report_lag_days"] = (pd.to_datetime(sae["reported_date"]) - pd.to_datetime(sae["onset_date"])).dt.days
        sae["clock_status"] = sae["report_lag_days"].apply(lambda d: "🔴 breach" if d > 15 else ("🟡 at risk" if d > 12 else "🟢 on time"))
        st.dataframe(
            sae.sort_values("report_lag_days", ascending=False)[
                ["ae_id", "subject_id", "site_id", "meddra_pt", "report_lag_days", "clock_status"]
            ],
            use_container_width=True,
            hide_index=True,
        )

# ---------------------------------------------------------------------------
elif role == "QTL Console":
    st.title("🎯 Quality Tolerance Limits")
    study_qtls = evaluate_study_qtls(site_features)
    st.subheader("Study-Level QTLs")
    for _, r in study_qtls.iterrows():
        status_emoji = {"within_limit": "🟢", "approaching_limit": "🟡", "breach": "🔴"}[r["status"]]
        st.markdown(f"**{status_emoji} {r['label']}** — current: `{r['value']}`  |  target: `{r['target']}`  |  limit: `{r['limit']}`  |  status: **{r['status'].replace('_', ' ')}**")

    st.subheader("Site-Level QTL Status")
    site_qtls = evaluate_qtls(site_features)
    st.dataframe(site_qtls, use_container_width=True, hide_index=True)
