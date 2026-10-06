"""Generates a synthetic multi-site clinical trial dataset.

The generator deliberately seeds a handful of known anomalies/signals so that
the ML modules (anomaly detection, KRI engine, QTL engine, AE clustering,
reconciliation) have real, verifiable things to catch. This makes the test
suite meaningful: we assert the engines actually recover the planted signals.

Run: `python -m rbqm.data_generator` (or via scripts/seed_db.py)
"""
from __future__ import annotations

import random
import uuid
from datetime import date, timedelta

import numpy as np
from faker import Faker
from sqlalchemy.orm import Session

from rbqm.config import N_SITES, N_SUBJECTS, RANDOM_SEED
from rbqm.db import (
    AdverseEvent,
    ConMed,
    EPROEntry,
    Lab,
    ProtocolDeviation,
    Query,
    Site,
    Subject,
    Visit,
    get_engine,
    get_session,
    init_db,
)

fake = Faker()

MEDDRA_TERMS = [
    ("Headache", "Nervous system disorders"),
    ("Nausea", "Gastrointestinal disorders"),
    ("Fatigue", "General disorders"),
    ("ALT increased", "Hepatobiliary disorders"),
    ("AST increased", "Hepatobiliary disorders"),
    ("Hepatic enzyme increased", "Hepatobiliary disorders"),
    ("Dizziness", "Nervous system disorders"),
    ("Rash", "Skin and subcutaneous tissue disorders"),
    ("Diarrhoea", "Gastrointestinal disorders"),
    ("Insomnia", "Psychiatric disorders"),
    ("Bleeding", "Vascular disorders"),
    ("Hypotension", "Vascular disorders"),
    ("Injection site reaction", "General disorders"),
]

CONMED_DRUGS = [
    ("Warfarin", "Anticoagulants"),
    ("Aspirin", "Antiplatelet agents"),
    ("Metformin", "Antidiabetics"),
    ("Atorvastatin", "Lipid modifying agents"),
    ("Omeprazole", "Antacids"),
    ("Ibuprofen", "NSAIDs"),
    ("Lisinopril", "ACE inhibitors"),
]

PD_CATEGORIES = [
    "Informed consent timing",
    "Visit window violation",
    "Eligibility criteria not met",
    "Prohibited concomitant medication",
    "Dosing error",
    "Missed assessment",
]

LAB_TESTS = {
    "ALT": (7, 56, "U/L"),
    "AST": (8, 48, "U/L"),
    "Creatinine": (0.6, 1.3, "mg/dL"),
    "Hemoglobin": (12.0, 17.5, "g/dL"),
    "Glucose": (70, 110, "mg/dL"),
    "Platelets": (150, 450, "10^3/uL"),
}

VISIT_SCHEDULE = ["Screening", "Baseline", "Week 4", "Week 8", "Week 12", "Week 24", "End of Study"]


def _rand_date(start: date, end: date) -> date:
    delta = (end - start).days
    if delta <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta))


def generate(session: Session, seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    Faker.seed(seed)

    study_start = date(2024, 1, 15)
    today = date(2026, 7, 1)

    sites = []
    for i in range(1, N_SITES + 1):
        site_id = f"SITE-{i:03d}"
        site = Site(
            site_id=site_id,
            site_name=f"{fake.city()} Clinical Research Center",
            country=random.choice(["USA", "USA", "USA", "Germany", "Poland", "India", "Brazil"]),
            activation_date=_rand_date(study_start, study_start + timedelta(days=90)),
        )
        session.add(site)
        sites.append(site)

    # --- Seeded site archetypes (this is what the ML layer should detect) ---
    # SITE-002: AE under-reporter (statistical outlier vs peers)
    # SITE-005: High protocol-deviation / high-risk operational site
    # SITE-009: High dropout / premature termination site
    # SITE-011: SAE late-reporting site (QTL breach candidate)
    underreporting_site = "SITE-002"
    high_pd_site = "SITE-005"
    high_dropout_site = "SITE-009"
    late_sae_site = "SITE-011"

    subjects = []
    subjects_per_site = N_SUBJECTS // N_SITES
    for site in sites:
        for _ in range(subjects_per_site):
            subject_id = f"{site.site_id}-{uuid.uuid4().hex[:6].upper()}"
            enrollment_date = _rand_date(site.activation_date, today - timedelta(days=30))

            dropout_prob = 0.28 if site.site_id == high_dropout_site else 0.09
            status_roll = random.random()
            if status_roll < 0.04:
                status, disc_date, disc_reason = "screen_failed", None, None
            elif status_roll < 0.04 + dropout_prob:
                status = "discontinued"
                disc_date = _rand_date(enrollment_date, today)
                disc_reason = random.choice(
                    ["Adverse event", "Withdrew consent", "Lost to follow-up", "Physician decision", "Lack of efficacy"]
                )
            else:
                status, disc_date, disc_reason = random.choice(["active", "active", "completed"]), None, None

            subj = Subject(
                subject_id=subject_id,
                site_id=site.site_id,
                arm=random.choice(["Treatment A", "Treatment B", "Placebo"]),
                status=status,
                enrollment_date=enrollment_date,
                discontinuation_date=disc_date,
                discontinuation_reason=disc_reason,
            )
            session.add(subj)
            subjects.append(subj)

    session.flush()

    # --- Visits ---
    for subj in subjects:
        cursor = subj.enrollment_date
        for visit_name in VISIT_SCHEDULE:
            expected = cursor
            # small % of visits missed / delayed
            miss_roll = random.random()
            actual = None
            if miss_roll > 0.08 and expected <= today:
                jitter = random.randint(-2, 6)
                actual = expected + timedelta(days=jitter)
            session.add(
                Visit(
                    visit_id=f"VIS-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    visit_name=visit_name,
                    expected_date=expected,
                    actual_date=actual,
                )
            )
            cursor = cursor + timedelta(days=28)

    # --- Adverse events (with under-reporting seeded at one site) ---
    for subj in subjects:
        base_rate = 1.4
        if subj.site_id == underreporting_site:
            base_rate = 0.3  # materially fewer AEs reported than peers -> CSM signal
        n_aes = np.random.poisson(base_rate)
        for _ in range(n_aes):
            term, soc = random.choice(MEDDRA_TERMS)
            onset = _rand_date(subj.enrollment_date, min(today, subj.enrollment_date + timedelta(days=300)))
            serious = random.random() < 0.08
            # seed a late-SAE-reporting pattern at one site
            if serious and subj.site_id == late_sae_site:
                report_lag = random.randint(16, 30)
            elif serious:
                report_lag = random.randint(0, 14)
            else:
                report_lag = random.randint(0, 5)
            session.add(
                AdverseEvent(
                    ae_id=f"AE-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    site_id=subj.site_id,
                    onset_date=onset,
                    reported_date=onset + timedelta(days=report_lag),
                    verbatim_term=term,
                    meddra_pt=term,
                    soc=soc,
                    severity=random.choice(["mild", "moderate", "moderate", "severe"]),
                    serious=serious,
                    causality=random.choice(["related", "possibly related", "not related", "not related"]),
                )
            )

    # --- Concomitant meds (seed a Warfarin + Bleeding interaction case) ---
    for subj in subjects:
        n_conmeds = np.random.poisson(1.1)
        for _ in range(n_conmeds):
            drug, whodrug_class = random.choice(CONMED_DRUGS)
            start = _rand_date(subj.enrollment_date, today)
            ongoing = random.random() < 0.6
            session.add(
                ConMed(
                    conmed_id=f"CM-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    drug_name=drug,
                    whodrug_class=whodrug_class,
                    start_date=start,
                    end_date=None if ongoing else start + timedelta(days=random.randint(5, 90)),
                    ongoing=ongoing,
                )
            )

    # Explicit seeded interaction case: Warfarin + Bleeding AE, same subject
    interaction_subject = subjects[len(subjects) // 3]
    session.add(
        ConMed(
            conmed_id=f"CM-{uuid.uuid4().hex[:10]}",
            subject_id=interaction_subject.subject_id,
            drug_name="Warfarin",
            whodrug_class="Anticoagulants",
            start_date=interaction_subject.enrollment_date + timedelta(days=10),
            end_date=None,
            ongoing=True,
        )
    )
    session.add(
        AdverseEvent(
            ae_id=f"AE-{uuid.uuid4().hex[:10]}",
            subject_id=interaction_subject.subject_id,
            site_id=interaction_subject.site_id,
            onset_date=interaction_subject.enrollment_date + timedelta(days=45),
            reported_date=interaction_subject.enrollment_date + timedelta(days=46),
            verbatim_term="Bleeding",
            meddra_pt="Bleeding",
            soc="Vascular disorders",
            severity="moderate",
            serious=True,
            causality="possibly related",
        )
    )

    # --- Protocol deviations (seed a high-PD site) ---
    for subj in subjects:
        base_rate = 1.9 if subj.site_id == high_pd_site else 0.35
        n_pd = np.random.poisson(base_rate)
        for _ in range(n_pd):
            category = (
                "Informed consent timing"
                if subj.site_id == high_pd_site and random.random() < 0.5
                else random.choice(PD_CATEGORIES)
            )
            session.add(
                ProtocolDeviation(
                    pd_id=f"PD-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    site_id=subj.site_id,
                    category=category,
                    major=random.random() < (0.55 if subj.site_id == high_pd_site else 0.25),
                    deviation_date=_rand_date(subj.enrollment_date, today),
                )
            )

    # --- Queries ---
    for subj in subjects:
        n_queries = np.random.poisson(2.2)
        slow_site = subj.site_id == high_pd_site
        for _ in range(n_queries):
            opened = _rand_date(subj.enrollment_date, today)
            is_open = random.random() < (0.35 if slow_site else 0.12)
            closed = None if is_open else opened + timedelta(days=int(np.random.exponential(9 if not slow_site else 22)))
            session.add(
                Query(
                    query_id=f"QRY-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    site_id=subj.site_id,
                    domain=random.choice(["Labs", "AE", "ConMed", "Vitals", "Demographics"]),
                    opened_date=opened,
                    closed_date=closed,
                    status="open" if is_open else "closed",
                )
            )

    # --- Labs (seed EDC vs central lab discrepancies for reconciliation demo) ---
    for subj in subjects:
        for test_name, (low, high, unit) in LAB_TESTS.items():
            for _ in range(random.randint(1, 3)):
                collection = _rand_date(subj.enrollment_date, today)
                central_value = round(np.random.normal((low + high) / 2, (high - low) / 4), 2)
                session.add(
                    Lab(
                        lab_id=f"LAB-{uuid.uuid4().hex[:10]}",
                        subject_id=subj.subject_id,
                        test_name=test_name,
                        value=max(central_value, 0.1),
                        unit=unit,
                        low_range=low,
                        high_range=high,
                        source="central_lab",
                        collection_date=collection,
                    )
                )
                # ~7% chance the EDC-transcribed value diverges materially -> reconciliation target
                edc_value = central_value
                if random.random() < 0.07:
                    edc_value = round(central_value * random.choice([1.0, 10.0, 0.1]) + random.uniform(-2, 2), 2)
                session.add(
                    Lab(
                        lab_id=f"LAB-{uuid.uuid4().hex[:10]}",
                        subject_id=subj.subject_id,
                        test_name=test_name,
                        value=max(edc_value, 0.1),
                        unit=unit,
                        low_range=low,
                        high_range=high,
                        source="edc_transcribed",
                        collection_date=collection,
                    )
                )

    # --- ePRO compliance (drop it hard for the high-dropout site) ---
    for subj in subjects:
        low_compliance = subj.site_id == high_dropout_site
        cursor = subj.enrollment_date
        while cursor <= today:
            completed_prob = 0.55 if low_compliance else 0.92
            session.add(
                EPROEntry(
                    entry_id=f"EPRO-{uuid.uuid4().hex[:10]}",
                    subject_id=subj.subject_id,
                    expected_date=cursor,
                    completed=random.random() < completed_prob,
                )
            )
            cursor += timedelta(days=7)

    session.commit()


def main() -> None:
    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    generate(session)
    session.close()
    print("Synthetic RBQM dataset generated at data/rbqm.db")


if __name__ == "__main__":
    main()
