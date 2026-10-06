"""SQLite-backed data model.

This stands in for the "curated zone" (Delta Lake / SDTM-shaped tables) in the
full AWS architecture. Locally, SQLite + SQLAlchemy gives us a single-file,
zero-install database that's trivial to reset and inspect (`sqlite3 data/rbqm.db`).
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from rbqm.config import DATA_DIR, DB_URL


class Base(DeclarativeBase):
    pass


class Site(Base):
    __tablename__ = "sites"

    site_id: Mapped[str] = mapped_column(String, primary_key=True)
    site_name: Mapped[str] = mapped_column(String)
    country: Mapped[str] = mapped_column(String)
    activation_date: Mapped[date] = mapped_column(Date)

    subjects: Mapped[list["Subject"]] = relationship(back_populates="site")


class Subject(Base):
    __tablename__ = "subjects"

    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"))
    arm: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)  # screened/active/completed/discontinued/screen_failed
    enrollment_date: Mapped[date] = mapped_column(Date)
    discontinuation_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    discontinuation_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    site: Mapped["Site"] = relationship(back_populates="subjects")


class Visit(Base):
    __tablename__ = "visits"

    visit_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    visit_name: Mapped[str] = mapped_column(String)
    expected_date: Mapped[date] = mapped_column(Date)
    actual_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)


class AdverseEvent(Base):
    __tablename__ = "adverse_events"

    ae_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"))
    onset_date: Mapped[date] = mapped_column(Date)
    reported_date: Mapped[date] = mapped_column(Date)
    verbatim_term: Mapped[str] = mapped_column(String)
    meddra_pt: Mapped[str] = mapped_column(String)
    soc: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)  # mild/moderate/severe
    serious: Mapped[bool] = mapped_column(Boolean)
    causality: Mapped[str] = mapped_column(String)  # related/not related/possibly related


class ConMed(Base):
    __tablename__ = "conmeds"

    conmed_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    drug_name: Mapped[str] = mapped_column(String)
    whodrug_class: Mapped[str] = mapped_column(String)
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    ongoing: Mapped[bool] = mapped_column(Boolean)


class Lab(Base):
    __tablename__ = "labs"

    lab_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    test_name: Mapped[str] = mapped_column(String)
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str] = mapped_column(String)
    low_range: Mapped[float] = mapped_column(Float)
    high_range: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String)  # central_lab | edc_transcribed
    collection_date: Mapped[date] = mapped_column(Date)


class ProtocolDeviation(Base):
    __tablename__ = "protocol_deviations"

    pd_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"))
    category: Mapped[str] = mapped_column(String)
    major: Mapped[bool] = mapped_column(Boolean)
    deviation_date: Mapped[date] = mapped_column(Date)


class Query(Base):
    __tablename__ = "queries"

    query_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    site_id: Mapped[str] = mapped_column(ForeignKey("sites.site_id"))
    domain: Mapped[str] = mapped_column(String)
    opened_date: Mapped[date] = mapped_column(Date)
    closed_date: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String)  # open | closed


class EPROEntry(Base):
    __tablename__ = "epro_entries"

    entry_id: Mapped[str] = mapped_column(String, primary_key=True)
    subject_id: Mapped[str] = mapped_column(ForeignKey("subjects.subject_id"))
    expected_date: Mapped[date] = mapped_column(Date)
    completed: Mapped[bool] = mapped_column(Boolean)


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

def get_engine(db_url: str = DB_URL):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(db_url, echo=False)


def init_db(engine=None) -> None:
    engine = engine or get_engine()
    Base.metadata.create_all(engine)


def get_session(engine=None) -> Session:
    engine = engine or get_engine()
    Session_ = sessionmaker(bind=engine)
    return Session_()
