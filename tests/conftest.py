import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from rbqm.data_generator import generate
from rbqm.db import get_engine, get_session, init_db
from rbqm.features import build_site_features, build_subject_features, load_tables


@pytest.fixture(scope="session")
def seeded_session():
    """In-memory SQLite DB, seeded once for the whole test session."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    session = get_session(engine)
    generate(session, seed=42)
    yield session
    session.close()


@pytest.fixture(scope="session")
def tables(seeded_session):
    return load_tables(seeded_session)


@pytest.fixture(scope="session")
def subject_features(tables):
    return build_subject_features(tables)


@pytest.fixture(scope="session")
def site_features(tables):
    return build_site_features(tables)
