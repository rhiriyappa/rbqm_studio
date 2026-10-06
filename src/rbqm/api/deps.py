from __future__ import annotations

from functools import lru_cache

from rbqm.db import get_engine, get_session
from rbqm.features import build_site_features, build_subject_features, load_tables


def fetch_all_tables():
    """Loads all tables fresh from SQLite. Dataset is small (local demo), so
    no caching is needed for correctness; a production deployment would read
    from the curated Delta Lake zone / feature store instead.
    """
    engine = get_engine()
    session = get_session(engine)
    try:
        tables = load_tables(session)
    finally:
        session.close()
    return tables


def get_feature_tables():
    tables = fetch_all_tables()
    subject_features = build_subject_features(tables)
    site_features = build_site_features(tables)
    return tables, subject_features, site_features
