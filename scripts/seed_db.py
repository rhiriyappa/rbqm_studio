#!/usr/bin/env python
"""Resets and (re)generates the local SQLite dataset.

Usage: python scripts/seed_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rbqm.config import DB_PATH
from rbqm.data_generator import generate
from rbqm.db import get_engine, get_session, init_db

if __name__ == "__main__":
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"Removed existing DB at {DB_PATH}")

    engine = get_engine()
    init_db(engine)
    session = get_session(engine)
    generate(session)
    session.close()
    print(f"✅ Seeded synthetic clinical trial dataset at {DB_PATH}")
