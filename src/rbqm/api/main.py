"""RBQM Studio API.

Run locally with:
    uvicorn rbqm.api.main:app --reload --port 8000

Then browse the interactive docs at http://127.0.0.1:8000/docs
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rbqm.api.routers import central_monitor, cpm, cra, dm, medical_monitor

app = FastAPI(
    title="RBQM Studio API",
    description=(
        "Clinical Data & Risk Surveillance Platform — role-based endpoints for "
        "Data Managers, CRAs, Central Monitors, Medical Monitors, and CPMs."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dm.router)
app.include_router(cra.router)
app.include_router(central_monitor.router)
app.include_router(medical_monitor.router)
app.include_router(cpm.router)


@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "service": "rbqm-studio-api"}
