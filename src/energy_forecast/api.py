"""REST API serving the trained model with FastAPI."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from energy_forecast import __version__
from energy_forecast.features import FEATURES, add_calendar_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "reports/metrics.json"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        state["model"] = joblib.load(MODEL_PATH)
    if METRICS_PATH.exists():
        state["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Energy Consumption Forecasting API", version=__version__,
              lifespan=lifespan)


class ForecastRequest(BaseModel):
    timestamp: datetime = Field(..., examples=["2024-12-02T09:00:00"])
    building_type: Literal["office", "residential", "retail"]
    temperature: float = Field(..., description="Outdoor temperature (°C)", examples=[6.5])
    lag_24: float = Field(..., ge=0, description="Consumption 24h earlier (kWh)")
    lag_168: float = Field(..., ge=0, description="Consumption 168h earlier (kWh)")
    rolling_mean_24: float = Field(..., ge=0, description="Mean consumption over last 24h")


class ForecastResponse(BaseModel):
    timestamp: datetime
    building_type: str
    predicted_consumption_kwh: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info() -> dict:
    return {"version": __version__, "features": FEATURES, "metrics": state.get("metrics")}


@app.post("/predict", response_model=list[ForecastResponse])
def predict(requests: list[ForecastRequest]) -> list[ForecastResponse]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    if not requests:
        raise HTTPException(status_code=422, detail="Empty request list.")

    df = add_calendar_features(pd.DataFrame([r.model_dump() for r in requests]))
    preds = state["model"].predict(df[FEATURES])
    return [
        ForecastResponse(timestamp=r.timestamp, building_type=r.building_type,
                         predicted_consumption_kwh=round(float(p), 3))
        for r, p in zip(requests, preds)
    ]
