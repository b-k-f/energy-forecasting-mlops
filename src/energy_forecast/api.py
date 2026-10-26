"""REST API serving the trained model with FastAPI."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from energy_forecast import __version__
from energy_forecast.features import FEATURES, add_request_features

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
QUANTILE_MODEL_PATH = Path(os.getenv("QUANTILE_MODEL_PATH", "models/quantile_models.joblib"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "reports/metrics.json"))
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        state["model"] = model
        # Building types the model was trained on (from its one-hot encoder)
        encoder = model.named_steps["preprocess"].named_transformers_["cat"]
        state["building_types"] = sorted(str(c) for c in encoder.categories_[0])
    if QUANTILE_MODEL_PATH.exists():
        state["quantile_models"] = joblib.load(QUANTILE_MODEL_PATH)
    if METRICS_PATH.exists():
        state["metrics"] = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    yield
    state.clear()


app = FastAPI(title="Energy Consumption Forecasting API", version=__version__,
              lifespan=lifespan)


class ForecastRequest(BaseModel):
    timestamp: datetime = Field(..., examples=["2024-12-02T09:00:00"])
    building_type: str = Field(..., examples=["office"],
                               description="One of the types listed by /model-info")
    temperature: float = Field(..., description="Outdoor temperature (°C)", examples=[6.5])
    lag_24: float = Field(..., ge=0, description="Consumption 24h earlier (kWh)")
    lag_168: float = Field(..., ge=0, description="Consumption 168h earlier (kWh)")
    rolling_mean_24: float = Field(..., ge=0, description="Mean consumption over last 24h")


class ForecastResponse(BaseModel):
    timestamp: datetime
    building_type: str
    predicted_consumption_kwh: float
    lower_kwh: float | None = Field(None, description="Lower bound of the prediction interval")
    upper_kwh: float | None = Field(None, description="Upper bound of the prediction interval")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/model-info")
def model_info() -> dict:
    return {"version": __version__, "features": FEATURES,
            "building_types": state.get("building_types"),
            "prediction_intervals": "quantile_models" in state,
            "metrics": state.get("metrics")}


@app.post("/predict", response_model=list[ForecastResponse])
def predict(requests: list[ForecastRequest]) -> list[ForecastResponse]:
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training first.")
    if not requests:
        raise HTTPException(status_code=422, detail="Empty request list.")

    unknown = sorted({r.building_type for r in requests} - set(state["building_types"]))
    if unknown:
        raise HTTPException(status_code=422,
                            detail=f"Unknown building_type {unknown}. "
                                   f"Expected one of {state['building_types']}.")

    df = add_request_features(pd.DataFrame([r.model_dump() for r in requests]))
    preds = state["model"].predict(df[FEATURES])
    lower = upper = [None] * len(requests)
    if "quantile_models" in state:
        low_q, high_q = sorted(state["quantile_models"])
        # The point forecast always lies inside its interval
        lower = np.minimum(state["quantile_models"][low_q].predict(df[FEATURES]), preds)
        upper = np.maximum(state["quantile_models"][high_q].predict(df[FEATURES]), preds)

    def r3(x):
        return None if x is None else round(float(x), 3)

    return [
        ForecastResponse(timestamp=r.timestamp, building_type=r.building_type,
                         predicted_consumption_kwh=r3(p), lower_kwh=r3(lo), upper_kwh=r3(hi))
        for r, p, lo, hi in zip(requests, preds, lower, upper)
    ]
