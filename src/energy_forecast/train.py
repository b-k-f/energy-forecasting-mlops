"""Train the forecasting model, compare it to a baseline, and track the run.

Experiments are logged to MLflow (params, metrics, model artifact). If MLflow
is not installed, metrics are still written to ``reports/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from energy_forecast.config import load_config
from energy_forecast.data import generate_dataset, load_dataset
from energy_forecast.features import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET, build_features, time_split,
)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # MLflow is optional for local experiments
    mlflow = None


def build_model(params: dict) -> Pipeline:
    """Preprocessing + gradient boosting regressor in one sklearn pipeline."""
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        random_state=params["random_state"],
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """MAE, RMSE and MAPE."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
    }


def train(cfg: dict, df: pd.DataFrame | None = None) -> dict[str, float]:
    """Run the full training pipeline and return the evaluation metrics."""
    if df is None:
        raw_path = Path(cfg["data"]["raw_path"])
        if raw_path.exists():
            df = load_dataset(raw_path)
        else:
            d = cfg["data"]
            df = generate_dataset(d["start"], d["periods_hours"],
                                  d["buildings_per_type"], d["seed"])

    features = build_features(df)
    train_df, test_df = time_split(features, cfg["split"]["test_days"])

    pipeline = build_model(cfg["model"])
    pipeline.fit(train_df[FEATURES], train_df[TARGET])

    preds = pipeline.predict(test_df[FEATURES])
    metrics = {f"model_{k}": v for k, v in regression_metrics(test_df[TARGET], preds).items()}
    # Seasonal naive baseline: same hour last week
    baseline = regression_metrics(test_df[TARGET], test_df["lag_168"])
    metrics.update({f"baseline_{k}": v for k, v in baseline.items()})
    metrics["mae_improvement_pct"] = float(
        100 * (baseline["mae"] - metrics["model_mae"]) / baseline["mae"]
    )

    art = cfg["artifacts"]
    for key in ("model_path", "reference_path", "metrics_path"):
        Path(art[key]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, art["model_path"])
    # Training features are kept as the reference distribution for drift checks
    train_df[NUMERIC_FEATURES].sample(min(20_000, len(train_df)), random_state=0) \
        .to_csv(art["reference_path"], index=False)
    with open(art["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if mlflow is not None and cfg["mlflow"].get("enabled", True):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run():
            mlflow.log_params(cfg["model"])
            mlflow.log_param("test_days", cfg["split"]["test_days"])
            mlflow.log_param("n_train_rows", len(train_df))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                input_example=train_df[FEATURES].head(5),
                skops_trusted_types=[
                    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
                ],
            )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forecasting model")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    metrics = train(load_config(args.config))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
