"""Train the forecasting model, evaluate it rigorously, and track the run.

One training run does four things:

1. Backtesting: trains and evaluates on several successive monthly windows,
   to measure how stable the error is over time.
2. Final model: trained on everything except the last ``test_days`` days,
   scored against a seasonal naive baseline (same hour last week).
3. Prediction intervals: two quantile models give a likely range around
   each prediction, and their coverage is measured on the test set.
4. Explainability: permutation importance shows which features matter.

Everything is logged to MLflow (params, metrics, model, reports). If MLflow
is not installed, results are still written to ``models/`` and ``reports/``.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from energy_forecast.config import load_config
from energy_forecast.data import load_training_data
from energy_forecast.features import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
    backtest_splits, build_features, time_split,
)

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # MLflow is optional for local experiments
    mlflow = None

SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

MODEL_PARAMS = ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf",
                "l2_regularization", "random_state")


def build_model(params: dict, quantile: float | None = None) -> Pipeline:
    """Preprocessing + gradient boosting regressor in one sklearn pipeline.

    With ``quantile`` set, the model predicts that quantile instead of the mean.
    """
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ("num", "passthrough", NUMERIC_FEATURES),
    ])
    kwargs = {k: params[k] for k in MODEL_PARAMS if k in params}
    if quantile is not None:
        kwargs.update(loss="quantile", quantile=quantile)
    model = HistGradientBoostingRegressor(**kwargs)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and MAPE."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape),
    }


def evaluate(pipeline: Pipeline, test_df: pd.DataFrame) -> dict[str, float]:
    """Model vs seasonal naive baseline on one test set."""
    model = regression_metrics(test_df[TARGET], pipeline.predict(test_df[FEATURES]))
    baseline = regression_metrics(test_df[TARGET], test_df["lag_168"])
    metrics = {f"model_{k}": v for k, v in model.items()}
    metrics.update({f"baseline_{k}": v for k, v in baseline.items()})
    metrics["mae_improvement_pct"] = 100 * (baseline["mae"] - model["mae"]) / baseline["mae"]
    return metrics


def run_backtest(features: pd.DataFrame, params: dict, folds: int,
                 fold_days: int) -> pd.DataFrame:
    """Train and evaluate on each backtest window; one row per fold."""
    rows = []
    for fold, train_df, test_df in backtest_splits(features, folds, fold_days):
        pipeline = build_model(params).fit(train_df[FEATURES], train_df[TARGET])
        rows.append({"fold": fold,
                     "test_start": test_df["timestamp"].min(),
                     "test_end": test_df["timestamp"].max(),
                     **evaluate(pipeline, test_df)})
    return pd.DataFrame(rows)


def interval_coverage(y_true, lower, upper) -> float:
    """Share of actual values that fall inside the predicted interval."""
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def feature_importance(pipeline: Pipeline, test_df: pd.DataFrame, sample_size: int,
                       n_repeats: int) -> pd.DataFrame:
    """Permutation importance: how much the error grows when a feature is shuffled."""
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=0)
    result = permutation_importance(pipeline, sample[FEATURES], sample[TARGET],
                                    scoring="neg_mean_absolute_error",
                                    n_repeats=n_repeats, random_state=0)
    return (pd.DataFrame({"feature": FEATURES,
                          "mae_increase": result.importances_mean,
                          "std": result.importances_std})
            .sort_values("mae_increase", ascending=False).reset_index(drop=True))


def plot_importance(importance: pd.DataFrame, path: str | Path) -> None:
    """Save a horizontal bar chart of feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(data["feature"], data["mae_increase"], xerr=data["std"], color="#4C78A8")
    ax.set_xlabel("MAE increase when shuffled (kWh)")
    ax.set_title("Permutation feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict, df: pd.DataFrame | None = None, source: str | None = None,
          params_override: dict | None = None) -> dict[str, float]:
    """Run the full training pipeline and return the evaluation metrics."""
    source = source or cfg["data"].get("source", "synthetic")
    if df is None:
        df = load_training_data(cfg, source)
    params = {**cfg["model"], **(params_override or {})}
    art = cfg["artifacts"]
    for key, path in art.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    features = build_features(df)

    # 1. Backtesting over several windows
    bt = cfg.get("backtest", {})
    backtest = run_backtest(features, params, bt.get("folds", 4), bt.get("fold_days", 30))
    backtest.to_csv(art["backtest_path"], index=False)

    # 2. Final model on the main split
    train_df, test_df = time_split(features, cfg["split"]["test_days"])
    pipeline = build_model(params).fit(train_df[FEATURES], train_df[TARGET])
    metrics = evaluate(pipeline, test_df)
    metrics["cv_model_mae_mean"] = float(backtest["model_mae"].mean())
    metrics["cv_model_mae_std"] = float(backtest["model_mae"].std(ddof=0))
    metrics["cv_baseline_mae_mean"] = float(backtest["baseline_mae"].mean())

    # 3. Prediction intervals
    iv = cfg.get("intervals", {"lower": 0.1, "upper": 0.9})
    quantile_models = {q: build_model(params, quantile=q).fit(train_df[FEATURES], train_df[TARGET])
                       for q in (iv["lower"], iv["upper"])}
    lower = quantile_models[iv["lower"]].predict(test_df[FEATURES])
    upper = quantile_models[iv["upper"]].predict(test_df[FEATURES])
    metrics["interval_coverage"] = interval_coverage(test_df[TARGET], lower, upper)
    metrics["interval_target_coverage"] = iv["upper"] - iv["lower"]
    metrics["interval_mean_width"] = float(np.mean(upper - lower))

    # 4. Explainability
    imp_cfg = cfg.get("importance", {})
    importance = feature_importance(pipeline, test_df, imp_cfg.get("sample_size", 5000),
                                    imp_cfg.get("n_repeats", 5))
    importance.to_csv(art["importance_path"], index=False)
    plot_importance(importance, art["importance_plot_path"])

    metrics = {k: float(v) for k, v in metrics.items()}
    joblib.dump(pipeline, art["model_path"])
    joblib.dump(quantile_models, art["quantile_model_path"])
    # Training features are kept as the reference distribution for drift checks
    train_df[NUMERIC_FEATURES].sample(min(20_000, len(train_df)), random_state=0) \
        .to_csv(art["reference_path"], index=False)
    with open(art["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if mlflow is not None and cfg["mlflow"].get("enabled", True):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run():
            mlflow.set_tag("data_source", source)
            mlflow.log_params(params)
            mlflow.log_params({"test_days": cfg["split"]["test_days"],
                               "n_train_rows": len(train_df),
                               "n_buildings": df["building_id"].nunique()})
            mlflow.log_metrics(metrics)
            for key in ("backtest_path", "importance_path", "importance_plot_path",
                        "quantile_model_path"):
                mlflow.log_artifact(art[key])
            log_kwargs = {"name": "model", "input_example": train_df[FEATURES].head(5)}
            # Recent MLflow saves with skops, which asks to confirm trusted types.
            # The model is trained right here, so its gradient boosting trees are trusted.
            if "skops_trusted_types" in inspect.signature(mlflow.sklearn.log_model).parameters:
                log_kwargs["skops_trusted_types"] = SKOPS_TRUSTED_TYPES
            mlflow.sklearn.log_model(pipeline, **log_kwargs)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forecasting model")
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", choices=["synthetic", "bdg2"], default=None,
                        help="Data source (default: data.source in the config)")
    parser.add_argument("--params", default=None,
                        help="YAML file of model parameters, e.g. reports/best_params.yaml")
    args = parser.parse_args()
    override = None
    if args.params:
        with open(args.params, encoding="utf-8") as f:
            override = yaml.safe_load(f)
    metrics = train(load_config(args.config), source=args.source, params_override=override)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
