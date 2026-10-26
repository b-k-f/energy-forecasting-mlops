"""Hyperparameter tuning with Optuna, tracked in MLflow.

Each trial proposes model settings and is scored by its mean backtest MAE,
so tuning optimises for errors that hold across several months, not one
lucky window. With MLflow, the study is one parent run and each trial a
nested child run, so trials can be compared in the dashboard.

The best settings are written to ``reports/best_params.yaml``; train with
them using ``python -m energy_forecast.train --params reports/best_params.yaml``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from energy_forecast.config import load_config
from energy_forecast.data import load_training_data
from energy_forecast.features import build_features
from energy_forecast.train import mlflow, run_backtest


def suggest_params(trial) -> dict:
    """The search space."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_iter": trial.suggest_int("max_iter", 100, 800, step=50),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 15, 127),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
    }


def tune(cfg: dict, n_trials: int | None = None, source: str | None = None, df=None) -> dict:
    """Run the study and return the best parameters."""
    import optuna

    t = cfg.get("tuning", {})
    n_trials = n_trials or t.get("n_trials", 20)
    features = build_features(df if df is not None else load_training_data(cfg, source))
    base = {"random_state": cfg["model"].get("random_state", 42)}
    use_mlflow = mlflow is not None and cfg["mlflow"].get("enabled", True)

    def objective(trial) -> float:
        params = {**base, **suggest_params(trial)}
        backtest = run_backtest(features, params, t.get("folds", 2), t.get("fold_days", 30))
        score = float(backtest["model_mae"].mean())
        if use_mlflow:
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_model_mae_mean", score)
        return score

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=base["random_state"]))
    if use_mlflow:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
        mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
        with mlflow.start_run(run_name="optuna-tuning"):
            mlflow.set_tag("data_source", source or cfg["data"].get("source", "synthetic"))
            study.optimize(objective, n_trials=n_trials)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_cv_model_mae_mean", study.best_value)
    else:
        study.optimize(objective, n_trials=n_trials)

    best = {**study.best_params, **base}
    out = Path(t.get("best_params_path", "reports/best_params.yaml"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(best, sort_keys=True), encoding="utf-8")
    print(f"Best mean backtest MAE: {study.best_value:.3f} kWh")
    print(f"Best parameters saved to {out}:\n{yaml.safe_dump(best, sort_keys=True)}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optuna")
    parser.add_argument("--config", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--source", choices=["synthetic", "bdg2"], default=None)
    args = parser.parse_args()
    tune(load_config(args.config), n_trials=args.trials, source=args.source)


if __name__ == "__main__":
    main()
