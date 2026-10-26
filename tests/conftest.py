import pytest

from energy_forecast.data import generate_dataset


@pytest.fixture(scope="session")
def small_df():
    """Two months of data, enough for 168h lags and a test window."""
    return generate_dataset(periods_hours=24 * 60, buildings_per_type=1, seed=0)


@pytest.fixture
def test_config(tmp_path):
    artifacts = {
        "model_path": "model.joblib",
        "quantile_model_path": "quantile_models.joblib",
        "reference_path": "reference.csv",
        "metrics_path": "metrics.json",
        "backtest_path": "backtest.csv",
        "importance_path": "importance.csv",
        "importance_plot_path": "importance.png",
        "drift_report_path": "drift.json",
    }
    return {
        "data": {"source": "synthetic", "raw_path": str(tmp_path / "missing.csv"),
                 "start": "2024-01-01", "periods_hours": 24 * 60,
                 "buildings_per_type": 1, "seed": 0},
        "split": {"test_days": 7},
        "backtest": {"folds": 2, "fold_days": 7},
        "model": {"max_iter": 50, "learning_rate": 0.1,
                  "max_leaf_nodes": 15, "random_state": 0},
        "intervals": {"lower": 0.1, "upper": 0.9},
        "importance": {"sample_size": 300, "n_repeats": 2},
        "tuning": {"n_trials": 2, "folds": 1, "fold_days": 7,
                   "best_params_path": str(tmp_path / "best_params.yaml")},
        "artifacts": {k: str(tmp_path / v) for k, v in artifacts.items()},
        "mlflow": {"enabled": False},
    }
