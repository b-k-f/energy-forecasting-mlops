import pytest

from energy_forecast.data import generate_dataset


@pytest.fixture(scope="session")
def small_df():
    """Two months of data, enough for 168h lags and a test window."""
    return generate_dataset(periods_hours=24 * 60, buildings_per_type=1, seed=0)


@pytest.fixture
def test_config(tmp_path):
    return {
        "data": {"raw_path": str(tmp_path / "missing.csv")},
        "split": {"test_days": 7},
        "model": {"max_iter": 50, "learning_rate": 0.1,
                  "max_leaf_nodes": 15, "random_state": 0},
        "artifacts": {
            "model_path": str(tmp_path / "model.joblib"),
            "reference_path": str(tmp_path / "reference.csv"),
            "metrics_path": str(tmp_path / "metrics.json"),
            "drift_report_path": str(tmp_path / "drift.json"),
        },
        "mlflow": {"enabled": False},
    }
