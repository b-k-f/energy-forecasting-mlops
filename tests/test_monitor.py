import numpy as np

from energy_forecast.data import generate_dataset
from energy_forecast.features import build_features
from energy_forecast.monitor import drift_report, psi


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi(x, x) < 1e-6


def test_no_drift_on_same_distribution():
    ref = build_features(generate_dataset(periods_hours=24 * 30, buildings_per_type=1, seed=0))
    cur = build_features(generate_dataset(periods_hours=24 * 30, buildings_per_type=1, seed=1))
    assert not drift_report(ref, cur)["dataset_drift"]


def test_detects_temperature_drift():
    ref = build_features(generate_dataset(periods_hours=24 * 30, buildings_per_type=1, seed=0))
    cur = build_features(generate_dataset(periods_hours=24 * 30, buildings_per_type=1,
                                          seed=1, temperature_shift=8.0))
    report = drift_report(ref, cur)
    assert report["dataset_drift"]
    assert "temperature" in report["drifted_features"]
