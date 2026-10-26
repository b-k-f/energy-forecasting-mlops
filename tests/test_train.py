import json
from pathlib import Path

import pandas as pd

from energy_forecast.train import interval_coverage, train


def test_training_beats_baseline_and_saves_artifacts(small_df, test_config):
    metrics = train(test_config, df=small_df)
    assert metrics["model_mae"] < metrics["baseline_mae"]
    for path in test_config["artifacts"].values():
        if not path.endswith("drift.json"):
            assert Path(path).exists(), path
    assert json.loads(Path(test_config["artifacts"]["metrics_path"]).read_text())


def test_backtest_and_intervals_are_reported(small_df, test_config):
    metrics = train(test_config, df=small_df)
    backtest = pd.read_csv(test_config["artifacts"]["backtest_path"])
    assert len(backtest) == 2
    assert metrics["cv_model_mae_mean"] > 0
    assert 0.5 < metrics["interval_coverage"] <= 1.0


def test_feature_importance_ranks_lags_high(small_df, test_config):
    train(test_config, df=small_df)
    importance = pd.read_csv(test_config["artifacts"]["importance_path"])
    assert importance["feature"].head(5).str.contains("lag|rolling|hour").any()


def test_params_override(small_df, test_config):
    metrics = train(test_config, df=small_df, params_override={"max_iter": 20})
    assert metrics["model_mae"] > 0


def test_interval_coverage():
    assert interval_coverage([1, 2, 3, 4], [0, 0, 0, 5], [5, 5, 5, 6]) == 0.75
