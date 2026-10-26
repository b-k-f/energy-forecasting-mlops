import pytest
import yaml

pytest.importorskip("optuna")

from energy_forecast.tune import tune  # noqa: E402


def test_tune_writes_best_params(small_df, test_config):
    best = tune(test_config, df=small_df)
    saved = yaml.safe_load(open(test_config["tuning"]["best_params_path"]))
    assert saved == best
    assert {"learning_rate", "max_iter", "max_leaf_nodes"} <= set(best)
