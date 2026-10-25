from pathlib import Path

from energy_forecast.train import train


def test_training_beats_baseline(small_df, test_config):
    metrics = train(test_config, df=small_df)
    assert metrics["model_mae"] < metrics["baseline_mae"]
    for key in ("model_path", "reference_path", "metrics_path"):
        assert Path(test_config["artifacts"][key]).exists()
