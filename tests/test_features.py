import numpy as np
import pandas as pd

from energy_forecast.features import (
    FEATURES, TARGET, backtest_splits, build_features, time_split,
)


def test_no_missing_features(small_df):
    features = build_features(small_df)
    assert features[FEATURES].isna().sum().sum() == 0


def test_lag_24_has_no_leakage(small_df):
    features = build_features(small_df)
    one = features[features["building_id"] == features["building_id"].iloc[0]]
    one = one.set_index("timestamp")
    t = one.index[30]
    assert one.loc[t, "lag_24"] == one.loc[t - pd.Timedelta(hours=24), TARGET]


def test_degree_hours(small_df):
    features = build_features(small_df)
    cold = features[features["temperature"] < 10]
    assert (cold["heating_degree"] > 0).all() and (cold["cooling_degree"] == 0).all()


def test_missing_target_rows_are_dropped(small_df):
    df = small_df.copy()
    df.loc[df.index[500], TARGET] = np.nan
    assert build_features(df)[TARGET].isna().sum() == 0


def test_time_split_is_chronological(small_df):
    features = build_features(small_df)
    train, test = time_split(features, test_days=7)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) > 0 and len(test) > 0


def test_backtest_folds_never_train_on_the_future(small_df):
    features = build_features(small_df)
    folds = list(backtest_splits(features, folds=3, fold_days=7))
    assert len(folds) == 3
    for _, train, test in folds:
        assert train["timestamp"].max() < test["timestamp"].min()
