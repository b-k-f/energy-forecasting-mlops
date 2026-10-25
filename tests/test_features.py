import pandas as pd

from energy_forecast.features import FEATURES, TARGET, build_features, time_split


def test_no_missing_features(small_df):
    features = build_features(small_df)
    assert features[FEATURES].isna().sum().sum() == 0


def test_lag_24_has_no_leakage(small_df):
    features = build_features(small_df)
    one = features[features["building_id"] == features["building_id"].iloc[0]]
    one = one.set_index("timestamp")
    t = one.index[30]
    assert one.loc[t, "lag_24"] == one.loc[t - pd.Timedelta(hours=24), TARGET]


def test_time_split_is_chronological(small_df):
    features = build_features(small_df)
    train, test = time_split(features, test_days=7)
    assert train["timestamp"].max() < test["timestamp"].min()
    assert len(train) > 0 and len(test) > 0
