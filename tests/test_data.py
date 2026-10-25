import warnings

from energy_forecast.data import BUILDING_PROFILES, generate_dataset


def test_schema_and_size(small_df):
    assert list(small_df.columns) == [
        "timestamp", "building_id", "building_type", "temperature", "consumption_kwh",
    ]
    assert len(small_df) == 24 * 60 * len(BUILDING_PROFILES)


def test_values_are_valid(small_df):
    assert small_df.isna().sum().sum() == 0
    assert (small_df["consumption_kwh"] >= 0).all()


def test_reproducible():
    a = generate_dataset(periods_hours=48, buildings_per_type=1, seed=1)
    b = generate_dataset(periods_hours=48, buildings_per_type=1, seed=1)
    assert a.equals(b)


def test_offices_consume_less_on_weekends(small_df):
    office = small_df[small_df["building_type"] == "office"]
    weekend = office["timestamp"].dt.day_of_week >= 5
    assert office[weekend]["consumption_kwh"].mean() < office[~weekend]["consumption_kwh"].mean()


def test_pandas_datetime_accessors_are_up_to_date():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        generate_dataset(periods_hours=24, buildings_per_type=1)
