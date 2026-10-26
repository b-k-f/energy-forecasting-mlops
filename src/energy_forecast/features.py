"""Feature engineering for hourly consumption forecasting.

Input data must be on a complete hourly grid per building (missing readings
as NaN rows, not absent rows), because lags are computed by position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "consumption_kwh"
CATEGORICAL_FEATURES = ["building_type"]
CALENDAR_FEATURES = [
    "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
WEATHER_FEATURES = ["temperature", "heating_degree", "cooling_degree"]
LAG_FEATURES = ["lag_24", "lag_168", "rolling_mean_24"]
NUMERIC_FEATURES = WEATHER_FEATURES + CALENDAR_FEATURES + LAG_FEATURES
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Standard base temperatures used in building energy analysis (°C)
HEATING_BASE = 15.5
COOLING_BASE = 22.0


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features that only depend on the timestamp."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    # Cyclical encodings: 23h is next to 0h, 31 December next to 1 January
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    doy = ts.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Heating and cooling degree-hours: how far temperature is from comfort."""
    df = df.copy()
    df["heating_degree"] = np.clip(HEATING_BASE - df["temperature"], 0, None)
    df["cooling_degree"] = np.clip(df["temperature"] - COOLING_BASE, 0, None)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag and rolling features computed per building, without leakage."""
    df = df.sort_values(["building_id", "timestamp"]).copy()
    grouped = df.groupby("building_id")[TARGET]
    df["lag_24"] = grouped.shift(24)
    df["lag_168"] = grouped.shift(168)
    # shift(1) so the rolling window only uses past values
    df["rolling_mean_24"] = grouped.transform(lambda s: s.shift(1).rolling(24).mean())
    return df


def add_request_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features computable from a single API request (no history needed)."""
    return add_weather_features(add_calendar_features(df))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline; drops rows with a missing target or lags."""
    df = add_lag_features(add_request_features(df))
    return df.dropna(subset=[TARGET, *LAG_FEATURES]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last ``test_days`` days are held out."""
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]


def backtest_splits(df: pd.DataFrame, folds: int, fold_days: int):
    """Rolling-origin backtesting: successive test windows, each trained on the past.

    Yields ``(fold, train_df, test_df)`` from the oldest to the most recent window.
    """
    end = df["timestamp"].max()
    for k in reversed(range(folds)):
        test_end = end - pd.Timedelta(days=k * fold_days)
        test_start = test_end - pd.Timedelta(days=fold_days)
        train_df = df[df["timestamp"] <= test_start]
        test_df = df[(df["timestamp"] > test_start) & (df["timestamp"] <= test_end)]
        if len(train_df) and len(test_df):
            yield folds - k, train_df, test_df
