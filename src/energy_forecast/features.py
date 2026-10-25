"""Feature engineering for hourly consumption forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "consumption_kwh"
CATEGORICAL_FEATURES = ["building_type"]
NUMERIC_FEATURES = [
    "temperature", "hour", "dayofweek", "month", "is_weekend",
    "hour_sin", "hour_cos", "lag_24", "lag_168", "rolling_mean_24",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features that only depend on the timestamp."""
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.day_of_week
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
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


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline; drops rows where lags are not yet available."""
    df = add_lag_features(add_calendar_features(df))
    return df.dropna(subset=["lag_24", "lag_168", "rolling_mean_24"]).reset_index(drop=True)


def time_split(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: the last ``test_days`` days are held out."""
    cutoff = df["timestamp"].max() - pd.Timedelta(days=test_days)
    return df[df["timestamp"] <= cutoff], df[df["timestamp"] > cutoff]
