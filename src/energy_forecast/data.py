"""Synthetic building energy dataset.

Generates hourly electricity consumption for several buildings of different
types (office, residential, retail). Consumption depends on occupancy
patterns, weekday/weekend, and heating/cooling demand driven by outdoor
temperature, so the data behaves like real smart-meter data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecast.config import load_config

BUILDING_PROFILES = {
    # base load (kWh), peak occupancy load, heating & cooling sensitivity
    "office": {"base": 20.0, "peak": 60.0, "heat": 2.5, "cool": 3.0},
    "residential": {"base": 8.0, "peak": 15.0, "heat": 1.8, "cool": 1.2},
    "retail": {"base": 15.0, "peak": 45.0, "heat": 2.0, "cool": 3.5},
}


def _temperature(index: pd.DatetimeIndex, rng: np.random.Generator,
                 shift: float = 0.0) -> np.ndarray:
    """Outdoor temperature (°C) with yearly and daily cycles."""
    day_of_year = index.day_of_year.to_numpy()
    hour = index.hour.to_numpy()
    yearly = 12 - 9 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    daily = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    noise = rng.normal(0, 1.5, len(index))
    return yearly + daily + noise + shift


def _occupancy(index: pd.DatetimeIndex, building_type: str) -> np.ndarray:
    """Occupancy factor for each hour."""
    hour = index.hour.to_numpy()
    weekend = index.day_of_week.to_numpy() >= 5

    if building_type == "office":
        occ = ((hour >= 8) & (hour < 19)).astype(float)
        occ[weekend] *= 0.1
    elif building_type == "retail":
        occ = ((hour >= 9) & (hour < 21)).astype(float)
        occ[weekend] *= 1.2
    else:  # residential: morning and evening peaks, higher on weekends
        occ = np.exp(-((hour - 7.5) ** 2) / 3) + np.exp(-((hour - 19.5) ** 2) / 5)
        occ[weekend] *= 1.3
    return np.clip(occ, 0, 1.3)


def generate_dataset(start: str = "2024-01-01", periods_hours: int = 8760,
                     buildings_per_type: int = 3, seed: int = 42,
                     temperature_shift: float = 0.0,
                     building_seed: int = 42) -> pd.DataFrame:
    """Generate a long-format dataframe with one row per building and hour.

    ``seed`` controls the weather and measurement noise, ``building_seed`` the
    buildings themselves (their size), so two datasets with different ``seed``
    but the same ``building_seed`` represent the same buildings over another
    period. ``temperature_shift`` lets you simulate a climate / data drift.
    """
    rng = np.random.default_rng(seed)
    building_rng = np.random.default_rng(building_seed)
    index = pd.date_range(start, periods=periods_hours, freq="h")
    temperature = _temperature(index, rng, shift=temperature_shift)

    frames = []
    for building_type, p in BUILDING_PROFILES.items():
        for i in range(buildings_per_type):
            scale = building_rng.uniform(0.7, 1.4)  # building size
            heating = np.clip(15 - temperature, 0, None) * p["heat"]
            cooling = np.clip(temperature - 22, 0, None) * p["cool"]
            occ = _occupancy(index, building_type)
            consumption = scale * (p["base"] + p["peak"] * occ + heating + cooling)
            consumption *= rng.normal(1, 0.05, len(index))
            frames.append(pd.DataFrame({
                "timestamp": index,
                "building_id": f"{building_type}_{i:02d}",
                "building_type": building_type,
                "temperature": temperature.round(2),
                "consumption_kwh": np.clip(consumption, 0, None).round(3),
            }))
    return pd.concat(frames, ignore_index=True)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset saved by :func:`main`."""
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_training_data(cfg: dict, source: str | None = None) -> pd.DataFrame:
    """Return the training data for the configured source.

    ``synthetic`` uses ``data.raw_path`` if it exists, otherwise generates data.
    ``bdg2`` downloads and prepares the real Building Data Genome 2 subset.
    """
    source = source or cfg["data"].get("source", "synthetic")
    if source == "bdg2":
        from energy_forecast.bdg2 import load_bdg2
        return load_bdg2(cfg)
    if source != "synthetic":
        raise ValueError(f"Unknown data source: {source!r} (use 'synthetic' or 'bdg2')")
    d = cfg["data"]
    raw_path = Path(d["raw_path"])
    if raw_path.exists():
        return load_dataset(raw_path)
    return generate_dataset(d["start"], d["periods_hours"], d["buildings_per_type"], d["seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic dataset")
    parser.add_argument("--config", default=None)
    parser.add_argument("--temperature-shift", type=float, default=0.0,
                        help="Shift temperatures (°C) to simulate drift")
    parser.add_argument("--start", default=None, help="Override start date")
    parser.add_argument("--periods", type=int, default=None, help="Override number of hours")
    parser.add_argument("--output", default=None)
    parser.add_argument("--source", choices=["synthetic", "bdg2"], default="synthetic",
                        help="bdg2 downloads and prepares the real dataset instead")
    args = parser.parse_args()

    if args.source == "bdg2":
        from energy_forecast.bdg2 import load_bdg2, summarize
        df = load_bdg2(load_config(args.config))
        print(summarize(df).to_string(index=False))
        return

    cfg = load_config(args.config)["data"]
    df = generate_dataset(args.start or cfg["start"],
                          args.periods or cfg["periods_hours"],
                          cfg["buildings_per_type"], cfg["seed"],
                          args.temperature_shift)
    out = Path(args.output or cfg["raw_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
