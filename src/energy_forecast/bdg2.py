"""Real data: the Building Data Genome Project 2 (BDG2).

BDG2 is the open dataset behind the ASHRAE Great Energy Predictor III
competition: hourly electricity meters (kWh) for 1,600+ non-residential
buildings over 2016-2017, with building metadata and hourly weather per site.

Reference: Miller, C. et al. The Building Data Genome Project 2, energy meter
data from the ASHRAE Great Energy Predictor III competition.
Sci Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x

This module downloads the three files once, selects a manageable subset of
buildings, and converts them to the same long format as the synthetic data:
``timestamp, building_id, building_type, temperature, consumption_kwh``.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

REPO = "buds-lab/building-data-genome-project-2"
FILES = {
    "electricity": "data/meters/cleaned/electricity_cleaned.csv",
    "weather": "data/weather/weather.csv",
    "metadata": "data/metadata/metadata.csv",
}
# Large files are stored with Git LFS: media.githubusercontent.com serves the
# real content, raw.githubusercontent.com is a fallback for regular files.
URL_TEMPLATES = [
    "https://media.githubusercontent.com/media/{repo}/master/{path}",
    "https://raw.githubusercontent.com/{repo}/master/{path}",
]
LFS_POINTER_PREFIX = b"version https://git-lfs"


def _download(path_in_repo: str, destination: Path) -> None:
    """Download one file, skipping Git LFS pointer files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for template in URL_TEMPLATES:
        url = template.format(repo=REPO, path=path_in_repo)
        tmp = destination.with_suffix(destination.suffix + ".part")
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, tmp)
            with open(tmp, "rb") as f:
                if f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                    raise ValueError("received a Git LFS pointer, not the data")
            tmp.replace(destination)
            return
        except Exception as error:  # try the next URL
            last_error = error
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {path_in_repo}: {last_error}")


def download_bdg2(cache_dir: str | Path) -> dict[str, Path]:
    """Download the BDG2 files into ``cache_dir`` if not already there."""
    cache_dir = Path(cache_dir)
    paths = {}
    for name, path_in_repo in FILES.items():
        destination = cache_dir / Path(path_in_repo).name
        if not destination.exists():
            _download(path_in_repo, destination)
        paths[name] = destination
    return paths


def _require(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing columns {missing}. Available: {list(df.columns)[:20]}")


def select_buildings(metadata: pd.DataFrame, electricity_columns: list[str],
                     primary_uses: dict[str, str], candidates_per_type: int,
                     seed: int) -> pd.DataFrame:
    """Candidate buildings per type that have an electricity meter."""
    _require(metadata, ["building_id", "site_id", "primaryspaceusage"], "metadata")
    available = set(electricity_columns)
    meta = metadata[metadata["building_id"].isin(available)].copy()
    use = meta["primaryspaceusage"].astype(str).str.lower()
    frames = []
    for source_name, building_type in primary_uses.items():
        subset = meta[use == source_name.lower()]
        if subset.empty:
            print(f"Warning: no buildings with primary use '{source_name}'")
            continue
        subset = subset.sample(min(candidates_per_type, len(subset)), random_state=seed)
        frames.append(subset.assign(building_type=building_type))
    if not frames:
        raise ValueError("No buildings matched the configured primary uses")
    return pd.concat(frames)[["building_id", "site_id", "building_type"]]


def load_bdg2(cfg: dict) -> pd.DataFrame:
    """Download (once) and prepare a BDG2 subset in the project's long format."""
    b = cfg["data"]["bdg2"]
    processed = Path(b["processed_path"])
    if processed.exists():
        return pd.read_csv(processed, parse_dates=["timestamp"])

    paths = download_bdg2(b["cache_dir"])
    metadata = pd.read_csv(paths["metadata"])
    header = pd.read_csv(paths["electricity"], nrows=0).columns.tolist()
    candidates = select_buildings(metadata, header[1:], b["primary_uses"],
                                  b["buildings_per_type"] * 3, b["seed"])

    # Read only the candidate columns: the full file has 1,500+ buildings
    electricity = pd.read_csv(paths["electricity"],
                              usecols=["timestamp", *candidates["building_id"]],
                              parse_dates=["timestamp"])
    electricity = electricity.set_index("timestamp").sort_index()
    full_index = pd.date_range(electricity.index.min(), electricity.index.max(), freq="h")
    electricity = electricity[~electricity.index.duplicated()].reindex(full_index)

    # Keep buildings with few missing readings, up to N per type
    missing_ratio = electricity.isna().mean()
    candidates["missing_ratio"] = candidates["building_id"].map(missing_ratio)
    selected = (candidates[candidates["missing_ratio"] <= b["max_missing_ratio"]]
                .sort_values("missing_ratio")
                .groupby("building_type").head(b["buildings_per_type"]))
    if selected.empty:
        raise ValueError("No building passed the missing-data filter; "
                         "increase data.bdg2.max_missing_ratio")

    long = (electricity[selected["building_id"]]
            .rename_axis("timestamp").reset_index()
            .melt(id_vars="timestamp", var_name="building_id", value_name="consumption_kwh")
            .merge(selected[["building_id", "site_id", "building_type"]], on="building_id"))

    weather = pd.read_csv(paths["weather"], parse_dates=["timestamp"])
    _require(weather, ["timestamp", "site_id", "airTemperature"], "weather")
    weather = (weather[weather["site_id"].isin(selected["site_id"].unique())]
               [["timestamp", "site_id", "airTemperature"]]
               .drop_duplicates(["site_id", "timestamp"])
               .rename(columns={"airTemperature": "temperature"}))
    long = long.merge(weather, on=["site_id", "timestamp"], how="left")
    # Fill short weather gaps per site (sensor dropouts of a few hours)
    long = long.sort_values(["building_id", "timestamp"])
    long["temperature"] = (long.groupby("building_id")["temperature"]
                           .transform(lambda s: s.interpolate(limit=6, limit_direction="both")))

    # Missing consumption stays NaN so the hourly grid stays complete
    # (lag features rely on it); those rows are dropped after feature building.
    long = long.dropna(subset=["temperature"])
    long["consumption_kwh"] = long["consumption_kwh"].astype(float).round(3)
    result = long[["timestamp", "building_id", "building_type", "temperature",
                   "consumption_kwh"]].reset_index(drop=True)

    processed.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(processed, index=False)
    summary = result.groupby("building_type")["building_id"].nunique().to_dict()
    print(f"Prepared {len(result):,} rows, buildings per type: {summary}")
    return result


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Per-building summary, handy for a first look at the real data."""
    return (df.groupby(["building_type", "building_id"])
              .agg(mean_kwh=("consumption_kwh", "mean"),
                   missing_pct=("consumption_kwh", lambda s: 100 * s.isna().mean()),
                   mean_temp=("temperature", "mean"))
              .round(2).reset_index())


__all__ = ["download_bdg2", "load_bdg2", "select_buildings", "summarize"]
