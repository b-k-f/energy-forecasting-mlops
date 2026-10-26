"""BDG2 loader tests on a tiny fake dataset in the real file format (no download)."""

import numpy as np
import pandas as pd

from energy_forecast import bdg2


def _write_fake_bdg2(folder):
    index = pd.date_range("2016-01-01", periods=24 * 20, freq="h")
    rng = np.random.default_rng(0)
    electricity = pd.DataFrame({"timestamp": index})
    for b in ["Fox_office_A", "Fox_office_B", "Fox_education_C", "Fox_parking_D"]:
        electricity[b] = rng.uniform(50, 100, len(index))
    electricity.loc[10:400, "Fox_office_B"] = np.nan  # too many missing values
    electricity.to_csv(folder / "electricity_cleaned.csv", index=False)
    pd.DataFrame({
        "building_id": ["Fox_office_A", "Fox_office_B", "Fox_education_C", "Fox_parking_D"],
        "site_id": ["Fox"] * 4,
        "primaryspaceusage": ["Office", "Office", "Education", "Parking"],
    }).to_csv(folder / "metadata.csv", index=False)
    weather = pd.DataFrame({"timestamp": index, "site_id": "Fox",
                            "airTemperature": rng.normal(10, 5, len(index))})
    weather.loc[5:7, "airTemperature"] = np.nan  # short gap, should be interpolated
    weather.to_csv(folder / "weather.csv", index=False)


def _config(tmp_path):
    return {"data": {"bdg2": {
        "cache_dir": str(tmp_path / "cache"),
        "processed_path": str(tmp_path / "processed.csv"),
        "primary_uses": {"Office": "office", "Education": "education"},
        "buildings_per_type": 5, "max_missing_ratio": 0.05, "seed": 0,
    }}}


def test_load_bdg2_converts_to_project_format(tmp_path):
    (tmp_path / "cache").mkdir()
    _write_fake_bdg2(tmp_path / "cache")
    df = bdg2.load_bdg2(_config(tmp_path))

    assert list(df.columns) == ["timestamp", "building_id", "building_type",
                                "temperature", "consumption_kwh"]
    assert set(df["building_id"]) == {"Fox_office_A", "Fox_education_C"}
    assert set(df["building_type"]) == {"office", "education"}
    assert df["temperature"].isna().sum() == 0
    assert (tmp_path / "processed.csv").exists()


def test_processed_file_is_reused(tmp_path, monkeypatch):
    (tmp_path / "cache").mkdir()
    _write_fake_bdg2(tmp_path / "cache")
    first = bdg2.load_bdg2(_config(tmp_path))
    monkeypatch.setattr(bdg2, "download_bdg2", lambda *_: (_ for _ in ()).throw(AssertionError))
    second = bdg2.load_bdg2(_config(tmp_path))
    assert len(first) == len(second)
