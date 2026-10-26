"""Data drift monitoring.

Compares the distribution of features in new (production) data with the
training reference using two complementary statistics:

* PSI (Population Stability Index): magnitude of the shift
* Kolmogorov-Smirnov test: statistical significance of the shift
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from energy_forecast.config import load_config
from energy_forecast.data import load_dataset
# Calendar features are skipped: their distribution is fixed by the calendar
from energy_forecast.features import CALENDAR_FEATURES, NUMERIC_FEATURES, build_features


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index using quantile bins from the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 psi_threshold: float = 0.2, ks_pvalue_threshold: float = 0.01) -> dict:
    """Per-feature drift statistics and an overall verdict."""
    features = [f for f in NUMERIC_FEATURES if f not in CALENDAR_FEATURES]
    results = {}
    for feature in features:
        ref, cur = reference[feature].dropna().to_numpy(), current[feature].dropna().to_numpy()
        psi_value = psi(ref, cur)
        ks = ks_2samp(ref, cur)
        results[feature] = {
            "psi": round(psi_value, 4),
            "ks_pvalue": float(ks.pvalue),
            "drift": bool(psi_value > psi_threshold and ks.pvalue < ks_pvalue_threshold),
        }
    drifted = [f for f, r in results.items() if r["drift"]]
    return {"dataset_drift": bool(drifted), "drifted_features": drifted, "features": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check new data for drift")
    parser.add_argument("current", help="CSV of new raw data (same schema as training)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--fail-on-drift", action="store_true",
                        help="Exit with code 1 if drift is detected (useful in CI/cron)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    reference = pd.read_csv(cfg["artifacts"]["reference_path"])
    current = build_features(load_dataset(args.current))
    report = drift_report(reference, current, **cfg["monitoring"])

    out = Path(cfg["artifacts"]["drift_report_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if args.fail_on_drift and report["dataset_drift"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
