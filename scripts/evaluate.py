"""Benchmark anomaly detection and forecasting against ground truth.

Method:
  1. Generate VALIDATION datasets (seeds 0-4). For each detector, pick the threshold
     with the best average F1 there.
  2. Generate separate TEST datasets (seeds 42-46) and score every detector with the
     threshold picked in step 1. No method sees its test data while being tuned.
  3. Backtest every forecaster on the test datasets (rolling origin, 3 x 30-day holdouts).

Writes:
  reports/results.json   machine-readable numbers (read by the dashboard)
  reports/RESULTS.md     human-readable tables

Usage:  python scripts/evaluate.py [--seeds 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analytics.data import Config, generate, service_daily, total_daily  # noqa: E402
from analytics.detection import (  # noqa: E402
    isolation_forest, rolling_zscore, static_threshold, stl_detector,
)
from analytics.evaluation import score_detector  # noqa: E402
from analytics.forecasting import FORECASTERS, backtest  # noqa: E402

HORIZON, FOLDS = 30, 3

# name -> (function, parameter to tune, grid). A grid of one value means "not tuned".
DETECTOR_GRID = {
    # The original rule. Its k is tuned too, so the comparison is fair; k=2 turns out to be its best.
    "Static mean + kσ (original)": (static_threshold, "k", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]),
    "Rolling z-score (weekday-aware)": (rolling_zscore, "k", [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]),
    "STL seasonal decomposition": (stl_detector, "k", [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0]),
    "IsolationForest": (isolation_forest, "contamination", [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]),
}


def datasets(seeds):
    for seed in seeds:
        cfg = replace(Config(), seed=seed)
        usage, labels = generate(cfg)
        yield cfg, usage, labels


def tune(val_seeds) -> dict:
    """Return {method: best parameter value} chosen by mean F1 on validation data."""
    val = [(cfg, service_daily(u), l) for cfg, u, l in datasets(val_seeds)]
    chosen = {}
    for name, (fn, param, grid) in DETECTOR_GRID.items():
        f1 = {v: np.mean([score_detector(fn(sd, **{param: v}), l, cfg.days)["f1"] for cfg, sd, l in val])
              for v in grid}
        chosen[name] = max(f1, key=f1.get)
    return chosen


def run(val_seeds, test_seeds) -> dict:
    params = tune(val_seeds)
    det_rows, fc_rows = [], []
    for cfg, usage, labels in datasets(test_seeds):
        sd = service_daily(usage)
        for name, (fn, param, _) in DETECTOR_GRID.items():
            out = fn(sd, **{param: params[name]})
            det_rows.append({"seed": cfg.seed, "method": name, **score_detector(out, labels, cfg.days)})
        series = total_daily(usage)
        for name, fn in FORECASTERS.items():
            scores = backtest(series, fn, HORIZON, FOLDS)
            fc_rows.append({"seed": cfg.seed, "method": name,
                            "mape_last30": scores[0], "mape_mean": float(np.mean(scores))})

    det = pd.DataFrame(det_rows)
    fc = pd.DataFrame(fc_rows)
    det_summary = det.groupby("method", sort=False).agg(
        recall=("recall", "mean"), precision=("precision", "mean"), f1=("f1", "mean"),
        false_alarms_per_month=("false_alarms_per_month", "mean"),
        events_caught=("events_caught", "sum"), events_total=("events_total", "sum"),
    ).reset_index()
    det_summary["param"] = [f"{DETECTOR_GRID[m][1]}={params[m]}" for m in det_summary["method"]]
    fc_summary = fc.groupby("method", sort=False).agg(
        mape_last30=("mape_last30", "mean"), mape_mean=("mape_mean", "mean"),
    ).reset_index()
    return {
        "validation_seeds": list(val_seeds),
        "test_seeds": list(test_seeds),
        "setup": {"days": Config().days, "anomalies_per_dataset": Config().n_anomalies,
                  "forecast_horizon_days": HORIZON, "backtest_folds": FOLDS},
        "tuned_params": params,
        "detection": det_summary.to_dict(orient="records"),
        "forecast": fc_summary.to_dict(orient="records"),
    }


def to_markdown(r: dict) -> str:
    s = r["setup"]
    lines = [
        "# Benchmark results",
        "",
        f"Each dataset is {s['days']} days of synthetic cost data for 6 services x 5 teams, with weekly cycles, "
        f"growth, a holiday dip, a legitimate workload launch, and {s['anomalies_per_dataset']} planted cost spikes "
        f"(+10% to +100% of the service's daily spend, lasting 1-3 days).",
        "",
        f"Thresholds were tuned on validation datasets (seeds {r['validation_seeds']}) and scored on separate "
        f"test datasets (seeds {r['test_seeds']}). Numbers are averages over the test datasets.",
        "",
        "## Anomaly detection (per service, per day)",
        "",
        "| Method | Setting | Recall (spikes caught) | Precision | F1 | False alarms / month |",
        "|---|---|---|---|---|---|",
    ]
    for d in r["detection"]:
        lines.append(f"| {d['method']} | {d['param']} | {d['recall']:.0%} ({d['events_caught']}/{d['events_total']}) | "
                     f"{d['precision']:.0%} | {d['f1']:.2f} | {d['false_alarms_per_month']:.1f} |")
    lines += [
        "",
        f"## Forecast backtest (total daily cost, {s['forecast_horizon_days']}-day horizon)",
        "",
        "MAPE = mean absolute percentage error. 'Last 30 days' hides the final month and forecasts it; "
        f"'{s['backtest_folds']}-fold' averages three consecutive 30-day holdouts (rolling origin).",
        "",
        "| Method | MAPE, last 30 days | MAPE, 3-fold average |",
        "|---|---|---|",
    ]
    for f in r["forecast"]:
        lines.append(f"| {f['method']} | {f['mape_last30']:.1f}% | {f['mape_mean']:.1f}% |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5, help="number of validation and of test datasets")
    args = ap.parse_args()
    results = run(range(0, args.seeds), range(42, 42 + args.seeds))
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    md = to_markdown(results)
    with open(os.path.join(ROOT, "reports", "RESULTS.md"), "w") as fh:
        fh.write(md)
    print(md)


if __name__ == "__main__":
    main()
