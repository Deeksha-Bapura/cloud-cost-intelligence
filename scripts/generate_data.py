"""Write the demo dataset and its ground-truth labels.

  data/cloud_usage.csv          one row per service x team x day (12 months)
  data/planted_anomalies.csv    the cost spikes planted in it (ground truth)

Usage:  python scripts/generate_data.py [--seed 42]
"""
import argparse
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analytics.data import Config, generate  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

usage, labels = generate(replace(Config(), seed=args.seed))
os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
usage.to_csv(os.path.join(ROOT, "data", "cloud_usage.csv"), index=False, date_format="%Y-%m-%d")
labels.to_csv(os.path.join(ROOT, "data", "planted_anomalies.csv"), index=False, date_format="%Y-%m-%d")

print(f"Wrote {len(usage):,} rows to data/cloud_usage.csv "
      f"and {labels['event_id'].nunique()} planted anomalies to data/planted_anomalies.csv")
