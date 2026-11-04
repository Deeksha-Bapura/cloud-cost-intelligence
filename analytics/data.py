"""Synthetic cloud billing data with known (planted) anomalies.

The generator models the things that make real cost data hard to monitor:
weekly cycles, steady growth, a holiday dip, a legitimate step change
(a new workload launch) and noise. On top of that it plants cost spikes
at known dates, so detectors can be scored against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Per-service shape: typical daily $ at start, weekend multiplier, monthly growth,
# and the $/hour used to back out usage hours.
SERVICES = {
    "EC2":        dict(base=420.0, weekend=0.62, growth=0.030, unit_price=0.42),
    "EKS":        dict(base=260.0, weekend=0.70, growth=0.045, unit_price=0.38),
    "RDS":        dict(base=210.0, weekend=0.92, growth=0.020, unit_price=0.55),
    "S3":         dict(base=90.0,  weekend=0.98, growth=0.050, unit_price=0.023),
    "Lambda":     dict(base=55.0,  weekend=0.45, growth=0.060, unit_price=0.20),
    "CloudFront": dict(base=110.0, weekend=0.80, growth=0.035, unit_price=0.085),
}
TEAMS = ["ml", "backend", "frontend", "data", "infra"]
REGIONS = ["us-east-1", "us-west-2", "eu-central-1"]


@dataclass(frozen=True)
class Config:
    start: str = "2025-11-01"
    days: int = 365
    n_anomalies: int = 30
    noise: float = 0.07           # lognormal sigma on each service x team row
    warmup_days: int = 35         # no planted spikes before this (detectors need history)
    seed: int = 42


def generate(cfg: Config = Config()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (usage, labels).

    usage:  date, service, team, region, usage_hours, cost  (one row per service x team x day)
    labels: one row per planted anomaly-day: event_id, date, service, team, extra_cost
    """
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start, periods=cfg.days, freq="D")
    t_months = np.arange(cfg.days) / 30.4
    dow = dates.dayofweek.to_numpy()
    weekend = dow >= 5

    # Holiday slowdown: Dec 22 - Jan 2 runs at ~80%.
    md = dates.month * 100 + dates.day
    holiday = np.where((md >= 1222) | (md <= 102), 0.80, 1.0)

    # Each team owns a random share of each service, fixed for the year,
    # and runs it in one region.
    shares = {s: rng.dirichlet(np.ones(len(TEAMS)) * 1.5) for s in SERVICES}
    region_of = {(s, t): REGIONS[rng.integers(len(REGIONS))] for s in SERVICES for t in TEAMS}

    # Legitimate step change: backend moves a workload onto EKS mid-year.
    launch = pd.Timestamp(cfg.start) + pd.Timedelta(days=int(cfg.days * 0.58))

    rows = []
    for s, p in SERVICES.items():
        weekly = np.where(weekend, p["weekend"], 1.0)
        trend = (1 + p["growth"]) ** t_months
        service_curve = p["base"] * trend * weekly * holiday
        for i, team in enumerate(TEAMS):
            cost = service_curve * shares[s][i]
            if s == "EKS" and team == "backend":
                cost = cost * np.where(dates >= launch, 1.45, 1.0)
            if team == "ml":  # month-end training/batch jobs
                cost = cost * np.where(dates.day >= 27, 1.12, 1.0)
            cost = cost * rng.lognormal(0.0, cfg.noise, cfg.days)
            for d, c in zip(dates, cost):
                rows.append((d, s, team, region_of[(s, team)], c))

    usage = pd.DataFrame(rows, columns=["date", "service", "team", "region", "cost"])

    # Plant anomalies. Size is relative to the service's daily total, so a spike is
    # +10%..+100% of what the whole service normally costs that day: some easy, some subtle.
    svc_daily = usage.groupby(["service", "date"])["cost"].sum()
    labels = []
    services = list(SERVICES)
    for event_id in range(cfg.n_anomalies):
        s = services[rng.integers(len(services))]
        team = TEAMS[rng.integers(len(TEAMS))]
        start_idx = int(rng.integers(cfg.warmup_days, cfg.days - 3))
        duration = int(rng.choice([1, 1, 1, 2, 3]))
        size = rng.uniform(0.10, 1.00)
        for k in range(duration):
            d = dates[start_idx + k]
            extra = float(svc_daily.loc[(s, d)] * size * rng.uniform(0.85, 1.15))
            labels.append((event_id, d, s, team, extra))
    labels = pd.DataFrame(labels, columns=["event_id", "date", "service", "team", "extra_cost"])
    # Overlapping draws on the same service-day: keep the first event's label.
    labels = labels.drop_duplicates(["date", "service"], keep="first").reset_index(drop=True)

    extra = labels.groupby(["date", "service", "team"])["extra_cost"].sum()
    key = pd.MultiIndex.from_frame(usage[["date", "service", "team"]])
    usage["cost"] = usage["cost"] + extra.reindex(key).fillna(0.0).to_numpy()

    unit = usage["service"].map({s: p["unit_price"] for s, p in SERVICES.items()})
    usage["usage_hours"] = (usage["cost"] / unit * rng.uniform(0.95, 1.05, len(usage))).round(1)
    usage["cost"] = usage["cost"].round(2)
    usage = usage[["date", "service", "team", "region", "usage_hours", "cost"]]
    usage = usage.sort_values(["date", "service", "team"]).reset_index(drop=True)
    labels["extra_cost"] = labels["extra_cost"].round(2)
    return usage, labels


def service_daily(usage: pd.DataFrame) -> pd.DataFrame:
    """Total cost per service per day, in long format (date, service, cost)."""
    out = usage.groupby(["service", "date"], as_index=False)["cost"].sum()
    return out.sort_values(["service", "date"]).reset_index(drop=True)


def total_daily(usage: pd.DataFrame) -> pd.Series:
    """Total cost per day, as a daily-frequency series."""
    s = usage.groupby("date")["cost"].sum().sort_index()
    s.index = pd.DatetimeIndex(s.index, freq="D")
    return s
