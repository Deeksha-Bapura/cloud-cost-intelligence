from dataclasses import replace

import pandas as pd

from analytics.data import SERVICES, TEAMS, Config, generate, service_daily, total_daily

CFG = replace(Config(), days=120, n_anomalies=8, seed=7)


def test_shape_and_columns():
    usage, labels = generate(CFG)
    assert list(usage.columns) == ["date", "service", "team", "region", "usage_hours", "cost"]
    assert len(usage) == CFG.days * len(SERVICES) * len(TEAMS)
    assert (usage["cost"] > 0).all()
    assert usage["date"].nunique() == CFG.days


def test_same_seed_same_data():
    a, la = generate(CFG)
    b, lb = generate(CFG)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(la, lb)


def test_planted_anomalies_are_in_range_and_after_warmup():
    usage, labels = generate(CFG)
    first = usage["date"].min()
    assert labels["event_id"].nunique() >= CFG.n_anomalies - 1  # an overlap may merge two
    assert (labels["date"] >= first + pd.Timedelta(days=CFG.warmup_days)).all()
    assert (labels["extra_cost"] > 0).all()


def test_weekends_cost_less_than_weekdays():
    usage, _ = generate(CFG)
    daily = total_daily(usage)
    weekend = daily[daily.index.dayofweek >= 5].mean()
    weekday = daily[daily.index.dayofweek < 5].mean()
    assert weekend < 0.9 * weekday


def test_service_daily_sums_match():
    usage, _ = generate(CFG)
    sd = service_daily(usage)
    assert len(sd) == CFG.days * len(SERVICES)
    assert abs(sd["cost"].sum() - usage["cost"].sum()) < 1e-6
