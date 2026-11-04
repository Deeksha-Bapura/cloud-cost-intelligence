import numpy as np
import pandas as pd
import pytest

from analytics.detection import (
    DETECTORS, isolation_forest, rolling_zscore, static_threshold, stl_detector,
)
from analytics.evaluation import score_detector


def seasonal_frame(days=140, spike_day=100, spike=1.0, seed=0):
    """One service with weekly seasonality, growth, noise and one spike of +`spike` x normal."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    weekly = np.where(dates.dayofweek >= 5, 0.6, 1.0)
    cost = 100 * weekly * np.linspace(1, 1.3, days) * rng.lognormal(0, 0.03, days)
    cost[spike_day] *= 1 + spike
    return pd.DataFrame({"date": dates, "service": "EC2", "cost": cost}), dates[spike_day]


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detectors_return_score_and_flag_for_every_row(name):
    sd, _ = seasonal_frame()
    out = DETECTORS[name](sd)
    assert len(out) == len(sd)
    assert {"score", "flag"} <= set(out.columns)
    assert out["flag"].dtype == bool


@pytest.mark.parametrize("fn", [rolling_zscore, stl_detector, isolation_forest])
def test_improved_detectors_catch_a_clear_spike(fn):
    sd, spike_date = seasonal_frame(spike=1.0)
    out = fn(sd)
    assert out.loc[out["date"] == spike_date, "flag"].item()


@pytest.mark.parametrize("fn", [rolling_zscore, stl_detector])
def test_seasonal_detectors_quiet_on_clean_data(fn):
    sd, _ = seasonal_frame(spike=0.0)
    assert fn(sd)["flag"].sum() <= 1


def test_static_threshold_misses_a_weekend_spike_that_seasonal_methods_catch():
    # A Saturday spike sits below weekday levels, so a single global threshold can't see it.
    sd, _ = seasonal_frame(spike=0.0)
    saturday = sd.index[sd["date"].dt.dayofweek == 5][14]
    sd.loc[saturday, "cost"] *= 1.5
    day = sd.loc[saturday, "date"]
    assert not static_threshold(sd).query("date == @day")["flag"].item()
    assert rolling_zscore(sd).query("date == @day")["flag"].item()
    assert stl_detector(sd).query("date == @day")["flag"].item()


def test_detectors_ignore_drops():
    sd, _ = seasonal_frame(spike=0.0)
    sd.loc[100, "cost"] *= 0.3  # a big drop is not a cost incident
    for fn in DETECTORS.values():
        assert not fn(sd).loc[100, "flag"]


def test_score_detector_counts():
    dates = pd.date_range("2026-01-01", periods=10)
    flags = pd.DataFrame({"date": dates, "service": "S3", "flag": False})
    flags.loc[[2, 3, 7], "flag"] = True  # day 2 hits event 0, day 7 hits nothing, day 3 hits event 0's 2nd day
    labels = pd.DataFrame({
        "event_id": [0, 0, 1],
        "date": [dates[2], dates[3], dates[5]],
        "service": ["S3", "S3", "S3"],
    })
    s = score_detector(flags, labels, days=10)
    assert s["events_caught"] == 1 and s["events_total"] == 2
    assert s["recall"] == 0.5
    assert s["false_alarms"] == 1
    assert s["precision"] == pytest.approx(2 / 3)
