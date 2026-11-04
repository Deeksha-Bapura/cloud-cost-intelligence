import numpy as np
import pandas as pd
import pytest

from analytics.forecasting import (
    FORECASTERS, backtest, clean_spikes, holt_winters, mape, moving_average, seasonal_naive,
)


def weekly_series(days=200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=days, freq="D")
    weekly = np.where(idx.dayofweek >= 5, 0.6, 1.0)
    return pd.Series(1000 * weekly * np.linspace(1, 1.4, days) * rng.lognormal(0, 0.02, days), index=idx)


def test_mape():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert mape([50], [50]) == 0.0


@pytest.mark.parametrize("name", list(FORECASTERS))
def test_forecasters_return_horizon_days_after_history(name):
    s = weekly_series()
    pred = FORECASTERS[name](s, 14)
    assert len(pred) == 14
    assert pred.index[0] == s.index[-1] + pd.Timedelta(days=1)
    assert np.isfinite(pred).all() and (pred > 0).all()


def test_baselines():
    s = weekly_series()
    assert np.allclose(moving_average(s, 3), s.iloc[-7:].mean())
    assert np.allclose(seasonal_naive(s, 7).to_numpy(), s.iloc[-7:].to_numpy())


def test_holt_winters_beats_moving_average_on_seasonal_data():
    s = weekly_series()
    hw = np.mean(backtest(s, holt_winters, 30, 2))
    ma = np.mean(backtest(s, moving_average, 30, 2))
    assert hw < ma / 2


def test_backtest_uses_only_past_data():
    s = weekly_series()
    seen = []

    def spy(train, horizon):
        seen.append(train.index[-1])
        return moving_average(train, horizon)

    backtest(s, spy, horizon=30, folds=3)
    assert seen == [s.index[-31], s.index[-61], s.index[-91]]


def test_clean_spikes_removes_one_off_spike():
    s = weekly_series()
    s.iloc[150] *= 3
    cleaned = clean_spikes(s)
    assert cleaned.iloc[150] < s.iloc[150] / 2
    assert (cleaned.drop(s.index[150]) == s.drop(s.index[150])).mean() > 0.95
