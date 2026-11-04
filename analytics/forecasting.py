"""Daily cost forecasting and backtesting."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def _future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def moving_average(train: pd.Series, horizon: int, window: int = 7) -> pd.Series:
    """The original approach: next days = average of the last `window` days."""
    return pd.Series(train.iloc[-window:].mean(), index=_future_index(train, horizon))


def seasonal_naive(train: pd.Series, horizon: int, period: int = 7) -> pd.Series:
    """Repeat the last observed week."""
    last = train.iloc[-period:].to_numpy()
    return pd.Series(np.resize(last, horizon), index=_future_index(train, horizon))


def holt_winters(train: pd.Series, horizon: int, period: int = 7) -> pd.Series:
    """Holt-Winters: damped additive trend + multiplicative weekly seasonality."""
    y = pd.Series(train.to_numpy(), index=pd.DatetimeIndex(train.index, freq="D"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=period,
            initialization_method="estimated",
        ).fit(optimized=True)
    return pd.Series(fit.forecast(horizon).to_numpy(), index=_future_index(train, horizon))


def clean_spikes(train: pd.Series, k: float = 3.5) -> pd.Series:
    """Replace spike days with the STL trend+season fit, so one-off incidents don't skew the model."""
    from .detection import stl_residual_z
    from statsmodels.tsa.seasonal import STL

    z = stl_residual_z(train)
    fit = STL(np.log(train.to_numpy()), period=7, robust=True).fit()
    expected = pd.Series(np.exp(fit.trend + fit.seasonal), index=train.index)
    return train.where(z <= k, expected)


def holt_winters_clean(train: pd.Series, horizon: int) -> pd.Series:
    """Holt-Winters trained on history with detected spikes removed."""
    return holt_winters(clean_spikes(train), horizon)


FORECASTERS = {
    "7-day moving average (original)": moving_average,
    "Seasonal naive (last week)": seasonal_naive,
    "Holt-Winters": holt_winters,
    "Holt-Winters + spike cleaning": holt_winters_clean,
}


def mape(actual, predicted) -> float:
    """Mean absolute percentage error, in percent."""
    a, p = np.asarray(actual, float), np.asarray(predicted, float)
    return float(np.mean(np.abs((a - p) / a)) * 100)


def backtest(series: pd.Series, model, horizon: int = 30, folds: int = 3) -> list[float]:
    """Rolling-origin backtest: hide the last `horizon` days, forecast them, score; then
    step the cutoff back by `horizon` days and repeat. Returns one MAPE per fold, newest first."""
    scores = []
    for f in range(folds):
        cut = len(series) - horizon * (f + 1)
        train, test = series.iloc[:cut], series.iloc[cut:cut + horizon]
        scores.append(mape(test, model(train, horizon)))
    return scores
