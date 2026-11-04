"""Anomaly detectors for daily cost per service.

Every detector takes a long frame with columns (date, service, cost), one row per
service per day, and returns a copy with two extra columns:
  score : higher means more unusual
  flag  : True if the detector raises an alert for that service-day
Only upward spikes are flagged; a cost drop is not a cost incident.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import STL

MAD_SCALE = 1.4826  # makes the median absolute deviation comparable to a std dev


def _per_service(sd: pd.DataFrame, fn) -> pd.DataFrame:
    sd = sd.sort_values(["service", "date"]).reset_index(drop=True)
    parts = [fn(g.copy()) for _, g in sd.groupby("service", sort=False)]
    return pd.concat(parts).sort_values(["service", "date"]).reset_index(drop=True)


def _robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * MAD_SCALE
    return (x - med) / (mad if mad > 0 else 1e-9)


# ---------------------------------------------------------------- baseline
def static_threshold(sd: pd.DataFrame, k: float = 2.0) -> pd.DataFrame:
    """The original rule: alert when cost > mean + k * std over the whole history."""
    def fn(g):
        mu, sigma = g["cost"].mean(), g["cost"].std()
        g["score"] = (g["cost"] - mu) / sigma
        g["flag"] = g["score"] > k
        return g
    return _per_service(sd, fn)


# ---------------------------------------------------------------- rolling
def weekday_log_ratio(cost: pd.Series, weeks: int = 4) -> pd.Series:
    """log(cost) minus the median log(cost) of the same weekday over the previous `weeks` weeks.

    Uses only past data, so it can run as a live alert.
    """
    logc = np.log(cost.clip(lower=1e-6).reset_index(drop=True))
    lags = pd.concat([logc.shift(7 * w) for w in range(1, weeks + 1)], axis=1)
    return (logc - lags.median(axis=1)).set_axis(cost.index)


def rolling_zscore(sd: pd.DataFrame, k: float = 4.0, weeks: int = 4, scale_window: int = 56) -> pd.DataFrame:
    """Weekday-aware rolling z-score per service.

    The residual is today's log cost against the same weekday in recent weeks; it is
    scaled by the robust spread (MAD) of residuals over the trailing `scale_window` days.
    Growth and weekly cycles cancel out, so the threshold adapts as the bill grows.
    """
    def fn(g):
        r = weekday_log_ratio(g["cost"], weeks)
        past = r.shift(1).rolling(scale_window, min_periods=14)
        med = past.median()
        mad = past.apply(lambda w: np.nanmedian(np.abs(w - np.nanmedian(w))), raw=True) * MAD_SCALE
        g["score"] = ((r - med) / mad.clip(lower=1e-3)).fillna(0.0)
        g["flag"] = g["score"] > k
        return g
    return _per_service(sd, fn)


# ---------------------------------------------------------------- seasonal decomposition
def stl_residual_z(cost: pd.Series, period: int = 7) -> pd.Series:
    res = STL(np.log(cost.clip(lower=1e-6)).to_numpy(), period=period, robust=True).fit()
    return _robust_z(pd.Series(res.resid, index=cost.index))


def stl_detector(sd: pd.DataFrame, k: float = 8.0, period: int = 7) -> pd.DataFrame:
    """Seasonal-trend decomposition (STL) per service; alert on large robust residuals.

    Uses the whole history (batch analysis, e.g. a monthly cost review).
    """
    def fn(g):
        g["score"] = stl_residual_z(g["cost"], period)
        g["flag"] = g["score"] > k
        return g
    return _per_service(sd, fn)


# ---------------------------------------------------------------- isolation forest
def build_features(sd: pd.DataFrame) -> pd.DataFrame:
    """Per service-day features that describe how unusual a day looks."""
    def fn(g):
        c = g["cost"]
        logc = np.log(c.clip(lower=1e-6))
        g["f_weekday_ratio"] = weekday_log_ratio(c).fillna(0.0)
        g["f_week_ratio"] = (logc - logc.shift(1).rolling(7, min_periods=3).median()).fillna(0.0)
        g["f_day_change"] = logc.diff().fillna(0.0)
        g["f_stl_z"] = stl_residual_z(c)
        return g
    return _per_service(sd, fn)


FEATURES = ["f_weekday_ratio", "f_week_ratio", "f_day_change", "f_stl_z"]


def isolation_forest(sd: pd.DataFrame, contamination: float = 0.025, seed: int = 0) -> pd.DataFrame:
    """scikit-learn IsolationForest over engineered features, pooled across services.

    `contamination` is the alert budget: the share of service-days the model may flag.
    Features are standardised per service so one model works for small and large services.
    """
    f = build_features(sd)
    X = f.groupby("service")[FEATURES].transform(lambda x: (x - x.median()) / (x.std() + 1e-9))
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(X)
    f["score"] = -model.score_samples(X)
    upward = f["f_weekday_ratio"] > 0
    f["flag"] = (model.predict(X) == -1) & upward
    return f.drop(columns=FEATURES)


DETECTORS = {
    "Static mean + 2σ (original)": static_threshold,
    "Rolling z-score (weekday-aware)": rolling_zscore,
    "STL seasonal decomposition": stl_detector,
    "IsolationForest": isolation_forest,
}
