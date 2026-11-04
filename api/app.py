import os
import sys

import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from analytics.data import service_daily, total_daily  # noqa: E402
from analytics.detection import (  # noqa: E402
    isolation_forest, rolling_zscore, static_threshold, stl_detector,
)
from analytics.forecasting import holt_winters  # noqa: E402

app = Flask(__name__)

file_path = os.path.join(BASE_DIR, "data", "cloud_usage.csv")
df = pd.read_csv(file_path, parse_dates=["date"])

DETECTORS = {
    "isolation_forest": isolation_forest,
    "rolling_zscore": rolling_zscore,
    "stl": stl_detector,
    "static": static_threshold,
}


def _records(frame: pd.DataFrame):
    out = frame.copy()
    for col in out.select_dtypes(include="datetime").columns:
        out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


@app.route("/")
def home():
    return "Cloud FinOps API is running"


@app.route("/health")
def health():
    return jsonify({"status": "ok", "rows": int(len(df))})


# Get all cost data
@app.route("/costs")
def get_costs():
    return jsonify(_records(df))


# Cost by service
@app.route("/costs/service")
def cost_by_service():
    result = df.groupby("service")["cost"].sum().round(2).reset_index()
    return jsonify(_records(result))


# Anomalies per service per day. ?method=isolation_forest|rolling_zscore|stl|static
@app.route("/anomalies")
def anomalies():
    method = request.args.get("method", "isolation_forest")
    if method not in DETECTORS:
        return jsonify({"error": f"unknown method '{method}'", "methods": list(DETECTORS)}), 400
    result = DETECTORS[method](service_daily(df))
    flagged = result.loc[result["flag"], ["date", "service", "cost", "score"]].round({"cost": 2, "score": 2})
    return jsonify({"method": method, "anomalies": _records(flagged)})


# Forecast total daily cost with Holt-Winters. ?days=30
@app.route("/forecast")
def forecast():
    days = max(1, min(int(request.args.get("days", 30)), 90))
    series = total_daily(df)
    pred = holt_winters(series, days)
    out = pd.DataFrame({"date": pred.index, "forecast": pred.round(2).to_numpy()})
    return jsonify({"method": "holt_winters", "forecast": _records(out)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
