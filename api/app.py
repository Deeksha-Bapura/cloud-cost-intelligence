from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load data
df = pd.read_csv("data/cloud_usage.csv")
df["date"] = pd.to_datetime(df["date"])

@app.route("/")
def home():
    return "Cloud FinOps API is running"

# Get all cost data
@app.route("/costs")
def get_costs():
    return jsonify(df.to_dict(orient="records"))

# Cost by service
@app.route("/costs/service")
def cost_by_service():
    result = df.groupby("service")["cost"].sum().reset_index()
    return jsonify(result.to_dict(orient="records"))

# Anomalies
@app.route("/anomalies")
def anomalies():
    daily = df.groupby("date")["cost"].sum().reset_index()
    threshold = daily["cost"].mean() + 2 * daily["cost"].std()
    result = daily[daily["cost"] > threshold]
    return jsonify(result.to_dict(orient="records"))

# Forecast
@app.route("/forecast")
def forecast():
    daily = df.groupby("date")["cost"].sum().reset_index()
    daily = daily.sort_values("date")
    daily["forecast"] = daily["cost"].rolling(window=7).mean()
    return jsonify(daily.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)