import json
import os
import sys
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from analytics.data import service_daily  # noqa: E402
from analytics.detection import (  # noqa: E402
    isolation_forest, rolling_zscore, static_threshold, stl_detector,
)
from analytics.forecasting import FORECASTERS, backtest, holt_winters  # noqa: E402

# Shown in the sidebar picker, best first.
DETECTORS = {
    "IsolationForest": isolation_forest,
    "STL seasonal decomposition": stl_detector,
    "Rolling z-score (weekday-aware)": rolling_zscore,
    "Static mean + 2σ (original)": static_threshold,
}

st.set_page_config(page_title="Cloud Cost Intelligence Platform", layout="wide")

# -------- SIDEBAR --------
st.sidebar.header("🔍 Filters")
theme = st.sidebar.radio("🎨 Theme", ["Dark", "Light"])

# -------- THEME CONFIG --------
if theme == "Dark":
    background = "#0E1117"
    text_color = "#FFFFFF"
    card_bg = "#262730"
    sidebar_bg = "#1c1f26"
    plot_template = "plotly_dark"
    primary_color = "#4CAF50"
    secondary_color = "#2196F3"
    anomaly_color = "#FF5252"
    button_color = "#4CAF50"
else:
    background = "#FFFFFF"
    text_color = "#000000"
    card_bg = "#F5F5F5"
    sidebar_bg = "#f0f2f6"
    plot_template = "plotly_white"
    primary_color = "#2E7D32"
    secondary_color = "#1565C0"
    anomaly_color = "#D32F2F"
    button_color = "#2E7D32"

# -------- CSS --------
st.markdown(f"""
<style>
* {{
    transition: all 0.3s ease-in-out !important;
}}

.stApp {{
    background-color: {background};
}}

header {{
    background-color: transparent !important;
}}

.block-container {{
    padding-top: 2rem !important;
}}

section[data-testid="stSidebar"] {{
    background-color: {sidebar_bg};
    border-right: 1px solid rgba(128,128,128,0.2);
}}

/* TEXT */
h1, h2, h3, h4, h5, h6 {{
    color: {text_color} !important;
}}

p, label {{
    color: {text_color} !important;
}}

body, span, div {{
    color: {text_color};
}}

/* INPUT FIX */
input, textarea {{
    color: {text_color} !important;
}}

[data-baseweb="input"] input {{
    color: {text_color} !important;
}}

[data-baseweb="tag"] {{
    color: white !important;
}}

[data-baseweb="input"] {{
    background-color: {card_bg} !important;
    border-radius: 8px;
}}

/* METRICS */
.stMetric {{
    background-color: {card_bg};
    padding: 15px;
    border-radius: 12px;
}}

/* TABS */
.stTabs [role="tab"] {{
    border-radius: 8px;
    padding: 6px 12px;
}}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {{
    background-color: {button_color} !important;
    color: white !important;
    border-radius: 8px;
    padding: 8px 16px;
    border: none;
}}

</style>
""", unsafe_allow_html=True)

# -------- AUTO REFRESH --------
st.sidebar.markdown("### 🔄 Auto Refresh")
refresh = st.sidebar.checkbox("Enable Auto Refresh")

# -------- HEADER --------
st.title("☁️ Cloud Cost Intelligence Platform")
st.markdown("Monitor, analyze, and optimize cloud spending")

# -------- LOAD DATA --------
# Reads the bundled CSV by default, so the dashboard runs anywhere (including
# Streamlit Community Cloud). Set COST_API_URL to read from the Flask API instead.
@st.cache_data
def load_data():
    api_url = os.environ.get("COST_API_URL")
    if api_url:
        df = pd.DataFrame(requests.get(f"{api_url.rstrip('/')}/costs", timeout=30).json())
    else:
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "cloud_usage.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_labels():
    path = os.path.join(BASE_DIR, "data", "planted_anomalies.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["event_id", "date", "service", "team", "extra_cost"])
    return pd.read_csv(path, parse_dates=["date"])


@st.cache_data
def load_results():
    path = os.path.join(BASE_DIR, "reports", "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


@st.cache_data
def run_detector(df, method):
    return DETECTORS[method](service_daily(df))


@st.cache_data
def run_forecast(series, horizon):
    return holt_winters(series, horizon)


@st.cache_data
def run_backtest(series):
    return {name: backtest(series, FORECASTERS[name], 30, 1)[0]
            for name in ["7-day moving average (original)", "Holt-Winters"]}


df = load_data()
labels = load_labels()

# -------- FILTERS --------
date_range = st.sidebar.date_input(
    "📅 Date Range",
    [df["date"].min(), df["date"].max()]
)

services = st.sidebar.multiselect(
    "Service",
    sorted(df["service"].unique()),
    default=sorted(df["service"].unique())
)

teams = st.sidebar.multiselect(
    "Team",
    sorted(df["team"].unique()),
    default=sorted(df["team"].unique())
)

method = st.sidebar.selectbox("🚨 Anomaly detector", list(DETECTORS), index=0)

if len(date_range) != 2 or not services or not teams:
    st.info("Pick a start and end date, at least one service and one team.")
    st.stop()

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
in_range = (df["date"] >= start) & (df["date"] <= end)
df_scope = df[df["service"].isin(services) & df["team"].isin(teams)]  # full history, for models
df_filtered = df_scope[(df_scope["date"] >= start) & (df_scope["date"] <= end)]

if df_filtered.empty:
    st.info("No data for these filters.")
    st.stop()

# -------- CALCULATIONS --------
total_cost = df_filtered["cost"].sum()
avg_cost = df_filtered.groupby("date")["cost"].sum().mean()
daily_cost = df_filtered.groupby("date")["cost"].sum().reset_index()
max_day = daily_cost.loc[daily_cost["cost"].idxmax()]

# Detectors run on the full history of the selected services and teams (they need it
# to learn what "normal" looks like); results are then cut to the chosen date range.
detected = run_detector(df_scope, method)
detected = detected[(detected["date"] >= start) & (detected["date"] <= end)]
anomalies = detected[detected["flag"]].sort_values("score", ascending=False)

# -------- EXECUTIVE SUMMARY --------
st.markdown("### 🧠 Executive Summary")

if not anomalies.empty:
    worst = anomalies.iloc[0]
    st.warning(
        f"{len(anomalies)} unusual service-days found by {method}. "
        f"Largest: {worst['service']} on {worst['date']:%b %d, %Y} (\\${worst['cost']:,.2f})."
    )
else:
    st.success("Cloud spending is stable.")

st.info(f"Total spend is \\${total_cost:,.2f} with an average daily cost of \\${avg_cost:,.2f}.")

# -------- TABS --------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📈 Trends", "🚨 Anomalies", "🔮 Forecast", "🧪 Model Evaluation"]
)


def style(fig, title):
    fig.update_layout(
        template=plot_template,
        title=title,
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=text_color), title="")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickfont=dict(color=text_color), title="Cost ($)")
    return fig


# -------- OVERVIEW --------
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💰 Total Cost", f"${total_cost:,.2f}")
    col2.metric("📈 Avg Daily Cost", f"${avg_cost:,.2f}")
    col3.metric("🚨 Highest Day", f"${max_day['cost']:,.2f}", f"{max_day['date']:%b %d}", delta_color="off")
    col4.metric("🔎 Anomalies", f"{len(anomalies)}")

    by_service = df_filtered.groupby("service")["cost"].sum().sort_values().reset_index()
    fig0 = px.bar(by_service, x="cost", y="service", orientation="h",
                  color_discrete_sequence=[primary_color])
    fig0 = style(fig0, "Cost by Service")
    fig0.update_xaxes(title="Cost ($)")
    fig0.update_yaxes(title="")
    st.plotly_chart(fig0, use_container_width=True)

# -------- TRENDS --------
with tab2:
    by_svc_day = df_filtered.groupby(["date", "service"])["cost"].sum().reset_index()
    fig1 = px.area(by_svc_day, x="date", y="cost", color="service")
    st.plotly_chart(style(fig1, "Daily Cost by Service"), use_container_width=True)

# -------- ANOMALIES --------
with tab3:
    st.caption(
        "Each service is checked separately, day by day. The dataset has known, planted cost "
        "spikes, so you can see what each detector catches and misses."
    )
    svc_pick = st.selectbox("Service", services, key="anomaly_service")
    svc = detected[detected["service"] == svc_pick]
    svc_flags = svc[svc["flag"]]

    fig4 = px.line(svc, x="date", y="cost", color_discrete_sequence=[primary_color])
    fig4.data[0].name = "Daily cost"
    fig4.data[0].showlegend = True

    planted = labels[(labels["service"] == svc_pick) & labels["date"].between(start, end)]
    if not planted.empty:
        p = svc.merge(planted[["date"]], on="date")
        fig4.add_scatter(x=p["date"], y=p["cost"], mode="markers", name="Planted spike (truth)",
                         marker=dict(size=16, color="rgba(0,0,0,0)",
                                     line=dict(color=secondary_color, width=2)))
    fig4.add_scatter(x=svc_flags["date"], y=svc_flags["cost"], mode="markers",
                     marker=dict(color=anomaly_color, size=9), name=f"Flagged by {method}")
    st.plotly_chart(style(fig4, f"Anomaly Detection: {svc_pick}"), use_container_width=True)

    if not planted.empty:
        caught = planted["date"].isin(svc_flags["date"]).sum()
        false_alarms = (~svc_flags["date"].isin(planted["date"])).sum()
        c1, c2 = st.columns(2)
        c1.metric("Planted spike-days caught", f"{caught} / {len(planted)}")
        c2.metric("False alarms", f"{false_alarms}")

    st.markdown("#### All flagged service-days")
    table = anomalies[["date", "service", "cost", "score"]].copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    st.dataframe(table.round({"cost": 2, "score": 2}), use_container_width=True, hide_index=True)

# -------- FORECAST --------
with tab4:
    series = df_scope.groupby("date")["cost"].sum().sort_index()
    series.index = pd.DatetimeIndex(series.index, freq="D")
    horizon = st.slider("Days to forecast", 7, 60, 30)
    pred = run_forecast(series, horizon)

    recent = series.iloc[-90:]
    fc_df = pd.concat([
        pd.DataFrame({"date": recent.index, "cost": recent.to_numpy(), "series": "Actual"}),
        pd.DataFrame({"date": pred.index, "cost": pred.to_numpy(), "series": "Holt-Winters forecast"}),
    ])
    fig5 = px.line(fc_df, x="date", y="cost", color="series",
                   color_discrete_sequence=[primary_color, secondary_color])
    fig5.update_traces(selector=dict(name="Holt-Winters forecast"), line=dict(dash="dash"))
    st.plotly_chart(style(fig5, "Forecast vs Actual"), use_container_width=True)

    st.metric(f"Projected spend, next {horizon} days", f"${pred.sum():,.0f}")
    bt = run_backtest(series)
    c1, c2 = st.columns(2)
    c1.metric("Backtest error, Holt-Winters (MAPE)", f"{bt['Holt-Winters']:.1f}%")
    c2.metric("Backtest error, 7-day average (MAPE)", f"{bt['7-day moving average (original)']:.1f}%")
    st.caption("Backtest: the last 30 days were hidden, forecast from the history before them, "
               "and compared with what actually happened.")

# -------- MODEL EVALUATION --------
with tab5:
    results = load_results()
    if results is None:
        st.info("Run `python scripts/evaluate.py` to generate the benchmark.")
    else:
        st.markdown(
            f"Benchmark over {len(results['test_seeds'])} held-out synthetic datasets "
            f"({results['setup']['days']} days each, {results['setup']['anomalies_per_dataset']} planted "
            f"spikes per dataset). Thresholds were tuned on separate validation datasets."
        )
        det = pd.DataFrame(results["detection"])
        det_view = pd.DataFrame({
            "Method": det["method"],
            "Setting": det["param"],
            "Recall": (det["recall"] * 100).round(0).astype(int).astype(str) + "%",
            "Precision": (det["precision"] * 100).round(0).astype(int).astype(str) + "%",
            "F1": det["f1"].round(2),
            "False alarms / month": det["false_alarms_per_month"].round(1),
        })
        st.markdown("#### Anomaly detection")
        st.dataframe(det_view, use_container_width=True, hide_index=True)

        fig6 = px.bar(det, x="method", y="recall", color_discrete_sequence=[primary_color])
        fig6 = style(fig6, "Share of planted spikes caught")
        fig6.update_yaxes(title="Recall", tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig6, use_container_width=True)

        fc = pd.DataFrame(results["forecast"])
        st.markdown("#### Forecast backtest (30-day horizon)")
        st.dataframe(pd.DataFrame({
            "Method": fc["method"],
            "MAPE, last 30 days": fc["mape_last30"].round(1).astype(str) + "%",
            "MAPE, 3-fold average": fc["mape_mean"].round(1).astype(str) + "%",
        }), use_container_width=True, hide_index=True)

# -------- DOWNLOAD --------
st.sidebar.markdown("### 📥 Export Data")
st.sidebar.download_button(
    label="Download CSV",
    data=df_filtered.to_csv(index=False),
    file_name="cloud_cost_data.csv",
    mime="text/csv"
)

# -------- AUTO REFRESH --------
if refresh:
    time.sleep(5)
    st.rerun()
