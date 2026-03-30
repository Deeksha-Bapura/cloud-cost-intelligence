import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import time

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
@st.cache_data
def load_data():
    url = "https://cloud-cost-intelligence.onrender.com/costs"
    data = requests.get(url).json()
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df

df = load_data()

# -------- FILTERS --------
date_range = st.sidebar.date_input(
    "📅 Date Range",
    [df["date"].min(), df["date"].max()]
)

services = st.sidebar.multiselect(
    "Service",
    df["service"].unique(),
    default=df["service"].unique()
)

teams = st.sidebar.multiselect(
    "Team",
    df["team"].unique(),
    default=df["team"].unique()
)

df_filtered = df[
    (df["date"] >= pd.to_datetime(date_range[0])) &
    (df["date"] <= pd.to_datetime(date_range[1])) &
    (df["service"].isin(services)) &
    (df["team"].isin(teams))
]

# -------- CALCULATIONS --------
total_cost = df_filtered["cost"].sum()
avg_cost = df_filtered.groupby("date")["cost"].sum().mean()
max_cost = df_filtered["cost"].max()

daily_cost = df_filtered.groupby("date")["cost"].sum().reset_index()

daily_cost["z_score"] = (
    (daily_cost["cost"] - daily_cost["cost"].mean()) /
    daily_cost["cost"].std()
)

anomalies = daily_cost[daily_cost["z_score"].abs() > 2]

# -------- EXECUTIVE SUMMARY --------
st.markdown("### 🧠 Executive Summary")

if not anomalies.empty:
    st.warning("Cloud costs show unusual spikes. Investigation recommended.")
else:
    st.success("Cloud spending is stable.")

st.info(f"Total spend is ${total_cost:,.2f} with an average daily cost of ${avg_cost:,.2f}.")

# -------- TABS --------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Trends", "🚨 Anomalies", "🔮 Forecast"])

# -------- OVERVIEW --------
with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Total Cost", f"${total_cost:,.2f}")
    col2.metric("📈 Avg Daily Cost", f"${avg_cost:,.2f}")
    col3.metric("🚨 Max Spike", f"${max_cost:,.2f}")

# -------- TRENDS --------
with tab2:
    fig1 = px.line(
        daily_cost,
        x="date",
        y="cost",
        markers=True,
        color_discrete_sequence=[primary_color]
    )

    fig1.update_layout(
        template=plot_template,
        title="Daily Cost Trend",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color)
    )

    fig1.update_xaxes(showgrid=False, tickfont=dict(color=text_color))
    fig1.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickfont=dict(color=text_color))

    st.plotly_chart(fig1, use_container_width=True)

# -------- ANOMALIES --------
with tab3:
    fig4 = px.line(
        daily_cost,
        x="date",
        y="cost",
        color_discrete_sequence=[primary_color]
    )

    fig4.add_scatter(
        x=anomalies["date"],
        y=anomalies["cost"],
        mode="markers",
        marker=dict(color=anomaly_color, size=10),
        name="Anomalies"
    )

    fig4.update_layout(
        template=plot_template,
        title="Anomaly Detection",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color)
    )

    fig4.update_xaxes(showgrid=False, tickfont=dict(color=text_color))
    fig4.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickfont=dict(color=text_color))

    st.plotly_chart(fig4, use_container_width=True)

# -------- FORECAST --------
with tab4:
    forecast_df = daily_cost.copy().sort_values("date")
    forecast_df["forecast"] = forecast_df["cost"].rolling(window=7).mean()

    fig5 = px.line(
        forecast_df,
        x="date",
        y=["cost", "forecast"],
        color_discrete_sequence=[primary_color, secondary_color]
    )

    fig5.update_layout(
        template=plot_template,
        title="Forecast vs Actual",
        title_x=0.5,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=text_color)
    )

    fig5.update_xaxes(showgrid=False, tickfont=dict(color=text_color))
    fig5.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickfont=dict(color=text_color))

    st.plotly_chart(fig5, use_container_width=True)

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