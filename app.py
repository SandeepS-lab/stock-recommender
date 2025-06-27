import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------------------
# Constants and Configurations
# ----------------------------
LOOKBACK_PERIOD = "3y"
INTERVAL = "1wk"
BENCHMARK_TICKER = "^NSEI"  # Nifty 50
RISK_FREE_RATE_ANNUAL = 0.04

RISK_SCORE_THRESHOLDS = {
    "Conservative": 0.8,
    "Moderate": 1.8,
    "Aggressive": 100  # No upper limit
}

INDIAN_STOCK_TICKERS = [
    'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'ICICIBANK.NS',
    'LT.NS', 'ITC.NS', 'SBIN.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'MARUTI.NS', 'ULTRACEMCO.NS', 'BHARTIARTL.NS',
    'ADANIENT.NS', 'BAJFINANCE.NS', 'IRCTC.NS', 'ZOMATO.NS',
    'DMART.NS', 'NYKAA.NS', 'PAYTM.NS'
]

# ----------------------------
# Risk Profiling Logic
# ----------------------------
def get_risk_profile(age, income, dependents, qualification, duration, investment_type, volatility_comfort):
    score = 0
    weights = {
        "age": 0.20,
        "income": 0.20,
        "dependents": 0.10,
        "qualification": 0.05,
        "duration": 0.25,
        "investment_type": 0.10,
        "volatility_comfort": 0.10
    }

    if age < 30:
        score += 2 * weights["age"]
    elif age < 45:
        score += 1 * weights["age"]

    if income > 150000:
        score += 2 * weights["income"]
    elif income > 75000:
        score += 1 * weights["income"]

    if dependents >= 3:
        score -= 1 * weights["dependents"]
    elif dependents == 2:
        score -= 0.5 * weights["dependents"]

    if qualification in ["Postgraduate", "Professional"]:
        score += 1 * weights["qualification"]

    if duration >= 10:
        score += 2 * weights["duration"]
    elif duration >= 5:
        score += 1 * weights["duration"]

    if investment_type == "SIP":
        score += 0.5 * weights["investment_type"]

    score += (volatility_comfort - 1) * 0.5 * weights["volatility_comfort"]

    if score <= RISK_SCORE_THRESHOLDS["Conservative"]:
        return "Conservative"
    elif score <= RISK_SCORE_THRESHOLDS["Moderate"]:
        return "Moderate"
    else:
        return "Aggressive"

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered", initial_sidebar_state="expanded")
st.title("📋 AI-Based Stock Recommender for Fund Managers")

st.markdown("""
This intelligent assistant recommends stock allocations based on a client's risk profile using dynamically calculated financial metrics.

**Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. Investment in securities markets are subject to market risks, read all the related documents carefully before investing.
""")

# ----------------------------
# Sidebar: Input for Risk Profiling
# ----------------------------

st.sidebar.header("Client Risk Profile")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=35)
income = st.sidebar.number_input("Monthly Income (INR)", min_value=10000, step=5000, value=75000)
dependents = st.sidebar.selectbox("Number of Dependents", options=[0, 1, 2, 3, 4])
qualification = st.sidebar.selectbox("Highest Qualification", options=["Undergraduate", "Postgraduate", "Professional", "Other"])
duration = st.sidebar.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.radio("Investment Type", ["SIP", "Lumpsum"])
volatility_comfort = st.sidebar.slider("Comfort with Volatility", 1, 5, 3)

if st.sidebar.button("Generate Risk Profile"):
    profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type, volatility_comfort)
    st.success(f"✅ Based on the inputs, the client's risk profile is: **{profile}**")

    # TODO: Plug in logic to fetch stock data, compute Sharpe ratios, etc.
    st.info("Next Step: Use this risk profile to filter and recommend appropriate stocks (based on risk-adjusted return, volatility, etc.)")
