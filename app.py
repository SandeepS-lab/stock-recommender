import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # Import matplotlib for plotting

# ----------------------------
# Constants and Configurations
# ----------------------------
LOOKBACK_PERIOD = "3y" # For historical data (e.g., 1 year, 3 years)
INTERVAL = "1wk"       # Data interval (e.g., "1d", "1wk", "1mo")
BENCHMARK_TICKER = "^NSEI" # Nifty 50 for Indian context
RISK_FREE_RATE_ANNUAL = 0.04 # Example risk-free rate (e.g., current FD rate)

# Map for risk scores to categories (adjust these based on your weighted scoring)
RISK_SCORE_THRESHOLDS = {
    "Conservative": 0.8,
    "Moderate": 1.8,
    "Aggressive": 100 # Effectively no upper limit for aggressive
}

# Example Indian Stock Tickers (Expand this list significantly in real app)
INDIAN_STOCK_TICKERS = [
    'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'ICICIBANK.NS',
    'LT.NS', 'ITC.NS', 'SBIN.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'MARUTI.NS', 'ULTRACEMCO.NS', 'BHARTIARTL.NS',
    'ADANIENT.NS', 'BAJFINANCE.NS', 'IRCTC.NS', 'ZOMATO.NS',
    'DMART.NS', 'NYKAA.NS', 'PAYTM.NS' # More volatile/growth examples
]

# ----------------------------
# Risk Profiling Logic
# ----------------------------

def get_risk_profile(age, income, dependents, qualification, duration, investment_type, volatility_comfort):
    score = 0
    # Assign weights to factors - calibrated for more granular impact
    weights = {
        "age": 0.20,
        "income": 0.20,
        "dependents": 0.10,
        "qualification": 0.05,
        "duration": 0.25,
        "investment_type": 0.10,
        "volatility_comfort": 0.10 # New factor
    }

    # Age: Younger can take more risk
    if age < 30:
        score += 2 * weights["age"]
    elif age < 45:
        score += 1 * weights["age"]
    # 45+ gives 0 from age perspective

    # Income: Higher income, higher risk capacity
    if income > 150000: # Higher bracket
        score += 2 * weights["income"]
    elif income > 75000:
        score += 1 * weights["income"]

    # Dependents: More dependents, less risk capacity
    if dependents >= 3:
        score -= 1 * weights["dependents"]
    elif dependents == 2:
        score -= 0.5 * weights["dependents"]

    # Qualification: Indicates financial literacy/understanding of risk
    if qualification in ["Postgraduate", "Professional"]:
        score += 1 * weights["qualification"]

    # Duration: Longer duration, more capacity for risk
    if duration >= 10:
        score += 2 * weights["duration"]
    elif duration >= 5:
        score += 1 * weights["duration"]

    # Investment Type: SIP implies averaged entry, slightly lower risk profile
    if investment_type == "SIP":
        score += 0.5 * weights["investment_type"] # Small positive for SIP, as it smooths out volatility

    # Volatility Comfort: Direct measure of psychological risk tolerance
    # Scale 1 (Very Uncomfortable) to 5 (Very Comfortable)
    score += (volatility_comfort - 1) * 0.5 * weights["volatility_comfort"] # Scale 0 to 2 for weight

    # Determine risk profile based on total score
    if score <= RISK_SCORE_THRESHOLDS["Conservative"]:
        return "Conservative"
    elif score <= RISK_SCORE_THRESHOLDS["Moderate"]:
        return "Moderate"
    else:
        return "Aggressive"

# [Other code remains unchanged]

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered", initial_sidebar_state="expanded")
st.title("\ud83d\udcbc AI-Based Stock Recommender for Fund Managers")
st.markdown("""
This intelligent assistant recommends stock allocations based on a client's risk profile using dynamically calculated financial metrics.

**Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. Investment in securities markets are subject to market risks, read all the related documents carefully before investing.
""")
