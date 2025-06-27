import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import sys
import locale
import os

# ----------------------------
# Safe UTF-8 setup
# ----------------------------
locale.setlocale(locale.LC_ALL, '')
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# ----------------------------
# Risk Profiling Logic
# ----------------------------
def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    score = 0
    if age < 30: score += 2
    elif age < 45: score += 1
    if income > 100000: score += 2
    elif income > 50000: score += 1
    if dependents >= 3: score -= 1
    if qualification in ["Postgraduate", "Professional"]: score += 1
    if duration >= 5: score += 1
    if investment_type == "SIP": score += 1

    if score <= 2:
        return "Conservative"
    elif score <= 5:
        return "Moderate"
    else:
        return "Aggressive"

# ----------------------------
# Live Stock Data (YFinance)
# ----------------------------
def safe_str(val):
    try:
        return str(val).encode('utf-8', 'ignore').decode('utf-8')
    except:
        return str(val)

def get_live_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'Price': float(info.get('currentPrice') or 0),
            '52 Week High': float(info.get('fiftyTwoWeekHigh') or 0),
            '52 Week Low': float(info.get('fiftyTwoWeekLow') or 0),
            'PE Ratio': float(info.get('trailingPE') or 0),
            'Dividend Yield': float(info.get('dividendYield') or 0),
            'Beta': float(info.get('beta') or 0)
        }
    except Exception as e:
        return {
            'Price': 0,
            '52 Week High': 0,
            '52 Week Low': 0,
            'PE Ratio': 0,
            'Dividend Yield': 0,
            'Beta': 0,
            'Error': safe_str(e)
        }

# ----------------------------
# Static Data Table
# ----------------------------
stock_mapping = {
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Zomato': 'ZOMATO.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'IRCTC': 'IRCTC.NS'
}

stock_risk = {
    'TCS': 'Conservative',
    'HDFC Bank': 'Moderate',
    'Infosys': 'Moderate',
    'Adani Enterprises': 'Aggressive',
    'Zomato': 'Aggressive',
    'Reliance Industries': 'Moderate',
    'Bajaj Finance': 'Moderate',
    'IRCTC': 'Aggressive'
}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender with Live Data")

def format_currency(val):
    return f"Rs. {val:,.0f}"

st.subheader("Enter Client Profile")
age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (Rs.)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (Rs.)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
live_data_toggle = st.checkbox("Use Live Data from YFinance")

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.write(f"**Risk Profile:** {risk_profile}")
    st.write(f"**Investment Allocation:** {format_currency(investment_amount)}")

    filtered_stocks = [s for s, r in stock_risk.items() if r == risk_profile]
    if len(filtered_stocks) < 5:
        all_stocks = list(stock_risk.keys())
        for s in all_stocks:
            if s not in filtered_stocks:
                filtered_stocks.append(s)
            if len(filtered_stocks) == 5:
                break

    yf_symbols = [stock_mapping[s] for s in filtered_stocks]
    raw_data = yf.download(yf_symbols, period="1y", interval="1d")
    prices = raw_data['Adj Close'] if 'Adj Close' in raw_data else raw_data['Close']
    prices = prices.dropna()
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    optimized_weights = ef.max_
