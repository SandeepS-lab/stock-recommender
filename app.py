import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import re

# ASCII sanitization
def ascii_only(val):
    try:
        return re.sub(r'[^\x00-\x7F]+', '', str(val))
    except:
        return str(val)

# Risk Profile Logic
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

# Fetch live stock data
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
            'Error': ascii_only(e)
        }

# Stock symbols and risk categories
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

st.set_page_config(page_title="Stock Recommender", layout="centered")
st.title(ascii_only("Stock Recommender"))

# Form input
with st.form("input_form"):
    st.subheader(ascii_only("Client Details"))
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Monthly Income (Rs)", value=50000)
    investment_amount = st.number_input("Investment Amount (Rs)", value=100000)
    dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
    qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
    duration = st.slider("Investment Duration (Years)", 1, 30, 5)
    investment_type = st.radio("Investment Mode", ["Lumpsum", "SIP"])
    live_data_toggle = st.checkbox("Enable Live Data")
    submitted = st.form_submit_button("Recommend Portfolio")

if submitted:
    risk_profile = ascii_only(get_risk_profile(age, income, dependents, qualification, duration, investment_type))
    st.write(ascii_only("Risk Profile:"), risk_profile)
    st.write(ascii_only
