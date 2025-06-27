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

# Set safe locale encoding
locale.setlocale(locale.LC_ALL, '')
if sys.platform == "win32":
    import os
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
    'TCS': 'TCS.NS', 'HDFC Bank': 'HDFCBANK.NS', 'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS', 'Zomato': 'ZOMATO.NS',
    'Reliance Industries': 'RELIANCE.NS', 'Bajaj Finance': 'BAJFINANCE.NS', 'IRCTC': 'IRCTC.NS'
}

stock_risk = {
    'TCS': 'Conservative', 'HDFC Bank': 'Moderate', 'Infosys': 'Moderate'
