import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# -----------------------------
# TICKER_MAP (No Zomato)
# -----------------------------
TICKER_MAP = {
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Eternal Limited': 'ETERNAL.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'IRCTC': 'IRCTC.NS'
}

# UI
st.title("📊 AI Stock Recommender")

uploaded_file = st.file_uploader("Upload your stock list CSV (optional)", type="csv")
if uploaded_file:
    user_df = pd.read_csv(uploaded_file)
    st.dataframe(user_df)
    stock_list = user_df['Stock'].dropna().tolist()
else:
    stock_list = list(TICKER_MAP.keys())

# Filter out ZOMATO if present
stock_list = [s for s in stock_list if 'ZOMATO' not in s.upper()]

# Fetch Live Data (demo purpose)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="3mo", auto_adjust=True)
        return df.tail(1)['Close'].values[0] if not df.empty else np.nan
    except:
        return np.nan

results = []
for stock in stock_list:
    ticker = TICKER_MAP.get(stock)
    if not ticker:
        continue
    price = fetch_data(ticker)
    results.append({'Stock': stock, 'Latest Price': price})
    st.write(f"{stock}: ₹{price}")

st.dataframe(pd.DataFrame(results))
