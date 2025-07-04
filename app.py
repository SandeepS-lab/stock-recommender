import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Ticker map (15 stocks)
TICKER_MAP = { ... }  # same as before

def fetch_live_data(stock_df):
    ...  # unchanged

def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    ...  # unchanged

def get_stock_list(risk_profile, investment_amount, diversify=False):
    ...  # unchanged

def simulate_earnings(amount, years):
    ...  # unchanged

def monte_carlo_simulation(...):
    ...  # unchanged

# Streamlit UI start
st.title("📈 AI-Based Stock Recommender for Fund Managers")
# Sidebar inputs including 'diversify' checkbox
# ...

if st.button("Generate Recommendation"):
    # Generate portfolio, live data, projections, Monte Carlo

    st.subheader("📉 Portfolio Backtest (Last 24 Months)")
    portfolio_weights = recommended_stocks.set_index("Stock")["Weight %"] / 100
    selected_tickers = [TICKER_MAP[s] for s in portfolio_weights.index]
    
    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)  # 24 months back
    price_data = yf.download(selected_tickers, start=start_date, end=end_date)['Close']
    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data.droplevel(0, axis=1)
    
    expected_days = 252 * 2  # Business days approximation
    valid = [col for col in price_data.columns if price_data[col].dropna().shape[0] >= expected_days]
    price_data = price_data[valid]
    
    # Sync portfolio_weights to valid tickers
    valid_names = [s for s in portfolio_weights.index if TICKER_MAP[s] in valid]
    pw = portfolio_weights[valid_names]
    price_data = price_data[pw.index.map(TICKER_MAP)]
    
    st.write("✅ Backtesting using", len(valid), "stocks:", valid)
    st.write("Date range:", price_data.index.min().date(), "to", price_data.index.max().date())

    # Portfolio and market returns
    normalized = price_data / price_data.iloc[0]
    port_ret = (normalized * pw.values).sum(axis=1)
    market_ret = normalized.mean(axis=1)
    ...
    
    st.line_chart(pd.DataFrame({"Portfolio": port_ret, "Market Avg": market_ret}))
    # Performance metrics displayed

if st.checkbox("📜 Show Historical Stock Data (Last 3 Months)"):
    ...  # unchanged
