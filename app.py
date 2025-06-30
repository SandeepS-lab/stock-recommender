import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime

# ----------------------------
# Ticker Map for Live Data
# ----------------------------
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

# ----------------------------
# Safe YFinance Download
# ----------------------------
def safe_download(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)['Close']
        if data.isnull().sum() > 0 or len(data) < 45:
            st.warning(f"⚠️ Incomplete or missing data for {ticker}, skipping.")
            return None
        return data
    except Exception as e:
        st.warning(f"❌ Failed to fetch data for {ticker}: {e}")
        return None

# ----------------------------
# Backtesting (3-Month Window)
# ----------------------------
def backtest_portfolio(stocks_df, investment_amount=100000):
    st.subheader("⏱️ Backtesting (Last 3 Months)")

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(months=3)

    price_data = {}
    weights = {}

    for _, row in stocks_df.iterrows():
        stock = row['Stock']
        ticker_symbol = TICKER_MAP.get(stock)
        if not ticker_symbol:
            continue

        data = safe_download(ticker_symbol, start, end)
        if data is not None:
            price_data[stock] = data
            weights[stock] = row['Weight %'] / 100

    if not price_data:
        st.error("No valid stock data found for backtesting.")
        return

    df_prices = pd.DataFrame(price_data).dropna()
    if df_prices.empty:
        st.error("No overlapping historical data between stocks.")
        return

    normalized = df_prices / df_prices.iloc[0]
    portfolio = normalized.dot(pd.Series(weights)) * investment_amount

    # Chart output
    fig, ax = plt.subplots()
    ax.plot(portfolio.index, portfolio, label='Portfolio Value', color='green')
    ax.set_title("Portfolio Value Over Last 3 Months")
    ax.set_ylabel("Value (₹)")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Stats
    returns = portfolio.pct_change().dropna()
    cumulative_return = (portfolio[-1] / portfolio[0]) - 1
    annualized_return = (1 + cumulative_return) ** (1 / 0.25) - 1  # 3 months = 0.25 year
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    st.subheader("📌 Backtest Summary")
    st.write({
        'Cumulative Return': f"{cumulative_return * 100:.2f}%",
        'Annualized Return': f"{annualized_return * 100:.2f}%",
        'Volatility': f"{volatility * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}"
    })
