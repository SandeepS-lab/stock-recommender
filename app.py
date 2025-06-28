import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from io import BytesIO
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ----------------------------
# Ticker Map
# ----------------------------
TICKER_MAP = {
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Zomato": "ZOMATO.NS",
    "Reliance": "RELIANCE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "IRCTC": "IRCTC.NS"
}

# Risk vs Return Plot (Safe + Fallback)
try:
    tickers = stocks['Stock'].map(TICKER_MAP).dropna().tolist()

    if len(tickers) < 2:
        raise ValueError("Need at least two stocks for plotting Risk vs Return.")

    raw_data = yf.download(tickers, period="1y", progress=False)

    # Handle both MultiIndex and flat structure
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Adj Close'].dropna(axis=1, how='any')
    else:
        data = raw_data.dropna(axis=1, how='any')

    if data.shape[1] < 2:
        raise ValueError("Not enough price history for selected tickers.")

    returns = data.pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    exp_ret = returns.mean() * 252
    sharpe = exp_ret / vol

    plot_df = pd.DataFrame({
        "Ticker": vol.index.str.replace(".NS", "", regex=False),
        "Volatility": vol.values,
        "Expected Return": exp_ret.values,
        "Sharpe Ratio": sharpe.values
    })

except Exception as e:
    # Fallback to predefined data
    st.warning("\u26a0\ufe0f Falling back to sample data for Risk vs Return.")
    plot_df = pd.DataFrame({
        "Ticker": ["TCS", "Infosys", "Reliance", "Zomato"],
        "Volatility": [0.20, 0.22, 0.18, 0.35],
        "Expected Return": [0.12, 0.11, 0.09, 0.15],
        "Sharpe Ratio": [1.2, 1.1, 1.0, 0.65]
    })
    try:
        with st.expander("🛠 Debug Log"):
            st.code(str(e))
    except UnicodeEncodeError:
        with st.expander("Debug Log"):
            st.code(str(e))

# Plot Risk vs Return
fig = px.scatter(
    plot_df,
    x="Volatility",
    y="Expected Return",
    color="Sharpe Ratio",
    size="Sharpe Ratio",
    hover_name="Ticker",
    color_continuous_scale="RdYlGn",
    title="Risk vs Return Bubble Chart"
)
st.plotly_chart(fig, use_container_width=True)
