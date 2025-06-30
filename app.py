import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

def safe_download(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)['Close']
        if data.isnull().sum() > 0 or len(data) < 90:
            print(f"⚠️ Incomplete data for {ticker}")
            return None
        return data
    except Exception as e:
        print(f"❌ Failed to download {ticker}: {e}")
        return None

def backtest_portfolio(stocks_df, investment_amount=100000):
    st.subheader("📊 Backtesting Over Past 6 Months")

    end = pd.Timestamp.today()
    start = end - pd.DateOffset(months=6)

    price_data = {}
    weights = {}

    for _, row in stocks_df.iterrows():
        stock = row['Stock']
        ticker_symbol = row.get('Ticker', f"{stock}.NS")

        if not ticker_symbol:
            continue

        data = safe_download(ticker_symbol, start, end)
        if data is not None:
            price_data[stock] = data
            weights[stock] = row['Weight %'] / 100

    if not price_data:
        return None, "❌ No valid stock data available for backtesting."

    # Combine and clean data
    df_prices = pd.DataFrame(price_data).dropna()
    if df_prices.empty:
        return None, "❌ No overlapping historical data found."

    # Normalize and compute portfolio
    normalized = df_prices / df_prices.iloc[0]
    portfolio = normalized.dot(pd.Series(weights)) * investment_amount

    # Plot
    fig, ax = plt.subplots()
    ax.plot(portfolio.index, portfolio, label='Portfolio Value', color='blue')
    ax.set_title("Portfolio Value Over 6 Months")
    ax.set_ylabel("Value (₹)")
    ax.set_xlabel("Date")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)

    # Calculate performance metrics
    returns = portfolio.pct_change().dropna()
    cumulative_return = (portfolio[-1] / portfolio[0]) - 1
    cagr = (1 + cumulative_return) ** (1 / 0.5) - 1  # 6 months = 0.5 year
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / volatility if volatility > 0 else 0

    stats = {
        'Cumulative Return': f"{cumulative_return * 100:.2f}%",
        'CAGR': f"{cagr * 100:.2f}%",
        'Volatility': f"{volatility * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
    }

    st.subheader("📌 Backtest Summary Stats")
    st.write(stats)

    return portfolio, stats
