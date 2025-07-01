import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to perform SMA crossover backtest
def sma_backtest(data, short_window=10, long_window=50):
    data = data.copy()
    data['SMA10'] = data['Close'].rolling(window=short_window).mean()
    data['SMA50'] = data['Close'].rolling(window=long_window).mean()
    data.dropna(inplace=True)

    # Generate signals
    data['Signal'] = 0
    data['Signal'][data['SMA10'] > data['SMA50']] = 1
    data['Signal'][data['SMA10'] < data['SMA50']] = -1
    data['Position'] = data['Signal'].shift(1)

    # Calculate daily returns
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position']

    # Calculate cumulative returns
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

    return data

# Title
st.title("📈 SMA Backtest: HDFC Bank, TCS, Infosys")

# Date range: last 3 months
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Tickers
tickers = {
    "HDFC Bank": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

# Perform and display backtest for each stock
for company, symbol in tickers.items():
    st.subheader(f"{company} ({symbol})")
    data = yf.download(symbol, start=start_date, end=end_date)

    if not data.empty:
        backtest_data = sma_backtest(data)

        st.line_chart(backtest_data[['Cumulative_Market', 'Cumulative_Strategy']])
        st.write("📊 Final Returns:")
        st.metric("Market Return (%)", f"{(backtest_data['Cumulative_Market'].iloc[-1] - 1) * 100:.2f}")
        st.metric("Strategy Return (%)", f"{(backtest_data['Cumulative_Strategy'].iloc[-1] - 1) * 100:.2f}")

        st.dataframe(backtest_data.tail(10))
    else:
        st.warning(f"⚠️ No data found for {symbol} in the given date range.")
