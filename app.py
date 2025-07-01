import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Function to perform SMA crossover backtest
def sma_backtest(data, short_window=10, long_window=50):
    data = data.copy()
    data['SMA10'] = data['Close'].rolling(window=short_window).mean()
    data['SMA50'] = data['Close'].rolling(window=long_window).mean()

    # Drop rows where either SMA is NaN
    data.dropna(inplace=True)

    if data.empty:
        return None

    # Generate signals
    data['Signal'] = 0
    data.loc[data['SMA10'] > data['SMA50'], 'Signal'] = 1
    data.loc[data['SMA10'] < data['SMA50'], 'Signal'] = -1
    data['Position'] = data['Signal'].shift(1)

    # Calculate returns
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Returns'] * data['Position']

    # Cumulative returns
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod()
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod()

    return data

# Title
st.title("📈 SMA Crossover Backtest: HDFC Bank, TCS, Infosys")

# Date range: last 3 months
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Ticker dictionary
tickers = {
    "HDFC Bank": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

# Loop and backtest each stock
for company, symbol in tickers.items():
    st.subheader(f"{company} ({symbol})")
    data = yf.download(symbol, start=start_date, end=end_date)

    if not data.empty:
        result = sma_backtest(data)

        if result is not None and 'Cumulative_Market' in result.columns:
            # Plot only if valid results
            st.line_chart(result[['Cumulative_Market', 'Cumulative_Strategy']])
            st.write("📊 Final Returns:")
            st.metric("Market Return (%)", f"{(result['Cumulative_Market'].iloc[-1] - 1) * 100:.2f}")
            st.metric("Strategy Return (%)", f"{(result['Cumulative_Strategy'].iloc[-1] - 1) * 100:.2f}")
            st.dataframe(result.tail(10))
        else:
            st.warning("⚠️ Not enough data to perform backtest (try a longer period).")
    else:
        st.warning(f"⚠️ No data found for {symbol}.")
