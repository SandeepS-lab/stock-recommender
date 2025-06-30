import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------
# Setup
# ----------------------------
st.title("📈 3-Month Backtest: Equal-Weighted Portfolio")

# Define stock tickers
tickers = {
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS"
}

# Investment assumptions
initial_investment = 100000
num_stocks = len(tickers)
investment_per_stock = initial_investment / num_stocks

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# ----------------------------
# Fetch Data
# ----------------------------
price_data = pd.DataFrame()

for name, symbol in tickers.items():
    data = yf.download(symbol, start=start_date, end=end_date)['Close']
    if data.empty:
        st.warning(f"⚠️ No data for {symbol}")
        continue
    data = data.rename(name)
    price_data = pd.concat([price_data, data], axis=1)

# ----------------------------
# Backtest Logic
# ----------------------------
if not price_data.empty:
    st.subheader("📊 Daily Closing Prices")
    st.dataframe(price_data)

    # Normalize to initial investment per stock
    normalized = price_data / price_data.iloc[0]  # returns = price / first price
    invested = normalized * investment_per_stock

    # Portfolio value over time
    portfolio_value = invested.sum(axis=1)

    st.subheader("📈 Portfolio Value Over Time")
    st.line_chart(portfolio_value)

    # Final stats
    total_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1) * 100
    cagr = ((portfolio_value.iloc[-1] / portfolio_value.iloc[0]) ** (365 / 90) - 1) * 100

    st.subheader("📌 Performance Summary")
    st.write(f"**Initial Investment:** ₹{initial_investment:,.0f}")
    st.write(f"**Final Value:** ₹{portfolio_value.iloc[-1]:,.2f}")
    st.write(f"**Total Return (3M):** {total_return:.2f}%")
    st.write(f"**CAGR (Annualized):** {cagr:.2f}%")
else:
    st.error("No valid data to run backtest.")
