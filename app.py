import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Title
st.title("📈 Historical Stock Data Viewer (Last 3 Months)")

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Tickers to fetch
tickers = {
    "HDFC Bank": "HDFCBANK.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS"
}

# Loop through each ticker and display data
for company, symbol in tickers.items():
    st.subheader(f"{company} ({symbol})")
    data = yf.download(symbol, start=start_date, end=end_date)

    if not data.empty:
        st.dataframe(data)
    else:
        st.warning(f"⚠️ No data found for {symbol} in the given date range.")
