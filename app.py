import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Title
st.title("📊 3-Month Historical Stock Data")

# Define ticker symbols
tickers = {
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS"
}

# Date range: last 3 months
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Display data for each stock
for company, symbol in tickers.items():
    st.subheader(f"{company} ({symbol})")
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if not data.empty:
            st.dataframe(data)
        else:
            st.warning(f"No data found for {symbol}")
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
