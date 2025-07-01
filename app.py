import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Title
st.title("📈 Historical Stock Data Viewer")

# Date range: last 3 months
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Fetch data for HDFC Bank
st.subheader("Historical Data for HDFC Bank (Last 3 Months)")
hdfc_data = yf.download("HDFCBANK.NS", start=start_date, end=end_date)

# Show data in app
if not hdfc_data.empty:
    st.dataframe(hdfc_data)
else:
    st.warning("⚠️ No data found for HDFCBANK.NS in the given date range.")
