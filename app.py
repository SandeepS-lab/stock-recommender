import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Title
st.title("📈 Historical Stock Data Viewer")

# Date range: last 3 months
end_date = datetime.today()
start_date = end_date - timedelta(days=90)

# Fetch data
st.subheader("Historical Data for TCS (Last 3 Months)")
tcs_data = yf.download("TCS.NS", start=start_date, end=end_date)

# Show data in app
if not tcs_data.empty:
    st.dataframe(tcs_data)
else:
    st.warning("⚠️ No data found for TCS.NS in the given date range.")
