import streamlit as st
import yfinance as yf
import pandas as pd

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Indian Stock Info", layout="wide")
st.title("📈 Indian Stock Overview")
st.markdown("View current statistics for selected Indian stocks using live data from Yahoo Finance.")

# ------------------------------
# Define NSE stock tickers
# ------------------------------
tickers = {
    'TCS': 'TCS.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ITC Limited': 'ITC.NS'
}

# ------------------------------
# Function to fetch stock info
# ------------------------------
@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'Current Price': info.get('currentPrice', 'N/A'),
            '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': round(info.get('dividendYield', 0) * 100, 2) if info.get('dividendYield') else 'N/A',
            'Beta': info.get('beta', 'N/A')
        }
    except Exception as e:
        return {'Error': str(e)}

# ------------------------------
# Fetch data for all tickers
# ------------------------------
with st.spinner("Fetching data from Yahoo Finance..."):
    stock_data = {name: fetch_stock_data(ticker) for name, ticker in tickers.items()}

df = pd.DataFrame.from_dict(stock_data, orient='index')

# ------------------------------
# Display the DataFrame
# ------------------------------
st.subheader("📊 Stock Summary")
st.dataframe(df, use_container_width=True)

# ------------------------------
# Option to download CSV
# ------------------------------
csv = df.to_csv().encode('utf-8')
st.download_button(
    label="⬇️ Download as CSV",
    data=csv,
    file_name='stock_summary.csv',
    mime='text/csv'
)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Data source: Yahoo Finance | Built with ❤️ using Streamlit")
