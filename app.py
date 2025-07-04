import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Live Stock Data", layout="centered")

st.title("📈 Live Stock Data Viewer")

# Define the stock tickers
tickers = {
    'TCS': 'TCS.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ITC Limited': 'ITC.NS'
}

# Function to fetch stock info
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'Current Price': info.get('currentPrice', 'N/A'),
            '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': round(info.get('dividendYield', 0)*100, 2) if info.get('dividendYield') else 'N/A',
            'Beta': info.get('beta', 'N/A')
        }
    except Exception as e:
        return {'Error': str(e)}

# Create a refresh button
if st.button("🔄 Refresh Live Data"):
    st.info("Fetching live data...")

    # Fetch and display stock info for each ticker
    stock_data = {}
    for name, ticker in tickers.items():
        stock_data[name] = fetch_stock_data(ticker)

    # Convert to DataFrame and display
    df = pd.DataFrame.from_dict(stock_data, orient='index')
    st.dataframe(df)
else:
    st.write("Click the button above to fetch live stock data.")
