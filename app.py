import yfinance as yf
import pandas as pd

# Define ticker symbols
tickers = {
    'TCS': 'TCS.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'ITC Limited': 'ITC.NS'
}

# Function to fetch data
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'Current Price': info.get('currentPrice', 'N/A'),
            '52W High': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52W Low': info.get('fiftyTwoWeekLow', 'N/A'),
            'PE Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Beta': info.get('beta', 'N/A')
        }
    except Exception as e:
        return {'Error': str(e)}

# Fetch data for all stocks
data = {name: fetch_stock_data(ticker) for name, ticker in tickers.items()}

# Display as DataFrame
df = pd.DataFrame(data).T
print(df)
