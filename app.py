import yfinance as yf

# TCS stock symbol for NSE on Yahoo Finance
ticker = 'TCS.NS'

# Fetch historical data (last 60 days in this case)
tcs_data = yf.download(ticker, period="60d", interval="1d")

# Display the first few rows
print(tcs_data.head())
