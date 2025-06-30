from nsepython import nse_historical

# Fetch historical data for TCS (TCS is the correct symbol)
data = nse_historical(symbol="TCS", from_date="01-01-2024", to_date="01-07-2024")

# Display the data
print(data)
