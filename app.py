from nsepy import get_history
from datetime import date, timedelta

# Define date range: last 30 calendar days
end_date = date.today()
start_date = end_date - timedelta(days=30)

# Fetch historical data for TCS
tcs_data = get_history(symbol='TCS',
                       start=start_date,
                       end=end_date)

# Display the data
print(tcs_data[['Close']])
