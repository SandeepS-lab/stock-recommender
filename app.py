from nsepy import get_history
from datetime import date, timedelta

def safe_get_history(symbol, start, end):
    # Force removal of frame-related keyword to prevent crash
    try:
        return get_history(symbol=symbol, start=start, end=end)
    except ValueError as e:
        if "FrameLocalsProxy" in str(e):
            return pd.DataFrame()  # Return empty if this bug happens
        else:
            raise e

# Use it like this:
end_date = date.today()
start_date = end_date - timedelta(days=30)
tcs_data = safe_get_history('TCS', start=start_date, end=end_date)
print(tcs_data[['Close']])
