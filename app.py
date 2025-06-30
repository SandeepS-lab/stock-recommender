from nsepy import get_history
from datetime import date, timedelta
import pandas as pd

# Wrapper to safely fetch data
def safe_get_history(symbol, start, end):
    try:
        return get_history(symbol=symbol, start=start, end=end)
    except ValueError as e:
        if "FrameLocalsProxy" in str(e):
            st.warning(f"⚠ NSE data error for {symbol} (FrameLocalsProxy bug). Returning empty.")
            return pd.DataFrame()
        else:
            raise e

# Example usage
end_date = date.today()
start_date = end_date - timedelta(days=30)
tcs_data = safe_get_history('TCS', start=start_date, end=end_date)

# Check result
if not tcs_data.empty:
    st.dataframe(tcs_data[['Close']])
else:
    st.info("No data returned for TCS. Possibly due to NSE block or system environment.")
