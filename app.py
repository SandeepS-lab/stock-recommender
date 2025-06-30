from nsepy import get_history
from datetime import date, timedelta
import pandas as pd

# -------------------------
# Configuration
# -------------------------
stocks = ['TCS', 'ITC', 'INFY']
end_date = date.today()
start_date = end_date - timedelta(days=90)

# -------------------------
# Fetch and Save Data
# -------------------------
for symbol in stocks:
    try:
        df = get_history(symbol=symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"❌ No data for {symbol}")
        else:
            df.to_csv(f"{symbol}_nse_3_months.csv")
            print(f"✅ Data fetched and saved: {symbol}_nse_3_months.csv")
    except Exception as e:
        print(f"⚠️ Error fetching {symbol}: {e}")
