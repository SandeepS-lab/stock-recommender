from nsepy import get_history
from datetime import date, timedelta

# Symbol map
BACKTEST_SYMBOL_MAP = {
    'TCS': 'TCS',
    'HDFC Bank': 'HDFCBANK',
    'Infosys': 'INFY',
    'Adani Enterprises': 'ADANIENT',
    'ITC': 'ITC',
    'Reliance Industries': 'RELIANCE',
    'Bajaj Finance': 'BAJFINANCE',
    'IRCTC': 'IRCTC'
}

# Dates for backtesting
end_date = date.today()
start_date = end_date - timedelta(days=90)

# Check validity
valid_symbols = []
invalid_symbols = []

print("🔍 Checking symbol validity...\n")

for name, symbol in BACKTEST_SYMBOL_MAP.items():
    try:
        df = get_history(symbol=symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"❌ No data for {symbol} ({name})")
            invalid_symbols.append((name, symbol))
        else:
            print(f"✅ Valid symbol: {symbol} ({name})")
            valid_symbols.append((name, symbol))
    except Exception as e:
        print(f"⚠️ Error for {symbol} ({name}): {e}")
        invalid_symbols.append((name, symbol))

print("\n✅ Working Symbols:")
for name, sym in valid_symbols:
    print(f"- {name}: {sym}")

print("\n❌ Invalid or Empty Symbols:")
for name, sym in invalid_symbols:
    print(f"- {name}: {sym}")
