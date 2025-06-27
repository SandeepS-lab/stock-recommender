import pandas as pd
import numpy as np
import yfinance as yf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from io import BytesIO
import re

def ascii_only(val):
    try:
        return re.sub(r'[^\x00-\x7F]+', '', str(val))
    except:
        return str(val)

def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    score = 0
    if age < 30: score += 2
    elif age < 45: score += 1
    if income > 100000: score += 2
    elif income > 50000: score += 1
    if dependents >= 3: score -= 1
    if qualification in ["Postgraduate", "Professional"]: score += 1
    if duration >= 5: score += 1
    if investment_type == "SIP": score += 1
    return "Conservative" if score <= 2 else "Moderate" if score <= 5 else "Aggressive"

def get_live_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'Price': float(info.get('currentPrice') or 0),
            'PE Ratio': float(info.get('trailingPE') or 0),
            'Dividend Yield': float(info.get('dividendYield') or 0),
            'Beta': float(info.get('beta') or 0)
        }
    except Exception as e:
        return {"Error": ascii_only(e)}

stock_mapping = {
    'TCS': 'TCS.NS', 'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS', 'Adani Enterprises': 'ADANIENT.NS',
    'Zomato': 'ZOMATO.NS', 'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS', 'IRCTC': 'IRCTC.NS'
}
stock_risk = {
    'TCS': 'Conservative', 'HDFC Bank': 'Moderate', 'Infosys': 'Moderate',
    'Adani Enterprises': 'Aggressive', 'Zomato': 'Aggressive',
    'Reliance Industries': 'Moderate', 'Bajaj Finance': 'Moderate', 'IRCTC': 'Aggressive'
}

# === User Input ===
print("=== ASCII-Only Stock Recommender ===")
age = int(input("Age: "))
income = int(input("Monthly Income (Rs): "))
amount = int(input("Investment Amount (Rs): "))
dependents = int(input("Dependents (0-4): "))
qualification = input("Qualification [Graduate/Postgraduate/Professional/Other]: ")
duration = int(input("Investment Duration (Years): "))
investment_type = input("Investment Type [Lumpsum/SIP]: ")

profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
print("\nRisk Profile:", profile)

# === Portfolio Logic ===
stocks = [s for s, r in stock_risk.items() if r == profile]
while len(stocks) < 5:
    for s in stock_risk:
        if s not in stocks:
            stocks.append(s)
        if len(stocks) >= 5: break

symbols = [stock_mapping[s] for s in stocks]
data = yf.download(symbols, period="1y", interval="1d", progress=False)
prices = data['Adj Close'].fillna(method="ffill").dropna(axis=1, how='any').dropna()

mu = mean_historical_return(prices)
S = CovarianceShrinkage(prices).ledoit_wolf()
ef = EfficientFrontier(mu, S)
weights = ef.clean_weights()

w_array = np.array([weights.get(stock_mapping[s], 0) for s in stocks])
investments = w_array * amount

portfolio = pd.DataFrame({
    'Stock': [ascii_only(s) for s in stocks],
    'Weight %': [round(w * 100, 2) for w in w_array],
    'Invested (Rs)': [round(v) for v in investments]
})

# Optional: Live metrics
use_live = input("Fetch live metrics? [y/n]: ").lower()
if use_live == 'y':
    metrics = [get_live_data(stock_mapping[s]) for s in portfolio['Stock']]
    portfolio = pd.concat([portfolio, pd.DataFrame(metrics)], axis=1)

# === Output ===
print("\n--- Recommended Portfolio ---")
print(portfolio.to_string(index=False))

# Export
save = input("\nSave to Excel? [y/n]: ").lower()
if save == 'y':
    out = BytesIO()
    portfolio_ascii = portfolio.applymap(ascii_only)
    with pd.ExcelWriter(out, engine='xlsxwriter') as writer:
        portfolio_ascii.to_excel(writer, index=False, sheet_name="Portfolio")
    with open("portfolio.xlsx", "wb") as f:
        f.write(out.getvalue())
    print("Excel saved as portfolio.xlsx")
