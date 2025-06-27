
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import sys
import os

# Ensure UTF-8 encoding on Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf8', buffering=1)

def safe_str(val):
    try:
        return str(val).encode('utf-8', 'ignore').decode('utf-8')
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

    if score <= 2:
        return "Conservative"
    elif score <= 5:
        return "Moderate"
    else:
        return "Aggressive"

def get_live_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'Price': float(info.get('currentPrice') or 0),
            '52 Week High': float(info.get('fiftyTwoWeekHigh') or 0),
            '52 Week Low': float(info.get('fiftyTwoWeekLow') or 0),
            'PE Ratio': float(info.get('trailingPE') or 0),
            'Dividend Yield': float(info.get('dividendYield') or 0),
            'Beta': float(info.get('beta') or 0)
        }
    except Exception as e:
        return {
            'Price': 0,
            '52 Week High': 0,
            '52 Week Low': 0,
            'PE Ratio': 0,
            'Dividend Yield': 0,
            'Beta': 0,
            'Error': safe_str(e)
        }

stock_mapping = {
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Zomato': 'ZOMATO.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'IRCTC': 'IRCTC.NS'
}

stock_risk = {
    'TCS': 'Conservative',
    'HDFC Bank': 'Moderate',
    'Infosys': 'Moderate',
    'Adani Enterprises': 'Aggressive',
    'Zomato': 'Aggressive',
    'Reliance Industries': 'Moderate',
    'Bajaj Finance': 'Moderate',
    'IRCTC': 'Aggressive'
}

st.set_page_config(page_title="Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender")

st.subheader("Client Profile")
age = st.slider("Age", 18, 75, 35)
income = st.number_input("Monthly Income (Rs.)", value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (Rs.)", value=100000, step=10000)
dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
live_data_toggle = st.checkbox("Use Live YFinance Data")

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.write(f"Risk Profile: {risk_profile}")
    st.write(f"Investment Amount: Rs. {investment_amount:,}")

    filtered_stocks = [s for s, r in stock_risk.items() if r == risk_profile]
    if len(filtered_stocks) < 5:
        for stock in stock_risk:
            if stock not in filtered_stocks:
                filtered_stocks.append(stock)
            if len(filtered_stocks) >= 5:
                break

    yf_symbols = [stock_mapping[s] for s in filtered_stocks]
    raw_data = yf.download(yf_symbols, period="1y", interval="1d")
    prices = raw_data['Adj Close'] if 'Adj Close' in raw_data else raw_data['Close']
    prices = prices.dropna()

    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    optimized_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    weights = np.array([cleaned_weights.get(stock_mapping[s], 0) for s in filtered_stocks])
    investment_per_stock = (weights * investment_amount)

    portfolio = pd.DataFrame({
        'Stock': filtered_stocks,
        'Weight %': [round(w * 100, 2) for w in weights],
        'Investment Amount (Rs.)': [round(i) for i in investment_per_stock]
    })

    if live_data_toggle:
        extra_data = []
        for stock in portfolio['Stock']:
            symbol = stock_mapping[stock]
            metrics = get_live_data(symbol)
            extra_data.append(metrics)
        portfolio = pd.concat([portfolio, pd.DataFrame(extra_data)], axis=1)

    st.subheader("Recommended Portfolio")
    st.dataframe(portfolio)

    fig, ax = plt.subplots()
    ax.pie(portfolio['Investment Amount (Rs.)'], labels=portfolio['Stock'], autopct='%1.1f%%')
    st.pyplot(fig)

    st.subheader("Projected Portfolio Value")
    projections = pd.DataFrame({'Year': list(range(duration + 1))})
    for label, rate in {'Bear': -0.05, 'Base': 0.08, 'Bull': 0.15}.items():
        projections[label] = investment_amount * ((1 + rate) ** projections['Year'])

    fig2, ax2 = plt.subplots()
    for col in projections.columns[1:]:
        ax2.plot(projections['Year'], projections[col], label=col)
    ax2.legend()
    st.pyplot(fig2)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
        projections.to_excel(writer, sheet_name='Projections', index=False)
    output.seek(0)
    st.download_button("Download Report (Excel)", output.read(), file_name="portfolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
