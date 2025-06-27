import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
import re

# ASCII cleanup
def ascii_only(val):
    try:
        return re.sub(r'[^\x00-\x7F]+', '', str(val))
    except:
        return str(val)

# Risk logic
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

# Live metrics
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
        return {'Price': 0, '52 Week High': 0, '52 Week Low': 0,
                'PE Ratio': 0, 'Dividend Yield': 0, 'Beta': 0, 'Error': ascii_only(e)}

# Configuration
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
    'TCS': 'Conservative', 'HDFC Bank': 'Moderate', 'Infosys': 'Moderate',
    'Adani Enterprises': 'Aggressive', 'Zomato': 'Aggressive',
    'Reliance Industries': 'Moderate', 'Bajaj Finance': 'Moderate', 'IRCTC': 'Aggressive'
}

st.set_page_config(page_title="Stock Recommender", layout="centered")
st.title(ascii_only("Stock Recommender"))

# Form UI
with st.form("profile_form"):
    st.subheader(ascii_only("Client Profile"))
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Monthly Income (Rs)", value=50000)
    investment_amount = st.number_input("Investment Amount (Rs)", value=100000)
    dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
    qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
    duration = st.slider("Investment Duration (Years)", 1, 30, 5)
    investment_type = st.radio("Mode", ["Lumpsum", "SIP"])
    live = st.checkbox("Use Live Data")
    submitted = st.form_submit_button("Generate Portfolio")

if submitted:
    profile = ascii_only(get_risk_profile(age, income, dependents, qualification, duration, investment_type))
    st.write("Risk Profile:", profile)
    st.write("Investment Amount (Rs):", f"{investment_amount:,}")

    picked = [s for s, r in stock_risk.items() if r == profile]
    while len(picked) < 5:
        for s in stock_risk:
            if s not in picked:
                picked.append(s)
            if len(picked) >= 5:
                break
    picked = [ascii_only(p) for p in picked]

    symbols = [stock_mapping[p] for p in picked]
    raw = yf.download(symbols, period="1y", interval="1d", progress=False)
    prices = raw['Adj Close'] if 'Adj Close' in raw else raw['Close']
    prices = prices.fillna(method='ffill').dropna(axis=1, how='any').dropna()

    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    weights = ef.clean_weights()
    w_array = np.array([weights.get(stock_mapping[s], 0) for s in picked])
    inv = w_array * investment_amount

    portfolio = pd.DataFrame({
        'Stock': [ascii_only(s) for s in picked],
        'Weight %': [round(w * 100, 2) for w in w_array],
        'Investment (Rs)': [round(v) for v in inv]
    })

    if live:
        extra = [get_live_data(stock_mapping[s]) for s in portfolio['Stock']]
        portfolio = pd.concat([portfolio, pd.DataFrame(extra)], axis=1)

    st.subheader("Recommended Portfolio")
    st.dataframe(portfolio)

    fig, ax = plt.subplots()
    wedges, _, _ = ax.pie(portfolio['Investment (Rs)'], autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.legend(wedges, [ascii_only(s) for s in portfolio['Stock']], title="Stocks", loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    st.subheader("Portfolio Allocation")
    st.pyplot(fig)

    st.subheader("Backtest (1 Year)")
    cumulative = (prices / prices.iloc[0]) * investment_amount
    fig2, ax2 = plt.subplots()
    cumulative.plot(ax=ax2)
    ax2.set_ylabel("Portfolio Value (Rs)")
    ax2.set_title("Performance")
    st.pyplot(fig2)

    output = BytesIO()
    portfolio_ascii = portfolio.applymap(ascii_only)
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        portfolio_ascii.to_excel(writer, index=False, sheet_name='Portfolio')
    output.seek(0)
    st.download_button("Download Excel", output.read(), file_name="portfolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
