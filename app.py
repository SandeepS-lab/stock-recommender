import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import yfinance as yf

# ----------------------------
# Risk Profiling Logic
# ----------------------------

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

# ----------------------------
# Stock Mapping (Ticker Map)
# ----------------------------

TICKER_MAP = {
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Zomato": "ZOMATO.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "IRCTC": "IRCTC.NS",
    "Adani Ent.": "ADANIENT.NS",
    "Reliance": "RELIANCE.NS",
    "Bajaj Fin.": "BAJFINANCE.NS"
}

# ----------------------------
# Fetch Live Data from yfinance
# ----------------------------

def fetch_live_data(stocks):
    live_data = []
    for stock in stocks:
        ticker = TICKER_MAP.get(stock)
        if not ticker:
            continue
        try:
            info = yf.Ticker(ticker).info
            live_data.append({
                "Stock": stock,
                "Live Price": round(info.get("currentPrice", 0), 2),
                "52W High": round(info.get("fiftyTwoWeekHigh", 0), 2),
                "52W Low": round(info.get("fiftyTwoWeekLow", 0), 2),
                "Dividend Yield": round((info.get("dividendYield", 0) or 0) * 100, 2),
                "P/E Ratio": round(info.get("trailingPE", 0) or 0, 2),
                "Beta": round(info.get("beta", 0), 2),
                "Market Cap (Cr)": round((info.get("marketCap", 0) or 0) / 1e7, 2)
            })
        except:
            live_data.append({
                "Stock": stock,
                "Live Price": 0,
                "52W High": 0,
                "52W Low": 0,
                "Dividend Yield": 0,
                "P/E Ratio": 0,
                "Beta": 0,
                "Market Cap (Cr)": 0
            })
    return pd.DataFrame(live_data)

# ----------------------------
# Basic Recommender
# ----------------------------

def get_stock_list(risk_profile, investment_amount, diversify=False):
    data = {
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises', 'Zomato',
                  'Reliance Industries', 'Bajaj Finance', 'IRCTC'],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'Volatility': [0.18, 0.20, 0.19, 0.25, 0.30, 0.22, 0.21, 0.28],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small', 'Large', 'Mid', 'Mid'],
        'Risk Category': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive',
                          'Moderate', 'Moderate', 'Aggressive']
    }
    df = pd.DataFrame(data)

    if diversify:
        portions = {'Conservative': 0.33, 'Moderate': 0.33, 'Aggressive': 0.34}
        dfs = []
        for cat, portion in portions.items():
            temp = df[df['Risk Category'] == cat].copy()
            temp['Score'] = temp['Sharpe Ratio'] / temp['Beta']
            temp['Weight %'] = temp['Score'] / temp['Score'].sum() * portion * 100
            temp['Investment Amount (₹)'] = (temp['Weight %'] / 100) * investment_amount
            dfs.append(temp)
        selected = pd.concat(dfs)
    else:
        selected = df[df['Risk Category'] == risk_profile].copy()
        if len(selected) < 5:
            others = df[df['Risk Category'] != risk_profile]
            selected = pd.concat([selected, others.head(5 - len(selected))])
        selected['Score'] = selected['Sharpe Ratio'] / selected['Beta']
        selected['Weight %'] = selected['Score'] / selected['Score'].sum() * 100
        selected['Investment Amount (₹)'] = (selected['Weight %'] / 100) * investment_amount

    return selected.round(2).drop(columns=['Score'])

# ----------------------------
# Enhanced Recommender
# ----------------------------

def enhanced_stock_selection(risk_profile, investment_amount):
    data = {
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Ent.', 'Zomato', 'Reliance', 'Bajaj Fin.', 'IRCTC'],
        'Sector': ['IT', 'Banking', 'IT', 'Infra', 'Tech', 'Energy', 'Finance', 'Travel'],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'P/E': [29, 21, 27, 42, 80, 31, 37, 65],
        'ROE': [24, 18, 22, 12, 3, 20, 21, 17],
        'Risk Category': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive',
                          'Moderate', 'Moderate', 'Aggressive']
    }
    df = pd.DataFrame(data)

    df['Score'] = (
        (df['Sharpe Ratio'] / df['Beta']) * 0.4 +
        (1 / df['P/E']) * 0.2 +
        (df['ROE'] / 100) * 0.4
    )

    filtered = df[df['Risk Category'] == risk_profile].copy()
    filtered = filtered.sort_values(by='Score', ascending=False).head(4)

    filtered['Weight %'] = filtered['Score'] / filtered['Score'].sum() * 100
    filtered['Investment Amount (₹)'] = filtered['Weight %'] / 100 * investment_amount

    return filtered[['Stock', 'Sector', 'Sharpe Ratio', 'Beta', 'P/E', 'ROE', 'Weight %', 'Investment Amount (₹)']].round(2)

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Stock Recommender", layout="centered")
st.title("Stock Recommender with Live Market Data")

st.markdown("Get smart stock picks based on risk profiles, with live market metrics.")

st.header("Client Profile Input")

age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
diversify = st.checkbox("Diversify portfolio across all risk levels")
strategy = st.radio("Recommendation Strategy", ["Basic", "Enhanced"])

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile: {risk_profile}")
    st.info(f"Investment Amount: ₹{investment_amount:,.0f}")

    if strategy == "Basic":
        recommended = get_stock_list(risk_profile, investment_amount, diversify)
    else:
        recommended = enhanced_stock_selection(risk_profile, investment_amount)

    if not recommended.empty:
        st.subheader("Recommended Portfolio")
        st.dataframe(recommended, use_container_width=True)

        st.subheader("📡 Live Stock Metrics")
        live_df = fetch_live_data(recommended["Stock"].tolist())
        st.dataframe(live_df, use_container_width=True)
    else:
        st.warning("No stocks matched the profile.")
