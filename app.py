import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------------
# Risk Profiling Logic
# ----------------------------

def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    score = 0

    # Age
    if age < 30:
        score += 2
    elif age < 45:
        score += 1

    # Income
    if income > 100000:
        score += 2
    elif income > 50000:
        score += 1

    # Dependents
    if dependents >= 3:
        score -= 1

    # Qualification
    if qualification in ["Postgraduate", "Professional"]:
        score += 1

    # Duration
    if duration >= 5:
        score += 1

    # SIP preference
    if investment_type == "SIP":
        score += 1

    if score <= 2:
        return "Conservative"
    elif score <= 5:
        return "Moderate"
    else:
        return "Aggressive"

# ----------------------------
# Stock Recommendation Logic
# ----------------------------

def get_stock_list(risk_profile):
    data = {
        'Stock': [
            'TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises',
            'Zomato', 'Reliance Industries', 'Bajaj Finance', 'IRCTC'
        ],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small', 'Large', 'Mid', 'Mid'],
        'Risk Category': [
            'Conservative', 'Moderate', 'Moderate', 'Aggressive',
            'Aggressive', 'Moderate', 'Moderate', 'Aggressive'
        ]
    }
    df = pd.DataFrame(data)
    return df[df['Risk Category'] == risk_profile]

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="Mutual Fund Stock Recommender", layout="centered")
st.title("📊 Mutual Fund Stock Recommender")
st.markdown("Use this AI-powered tool to generate stock recommendations based on your client's risk profile.")

st.header("📋 Client Profile Information")

# Input fields
age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])

# Recommendation button
if st.button("Get Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)

    st.success(f"🎯 Recommended Risk Profile: **{risk_profile}**")
    st.info(f"💰 Suggested Allocation for Investment Amount ₹{investment_amount:,}")

    recommended_stocks = get_stock_list(risk_profile)

    st.markdown("### 📈 Suggested Stocks Based on Risk Profile")
    st.dataframe(recommended_stocks, use_container_width=True)
