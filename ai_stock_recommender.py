
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Risk Profiling Logic
def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    score = 0
    if age < 30:
        score += 2
    elif age < 45:
        score += 1
    if income > 100000:
        score += 2
    elif income > 50000:
        score += 1
    if dependents >= 3:
        score -= 1
    if qualification in ["Postgraduate", "Professional"]:
        score += 1
    if duration >= 5:
        score += 1
    if investment_type == "SIP":
        score += 1
    if score <= 2:
        return "Conservative"
    elif score <= 5:
        return "Moderate"
    else:
        return "Aggressive"

# Dummy Stock Recommendations
def get_stock_list(risk_profile):
    data = {
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Ent', 'Zomato'],
        'Sharpe': [1.2, 1.0, 1.15, 0.85, 0.65],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small'],
        'Profile Fit': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive']
    }
    df = pd.DataFrame(data)
    return df[df['Profile Fit'] == risk_profile]

# Streamlit App UI
st.set_page_config(page_title="Mutual Fund Stock Recommender", layout="centered")
st.title("🧠 AI-Based Stock Recommender for Fund Managers")
st.subheader("Enter Client Profile")

age = st.slider("Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", value=50000, step=5000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])

if st.button("Get Recommendatiation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.subheader(f"Recommended Risk Profile: **{risk_profile}**")
    st.subheader("Suggested Stocks based on Risk Profile:")
    recommended_stocks = get_stock_list(risk_profile)
    st.dataframe(recommended_stocks)
