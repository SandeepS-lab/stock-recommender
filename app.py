import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

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
# Stock Recommendation Logic
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
        selected = pd.concat(dfs, ignore_index=True)
    else:
        selected = df[df['Risk Category'] == risk_profile].copy()
        selected['Score'] = selected['Sharpe Ratio'] / selected['Beta']
        selected['Weight %'] = selected['Score'] / selected['Score'].sum() * 100
        selected['Investment Amount (₹)'] = (selected['Weight %'] / 100) * investment_amount

    return selected[['Stock', 'Risk Category', 'Sharpe Ratio', 'Beta', 'Weight %', 'Investment Amount (₹)']]

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Stock Recommender", layout="centered")
st.title("📈 AI-Based Stock Recommender")

with st.form("user_input"):
    age = st.slider("Age", 18, 75, 35)
    income = st.number_input("Monthly Income (₹)", value=50000)
    investment_amount = st.number_input("Investment Amount (₹)", value=100000)
    dependents = st.select
