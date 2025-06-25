import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------------
# Risk Profiling Logic
# ----------------------------

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

# ----------------------------
# Stock Recommendation Logic
# ----------------------------

def get_stock_list(risk_profile, investment_amount):
    data = {
        'Stock': [
            'TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises',
            'Zomato', 'Reliance Industries', 'Bajaj Finance', 'IRCTC'
        ],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'Volatility': [0.18, 0.20, 0.19, 0.25, 0.30, 0.22, 0.21, 0.28],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small', 'Large', 'Mid', 'Mid'],
        'Risk Category': [
            'Conservative', 'Moderate', 'Moderate', 'Aggressive',
            'Aggressive', 'Moderate', 'Moderate', 'Aggressive'
        ]
    }
    df = pd.DataFrame(data)
    selected = df[df['Risk Category'] == risk_profile].copy()

    # Assign proportion (equal weight for simplicity or based on inverse beta)
    if not selected.empty:
        selected['Weight %'] = (1 / selected['Beta'])
        selected['Weight %'] = selected['Weight %'] / selected['Weight %'].sum() * 100
        selected['Investment Amount (₹)'] = (selected['Weight %'] / 100) * investment_amount
        selected = selected.round({'Weight %': 2, 'Investment Amount (₹)': 0, 'Sharpe Ratio': 2, 'Beta': 2, 'Volatility': 2})

    return selected

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered")
st.title("💼 AI-Based Stock Recommender for Mutual Fund Managers")
st.markdown("This intelligent assistant recommends stock allocations based on a client's risk profile using advanced metrics.")

st.header("📋 Enter Client Profile")

# Input fields
age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])

# Recommendation button
if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"📊 Risk Profile: **{risk_profile}**")
    st.info(f"💰 Investment Allocation for ₹{investment_amount:,.0f}")

    recommended_stocks = get_stock_list(risk_profile, investment_amount)

    if not recommended_stocks.empty:
        st.markdown("### 📈 Recommended Stock Portfolio")
        st.dataframe(recommended_stocks, use_container_width=True)
    else:
        st.warning("No suitable stocks found for this risk profile.")
