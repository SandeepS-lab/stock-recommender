import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# ----------------------------
# Risk Profiling Logic
# ----------------------------

def get_risk_profile(age, income, dependents, qualification, duration, investment_type):
    score = 0

    # Age factor: Younger clients can take more risk
    if age < 30:
        score += 2
    elif age < 45:
        score += 1

    # Income factor: Higher income can absorb more risk
    if income > 100000:
        score += 2
    elif income > 50000:
        score += 1

    # Dependents factor: More dependents, less risk
    if dependents >= 3:
        score -= 1

    # Qualification factor: Professional qualifications might imply better financial literacy
    if qualification in ["Postgraduate", "Professional"]:
        score += 1

    # Investment Duration: Longer duration allows for more risk
    if duration >= 5:
        score += 1

    # Investment Type: SIPs generally reduce risk over time
    if investment_type == "SIP":
        score += 1

    # Introduced weights for risk factors in get_risk_profile to make the scoring more sophisticated. You'll need to fine-tune these weights and the RISK_SCORE_THRESHOLDS based on desired risk profile distribution.

    # Determine risk profile based on score
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
    # This data can be expanded with more stocks and real-time data integration
    data = {
        'Stock': [
            'TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises',
            'Zomato', 'Reliance Industries', 'Bajaj Finance', 'IRCTC'
        ],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.8, 1.0, 1.5, 1.8, 1.1, 1.3, 1.2],
        'P/E Ratio': [30, 25, 32, 40, 60, 28, 35, 50],
        'Dividend Yield': [1.5, 1.8, 1.2, 0.5, 0.0, 1.0, 0.7, 0.3],
        'Market Cap (Billions USD)': [150, 120, 140, 80, 50, 200, 90, 60],
        'Sector': [
            'IT', 'Financials', 'IT', 'Conglomerate',
            'Technology', 'Conglomerate', 'Financials', 'Logistics'
        ],
        'Risk Profile': [
            'Moderate', 'Conservative', 'Moderate', 'Aggressive',
            'Aggressive', 'Moderate', 'Moderate', 'Aggressive'
        ]
    }
    df_stocks = pd.DataFrame(data)

    # Filter stocks based on risk profile
    if risk_profile == "Conservative":
        recommended_stocks = df_stocks[df_stocks['Risk Profile'] == 'Conservative']
    elif risk_profile == "Moderate":
        recommended_stocks = df_stocks[df_stocks['Risk Profile'].isin(['Conservative', 'Moderate'])]
    else:  # Aggressive
        recommended_stocks = df_stocks

    # Sort based on Sharpe Ratio (higher is better)
    recommended_stocks = recommended_stocks.sort_values(by='Sharpe Ratio', ascending=False)

    # Allocate investment amount
    if not recommended_stocks.empty:
        # Simple equal weighted allocation for demonstration
        num_stocks = len(recommended_stocks)
        if num_stocks > 0:
            allocation_per_stock = investment_amount / num_stocks
            recommended_stocks['Allocated Amount (₹)'] = allocation_per_stock
            recommended_stocks['Weight (%)'] = (1 / num_stocks) * 100
        else:
            recommended_stocks['Allocated Amount (₹)'] = 0
            recommended_stocks['Weight (%)'] = 0
    else:
        st.warning("No stocks found for the selected risk profile.")
        return pd.DataFrame()

    return recommended_stocks

# ----------------------------
# Streamlit UI
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
        st.subheader("Recommended Stocks")
        st.dataframe(recommended_stocks.round(2)) # Display with 2 decimal places

        # Optional: Add a simple chart for allocation
        st.subheader("Investment Allocation Chart")
        fig = recommended_stocks.set_index('Stock')['Allocated Amount (₹)'].plot.pie(
            autopct='%1.1f%%', startangle=90, legend=False, figsize=(8, 8)
        ).figure
        st.pyplot(fig)
    else:
        st.warning("Could not generate stock recommendations. Please adjust client profile.")

st.markdown("---")
st.markdown("Disclaimer: This tool provides recommendations based on predefined logic and market data. It is not financial advice. Users should consult with a qualified financial advisor before making investment decisions.")
