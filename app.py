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
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises', 'Zomato', 'Reliance Industries', 'Bajaj Finance', 'IRCTC'],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'Volatility': [0.18, 0.20, 0.19, 0.25, 0.30, 0.22, 0.21, 0.28],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small', 'Large', 'Mid', 'Mid'],
        'Risk Category': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive', 'Moderate', 'Moderate', 'Aggressive']
    }
    df = pd.DataFrame(data)

    if diversify:
        portions = {'Conservative': 0.33, 'Moderate': 0.33, 'Aggressive': 0.34}
        dfs = []
        for cat, portion in portions.items():
            temp = df[df['Risk Category'] == cat].copy()
            temp = temp.drop_duplicates(subset='Stock')
            if not temp.empty:
                temp['Score'] = temp['Sharpe Ratio'] / temp['Beta']
                temp['Weight %'] = temp['Score'] / temp['Score'].sum() * portion * 100
                temp['Investment Amount (₹)'] = (temp['Weight %'] / 100) * investment_amount
                dfs.append(temp)
        selected = pd.concat(dfs)
    else:
        selected = df[df['Risk Category'] == risk_profile].copy().drop_duplicates(subset='Stock')
        if len(selected) < 5:
            others = df[df['Risk Category'] != risk_profile].drop_duplicates(subset='Stock')
            selected = pd.concat([selected, others.head(5 - len(selected))])
        selected['Score'] = selected['Sharpe Ratio'] / selected['Beta']
        selected['Weight %'] = selected['Score'] / selected['Score'].sum() * 100
        selected['Investment Amount (₹)'] = (selected['Weight %'] / 100) * investment_amount

    selected = selected.round({'Weight %': 2, 'Investment Amount (₹)': 0, 'Sharpe Ratio': 2, 'Beta': 2, 'Volatility': 2})
    return selected.drop(columns=['Score'])

# ----------------------------
# Earnings Simulation
# ----------------------------

def simulate_earnings(amount, years):
    rates = {'Bear (-5%)': -0.05, 'Base (8%)': 0.08, 'Bull (15%)': 0.15}
    result = pd.DataFrame({'Year': list(range(0, years + 1))})
    for label, rate in rates.items():
        result[label] = amount * ((1 + rate) ** result['Year'])
    return result

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender for Mutual Fund Managers")

st.markdown("Get stock allocations based on your client's risk profile with earnings forecasts under multiple market conditions.")

st.header("Enter Client Profile")

# Inputs
age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
diversify = st.checkbox("Diversify portfolio across all risk levels")

# Generate button
if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile: {risk_profile}")
    st.info(f"Investment Allocation for ₹{investment_amount:,.0f}")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=diversify)

    if not recommended_stocks.empty:
        st.markdown("### Recommended Stock Portfolio")
        st.dataframe(recommended_stocks, use_container_width=True)

        # --------- Pie Chart ---------
        if recommended_stocks['Investment Amount (₹)'].sum() > 0:
            fig1, ax1 = plt.subplots()
            ax1.pie(recommended_stocks['Investment Amount (₹)'], labels=recommended_stocks['Stock'], autopct='%1.1f%%')
            ax1.set_title("Investment Allocation Breakdown")
            st.pyplot(fig1)

        # --------- Bar Chart ---------
        fig2, ax2 = plt.subplots()
        ax2.bar(recommended_stocks['Stock'], recommended_stocks['Weight %'], color='skyblue')
        ax2.set_title("Portfolio Weights by Stock")
        ax2.set_ylabel("Weight (%)")
        ax2.set_xticklabels(recommended_stocks['Stock'], rotation=45)
        st.pyplot(fig2)

        # --------- Line Chart (Projection) ---------
        st.markdown("### Projected Earnings Scenarios")
        earnings = simulate_earnings(investment_amount, duration)
        fig3, ax3 = plt.subplots()
        for col in earnings.columns[1:]:
            ax3.plot(earnings['Year'], earnings[col], label=col)
        ax3.set_title("Projected Portfolio Value Over Time")
        ax3.set_ylabel("Portfolio Value (₹)")
        ax3.set_xlabel("Year")
        ax3.legend()
        st.pyplot(fig3)

        # --------- Excel Export ---------
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            recommended_stocks.to_excel(writer, sheet_name='Portfolio', index=False)
            earnings.to_excel(writer, sheet_name='Projections', index=False)
        st.download_button(
            label="Download Excel Report",
            data=output.getvalue(),
            file_name="stock_recommendation_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("No suitable stocks found for this risk profile.")
