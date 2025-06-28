import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import yfinance as yf

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.expected_returns import mean_historical_return

# -------------------------------------
# Ticker Mapping for PyPortfolioOpt use
# -------------------------------------
TICKER_MAP = {
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Zomato": "ZOMATO.NS",
    "Reliance Industries": "RELIANCE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "IRCTC": "IRCTC.NS"
}

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

    if score <= 2: return "Conservative"
    elif score <= 5: return "Moderate"
    else: return "Aggressive"

# ----------------------------
# Basic Recommender
# ----------------------------
def get_stock_list(risk_profile, investment_amount, diversify=False):
    data = {
        'Stock': list(TICKER_MAP.keys()),
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
# Enhanced Scoring Recommender
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
    filtered = df[df['Risk Category'] == risk_profile].copy().sort_values(by='Score', ascending=False).head(4)
    filtered['Weight %'] = filtered['Score'] / filtered['Score'].sum() * 100
    filtered['Investment Amount (₹)'] = filtered['Weight %'] / 100 * investment_amount
    return filtered[['Stock', 'Sector', 'Sharpe Ratio', 'Beta', 'P/E', 'ROE', 'Weight %', 'Investment Amount (₹)']].round(2)

# ----------------------------
# MPT Optimization Strategy
# ----------------------------
def optimize_portfolio(tickers, total_investment, start="2022-01-01", end=None):
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')
    try:
        data = yf.download(tickers, start=start, end=end)['Adj Close'].dropna()
        mu = mean_historical_return(data)
        S = CovarianceShrinkage(data).ledoit_wolf()
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        weights_df = pd.DataFrame(list(cleaned_weights.items()), columns=["Ticker", "Weight"])
        weights_df = weights_df[weights_df['Weight'] > 0]
        weights_df['Investment Amount (₹)'] = weights_df['Weight'] * total_investment
        weights_df['Weight %'] = weights_df['Weight'] * 100
        weights_df['Stock'] = weights_df['Ticker'].map({v: k for k, v in TICKER_MAP.items()})
        return weights_df[['Stock', 'Ticker', 'Weight %', 'Investment Amount (₹)']].round(2)
    except Exception as e:
        return pd.DataFrame({'Error': [str(e)]})

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
# AI Commentary Generator
# ----------------------------
def generate_ai_commentary(risk_profile, selected_stocks, duration):
    sector_col = 'Sector' if 'Sector' in selected_stocks.columns else 'Stock'
    dominant_sector = selected_stocks[sector_col].mode().values[0]
    risk_summary = {
        "Conservative": "risk-averse with a focus on preserving capital and generating stable returns.",
        "Moderate": "balanced, aiming for a mix of growth and income while managing moderate risk.",
        "Aggressive": "growth-oriented, aiming for higher returns with an acceptance of market volatility."
    }
    return (
        f"Based on your risk profile of **{risk_profile}**, the recommended portfolio is "
        f"{risk_summary[risk_profile]} The portfolio shows significant allocation to "
        f"**{dominant_sector}**. Over a {duration}-year horizon, this strategy aligns well with "
        f"your investment goals."
    )

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender for Mutual Fund Managers")
st.markdown("Get stock allocations based on your client's risk profile with earnings forecasts under multiple market conditions.")

st.header("Client Profile")
age = st.slider("Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
diversify = st.checkbox("Diversify across risk levels")
strategy = st.radio("Recommendation Strategy", ["Basic AI", "Enhanced Scoring", "Optimized (Max Sharpe)"])

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile: {risk_profile}")
    st.info(f"Investment Allocation for ₹{investment_amount:,.0f}")

    if strategy == "Basic AI":
        recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=diversify)
    elif strategy == "Enhanced Scoring":
        recommended_stocks = enhanced_stock_selection(risk_profile, investment_amount)
    else:
        tickers = list(TICKER_MAP.values())
        recommended_stocks = optimize_portfolio(tickers, investment_amount)

    if not recommended_stocks.empty:
        st.markdown("### 📈 Recommended Portfolio")
        st.dataframe(recommended_stocks, use_container_width=True)

        if 'Investment Amount (₹)' in recommended_stocks.columns:
            fig1, ax1 = plt.subplots()
            ax1.pie(recommended_stocks['Investment Amount (₹)'], labels=recommended_stocks['Stock'], autopct='%1.1f%%')
            ax1.set_title("Investment Allocation")
            st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        ax2.bar(recommended_stocks['Stock'], recommended_stocks['Weight %'], color='skyblue')
        ax2.set_ylabel("Weight (%)")
        ax2.set_title("Portfolio Weights")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

        st.markdown("### 💰 Projected Earnings")
        earnings = simulate_earnings(investment_amount, duration)
        fig3, ax3 = plt.subplots()
        for col in earnings.columns[1:]:
            ax3.plot(earnings['Year'], earnings[col], label=col)
        ax3.legend()
        ax3.set_title("Projected Value Over Time")
        st.pyplot(fig3)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            recommended_stocks.to_excel(writer, sheet_name='Portfolio', index=False)
            earnings.to_excel(writer, sheet_name='Projections', index=False)
        st.download_button("📥 Download Excel Report", output.getvalue(), file_name="portfolio_report.xlsx")

        st.markdown("### 🤖 AI Commentary")
        commentary = generate_ai_commentary(risk_profile, recommended_stocks, duration)
        st.info(commentary)
    else:
        st.warning("⚠️ No suitable stocks found or optimization failed.")
