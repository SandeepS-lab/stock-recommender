import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.express as px
from io import BytesIO
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ----------------------------
# Ticker Map
# ----------------------------
TICKER_MAP = {
    "TCS": "TCS.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Zomato": "ZOMATO.NS",
    "Reliance": "RELIANCE.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "IRCTC": "IRCTC.NS"
}

# ----------------------------
# Risk Profiling
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
# Enhanced Recommender
# ----------------------------
def enhanced_stock_selection(risk_profile, investment_amount):
    data = {
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises', 'Zomato', 'Reliance', 'Bajaj Finance', 'IRCTC'],
        'Sector': ['IT', 'Banking', 'IT', 'Infra', 'Tech', 'Energy', 'Finance', 'Travel'],
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'P/E': [29, 21, 27, 42, 80, 31, 37, 65],
        'ROE': [24, 18, 22, 12, 3, 20, 21, 17],
        'Risk Category': ['Conservative', 'Conservative', 'Moderate', 'Aggressive', 'Aggressive',
                          'Moderate', 'Moderate', 'Aggressive']
    }

    df = pd.DataFrame(data)

    df['Score'] = (
        (df['Sharpe Ratio'] / df['Beta']) * 0.4 +
        (1 / df['P/E']) * 0.2 +
        (df['ROE'] / 100) * 0.4
    )

    if risk_profile == "Conservative":
        risk_pool = ["Conservative", "Moderate"]
    elif risk_profile == "Moderate":
        risk_pool = ["Moderate", "Aggressive"]
    else:
        risk_pool = ["Aggressive", "Moderate"]

    filtered = df[df['Risk Category'].isin(risk_pool)].copy()
    filtered = filtered.sort_values(by='Score', ascending=False).head(7)
    filtered['Weight %'] = filtered['Score'] / filtered['Score'].sum() * 100
    filtered['Investment Amount (₹)'] = filtered['Weight %'] / 100 * investment_amount

    return filtered.round(2)

# ----------------------------
# Sharpe Ratio Optimizer
# ----------------------------
def optimize_sharpe_ratio(selected_stocks, investment_amount):
    tickers = selected_stocks['Stock'].map(TICKER_MAP).dropna().tolist()
    if len(tickers) < 2:
        return selected_stocks

    try:
        raw_data = yf.download(tickers, period="1y", progress=False)
        data = raw_data['Adj Close'].dropna(axis=1, how='any')
        if data.shape[1] < 2:
            return selected_stocks

        mu = expected_returns.mean_historical_return(data)
        S = risk_models.sample_cov(data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        selected_stocks = selected_stocks[selected_stocks['Stock'].map(TICKER_MAP).isin(cleaned_weights.keys())].copy()
        selected_stocks['Weight %'] = selected_stocks['Stock'].map(cleaned_weights).fillna(0) * 100
        selected_stocks['Investment Amount (₹)'] = selected_stocks['Weight %'] / 100 * investment_amount

        return selected_stocks.round(2)

    except Exception:
        return selected_stocks

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
st.set_page_config(page_title="Sharpe Optimizer Recommender", layout="centered")
st.title("📊 Sharpe Optimized Stock Recommender")
st.markdown("AI-based stock selection powered by Sharpe Ratio and client profiling.")

# UI Inputs
st.header("Client Profile")
age = st.slider("Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
strategy = st.radio("Strategy", ["Enhanced Scoring", "Sharpe Optimized Portfolio (MPT)"])

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile Identified: **{risk_profile}**")

    debug_info = ""

    stocks = enhanced_stock_selection(risk_profile, investment_amount)
    if strategy == "Sharpe Optimized Portfolio (MPT)":
        optimized = optimize_sharpe_ratio(stocks, investment_amount)
        if optimized.equals(stocks):
            st.warning("⚠️ Sharpe optimization could not be applied. Showing enhanced scoring.")
        else:
            stocks = optimized

    st.markdown("### 📌 Portfolio Recommendation")
    st.dataframe(stocks, use_container_width=True)

    # Pie Chart
    fig1, ax1 = plt.subplots()
    ax1.pie(stocks['Investment Amount (₹)'], labels=stocks['Stock'], autopct='%1.1f%%')
    ax1.set_title("Investment Allocation")
    st.pyplot(fig1)

    # Risk vs Return using Plotly
    st.markdown("### 📉 Risk vs Return (Interactive)")
    try:
        tickers = stocks['Stock'].map(TICKER_MAP).tolist()
        raw_data = yf.download(tickers, period="1y", progress=False)
        data = raw_data['Adj Close'].dropna(axis=1, how='any')

        if data.shape[1] < 1:
            raise ValueError("Not enough historical price data for selected stocks.")

        returns = data.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        exp_ret = returns.mean() * 252
        sharpe = exp_ret / vol

        plot_df = pd.DataFrame({
            "Ticker": vol.index.str.replace(".NS", "", regex=False),
            "Volatility": vol.values,
            "Expected Return": exp_ret.values,
            "Sharpe Ratio": sharpe.values
        })

        fig = px.scatter(
            plot_df,
            x="Volatility",
            y="Expected Return",
            color="Sharpe Ratio",
            size="Sharpe Ratio",
            hover_name="Ticker",
            color_continuous_scale="RdYlGn",
            title="Risk vs Return Bubble Chart"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        debug_info = str(e)
        st.error("⚠️ Could not plot Risk vs Return chart due to an error.")

    # Earnings Simulation
    earnings = simulate_earnings(investment_amount, duration)
    st.markdown("### 📈 Projected Portfolio Value")
    fig2, ax2 = plt.subplots()
    for col in earnings.columns[1:]:
        ax2.plot(earnings['Year'], earnings[col], label=col)
    ax2.legend()
    st.pyplot(fig2)

    # Excel Download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        stocks.to_excel(writer, sheet_name='Portfolio', index=False)
        earnings.to_excel(writer, sheet_name='Earnings', index=False)
    st.download_button("📥 Download Excel Report", data=output.getvalue(), file_name="recommendation.xlsx")

    # Debug log
    if debug_info:
        with st.expander("🛠 Show Debug Log"):
            st.code(debug_info)
