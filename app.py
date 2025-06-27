import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from io import BytesIO
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

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
# Live Stock Data (YFinance)
# ----------------------------
def get_live_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'Price': info.get('currentPrice'),
            '52 Week High': info.get('fiftyTwoWeekHigh'),
            '52 Week Low': info.get('fiftyTwoWeekLow'),
            'PE Ratio': info.get('trailingPE'),
            'Dividend Yield': info.get('dividendYield'),
            'Beta': info.get('beta')
        }
    except Exception as e:
        return {'Error': str(e)}

# ----------------------------
# Static Data Table
# ----------------------------
stock_mapping = {
    'TCS': 'TCS.NS', 'HDFC Bank': 'HDFCBANK.NS', 'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS', 'Zomato': 'ZOMATO.NS',
    'Reliance Industries': 'RELIANCE.NS', 'Bajaj Finance': 'BAJFINANCE.NS', 'IRCTC': 'IRCTC.NS'
}

stock_risk = {
    'TCS': 'Conservative', 'HDFC Bank': 'Moderate', 'Infosys': 'Moderate',
    'Adani Enterprises': 'Aggressive', 'Zomato': 'Aggressive',
    'Reliance Industries': 'Moderate', 'Bajaj Finance': 'Moderate', 'IRCTC': 'Aggressive'
}

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender with Live Data")

st.header("Enter Client Profile")
age = st.slider("Client Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
live_data_toggle = st.checkbox("Use Live Data from YFinance")

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile: {risk_profile}")
    st.info(f"Investment Allocation for ₹{investment_amount:,.0f}")

    filtered_stocks = [s for s, r in stock_risk.items() if r == risk_profile]
    if len(filtered_stocks) < 5:
        all_stocks = list(stock_risk.keys())
        for s in all_stocks:
            if s not in filtered_stocks:
                filtered_stocks.append(s)
            if len(filtered_stocks) == 5:
                break

    yf_symbols = [stock_mapping[s] for s in filtered_stocks]
    prices = yf.download(yf_symbols, period="1y")['Adj Close'].dropna()
    mu = mean_historical_return(prices)
    S = CovarianceShrinkage(prices).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    optimized_weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    weights = np.array([cleaned_weights.get(stock_mapping[s], 0) for s in filtered_stocks])
    investment_per_stock = (weights * investment_amount)

    portfolio = pd.DataFrame({
        'Stock': filtered_stocks,
        'Weight %': [round(w * 100, 2) for w in weights],
        'Investment Amount (₹)': [round(inv) for inv in investment_per_stock]
    })

    if live_data_toggle:
        extra_data = []
        for stock in portfolio['Stock']:
            yf_symbol = stock_mapping.get(stock)
            live_metrics = get_live_data(yf_symbol)
            extra_data.append(live_metrics)
        extra_df = pd.DataFrame(extra_data)
        portfolio = pd.concat([portfolio, extra_df], axis=1)

    st.markdown("### Recommended Portfolio")
    st.dataframe(portfolio, use_container_width=True)

    fig, ax = plt.subplots()
    ax.pie(portfolio['Investment Amount (₹)'], labels=portfolio['Stock'], autopct='%1.1f%%')
    ax.set_title("Investment Allocation")
    st.pyplot(fig)

    # Earnings simulation
    st.markdown("### Projected Earnings")
    rates = {'Bear (-5%)': -0.05, 'Base (8%)': 0.08, 'Bull (15%)': 0.15}
    projections = pd.DataFrame({'Year': list(range(0, duration + 1))})
    for name, rate in rates.items():
        projections[name] = investment_amount * ((1 + rate) ** projections['Year'])

    fig2, ax2 = plt.subplots()
    for col in projections.columns[1:]:
        ax2.plot(projections['Year'], projections[col], label=col)
    ax2.set_title("Projected Portfolio Value")
    ax2.legend()
    st.pyplot(fig2)

    # Download button
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        portfolio.to_excel(writer, sheet_name='Portfolio', index=False)
        projections.to_excel(writer, sheet_name='Projection', index=False)
    st.download_button("Download Excel Report", data=output.getvalue(), file_name="portfolio_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
