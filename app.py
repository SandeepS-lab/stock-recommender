import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nsepy import get_history
from datetime import date, timedelta

# ----------------------------
# Ticker Map (NSE symbols)
# ----------------------------
TICKER_MAP = {
    'TCS': 'TCS',
    'HDFC Bank': 'HDFCBANK',
    'Infosys': 'INFY',
    'Adani Enterprises': 'ADANIENT',
    'Zomato': 'ZOMATO',
    'Reliance Industries': 'RELIANCE',
    'Bajaj Finance': 'BAJFINANCE',
    'IRCTC': 'IRCTC'
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

    if score <= 2:
        return "Conservative"
    elif score <= 5:
        return "Moderate"
    else:
        return "Aggressive"

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
# Earnings Simulation
# ----------------------------
def simulate_earnings(amount, years):
    rates = {'Bear (-5%)': -0.05, 'Base (8%)': 0.08, 'Bull (15%)': 0.15}
    result = pd.DataFrame({'Year': list(range(0, years + 1))})
    for label, rate in rates.items():
        result[label] = amount * ((1 + rate) ** result['Year'])
    return result

# ----------------------------
# Monte Carlo Simulation
# ----------------------------
def monte_carlo_simulation(initial_investment, expected_return, volatility, years, n_simulations=500):
    np.random.seed(42)
    simulations = np.zeros((n_simulations, years + 1))
    simulations[:, 0] = initial_investment
    for i in range(1, years + 1):
        random_returns = np.random.normal(loc=expected_return, scale=volatility, size=n_simulations)
        simulations[:, i] = simulations[:, i - 1] * (1 + random_returns)
    return simulations

# ----------------------------
# NSE-Based Backtest
# ----------------------------
def backtest_portfolio(stocks_df, investment_amount):
    st.subheader("📊 Backtesting Over Past 6 Months")
    end = date.today()
    start = end - timedelta(days=182)
    price_data = {}
    weights = {}

    for _, row in stocks_df.iterrows():
        symbol = TICKER_MAP.get(row['Stock'])
        try:
            df = get_history(symbol=symbol, start=start, end=end)
            series = df['Close']
            if series.isnull().all() or len(series) < 30:
                continue
            price_data[row['Stock']] = series
            weights[row['Stock']] = row['Weight %'] / 100
        except Exception:
            continue

    if not price_data:
        st.warning("No valid NSE data available for selected stocks.")
        return

    df_prices = pd.DataFrame(price_data).dropna()
    if df_prices.empty:
        st.warning("Insufficient data for backtesting.")
        return

    normalized = df_prices / df_prices.iloc[0]
    portfolio = normalized.dot(pd.Series(weights)) * investment_amount
    fig, ax = plt.subplots()
    ax.plot(portfolio.index, portfolio, label='Portfolio Value')
    ax.set_title("Portfolio Value Over 6 Months")
    ax.set_ylabel("Value (₹)")
    ax.legend()
    st.pyplot(fig)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 AI-Based Stock Recommender (NSE Data Only)")

st.sidebar.header("Client Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Highest Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{risk_profile}**")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=True)
    st.subheader("📈 Recommended Portfolio")
    st.dataframe(recommended_stocks)

    st.subheader("📈 Projected Earnings Scenarios")
    earning_df = simulate_earnings(investment_amount, duration)
    st.line_chart(earning_df.set_index("Year"))

    st.subheader("🧪 Monte Carlo Simulation (500 Scenarios)")
    avg_return = (recommended_stocks['Sharpe Ratio'] * recommended_stocks['Weight %'] / 100).sum()
    avg_volatility = (recommended_stocks['Volatility'] * recommended_stocks['Weight %'] / 100).sum()
    mc_results = monte_carlo_simulation(investment_amount, avg_return, avg_volatility, duration)

    fig_mc, ax_mc = plt.subplots(figsize=(10, 5))
    for i in range(min(100, mc_results.shape[0])):
        ax_mc.plot(range(duration + 1), mc_results[i], color='grey', alpha=0.1)
    median = np.percentile(mc_results, 50, axis=0)
    p10 = np.percentile(mc_results, 10, axis=0)
    p90 = np.percentile(mc_results, 90, axis=0)
    ax_mc.plot(median, color='blue', label='Median Projection')
    ax_mc.fill_between(range(duration + 1), p10, p90, color='blue', alpha=0.2, label='10%-90% Confidence')
    ax_mc.set_title("Monte Carlo Simulation")
    ax_mc.set_xlabel("Year")
    ax_mc.set_ylabel("Portfolio Value (₹)")
    ax_mc.legend()
    st.pyplot(fig_mc)

    backtest_portfolio(recommended_stocks, investment_amount)
