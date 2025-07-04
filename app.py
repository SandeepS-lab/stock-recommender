# PART 1/2

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, expected_returns, risk_models, objective_functions

# ----------------------------
# Ticker Map (14 NSE Stocks)
# ----------------------------
TICKER_MAP = {
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Asian Paints': 'ASIANPAINT.NS',
    'Larsen & Toubro': 'LT.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'ITC Ltd': 'ITC.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'State Bank of India': 'SBIN.NS'
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
# Stock Recommendation Logic
# ----------------------------
def get_stock_list(risk_profile, investment_amount, diversify=False):
    data = {
        'Stock': list(TICKER_MAP.keys()),
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 1.05, 0.95, 1.1, 0.9, 1.0, 1.2, 1.0, 1.05, 1.0, 0.95],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.0, 1.2, 0.95, 1.1, 1.0, 0.8, 1.1, 0.9, 1.0, 1.2],
        'Volatility': [0.18, 0.20, 0.19, 0.25, 0.22, 0.21, 0.19, 0.23, 0.20, 0.17, 0.24, 0.20, 0.21, 0.22],
        'Market Cap': ['Large']*14,
        'Risk Category': [
            'Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Moderate',
            'Moderate', 'Conservative', 'Moderate', 'Moderate',
            'Conservative', 'Aggressive', 'Conservative', 'Moderate', 'Aggressive'
        ]
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
# Robust yfinance Download with Retry + Logging
# ----------------------------
def safe_yf_download(tickers, start, end, max_attempts=2):
    st.write("📦 Requested Tickers:", tickers)
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(0, axis=1)
        data = data.ffill().bfill()
        data = data.dropna(axis=1, how='any')

        if data.empty and max_attempts > 0:
            st.warning("⚠️ First attempt failed. Retrying with fewer tickers...")
            return safe_yf_download(tickers[:3], start, end, max_attempts=max_attempts-1)

        st.write("✅ Final Columns:", data.columns.tolist())
        return data
    except Exception as e:
        st.error(f"❌ yfinance error: {e}")
        return pd.DataFrame()

# ----------------------------
# Streamlit Sidebar Inputs
# ----------------------------
st.set_page_config(layout="wide")
st.title("📈 AI-Based Stock Recommender for Fund Managers")

st.sidebar.header("🧾 Client Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Highest Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)
diversify = st.sidebar.checkbox("Diversify Across Risk Categories", value=False)
# ----------------------------
# Earnings Projection
# ----------------------------
def simulate_earnings(amount, years):
    rates = {'Bear (-5%)': -0.05, 'Base (8%)': 0.08, 'Bull (15%)': 0.15}
    df = pd.DataFrame({'Year': list(range(years + 1))})
    for label, rate in rates.items():
        df[label] = amount * ((1 + rate) ** df['Year'])
    return df

# ----------------------------
# Monte Carlo Simulation
# ----------------------------
def monte_carlo_simulation(initial_investment, expected_return, volatility, years, n_simulations=500):
    np.random.seed(42)
    results = np.zeros((n_simulations, years + 1))
    results[:, 0] = initial_investment
    for i in range(1, years + 1):
        random_returns = np.random.normal(expected_return, volatility, n_simulations)
        results[:, i] = results[:, i - 1] * (1 + random_returns)
    return results

# ----------------------------
# Generate Recommendation
# ----------------------------
if st.button("🚀 Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{risk_profile}**")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify)
    st.subheader("📊 Recommended Portfolio")
    st.dataframe(recommended_stocks)

    st.subheader("⚙️ Portfolio Optimization")
    opt_method = st.selectbox("Optimization Objective", ["Max Sharpe Ratio", "Max Return", "Min Volatility"])

    tickers = [TICKER_MAP[stock] for stock in recommended_stocks['Stock']]
    start_date = datetime.today() - timedelta(days=730)
    end_date = datetime.today()

    price_data = safe_yf_download(tickers, start=start_date, end=end_date)

    if not price_data.empty:
        mu = expected_returns.mean_historical_return(price_data)
        S = risk_models.sample_cov(price_data)
        ef = EfficientFrontier(mu, S)
        ef.add_objective(objective_functions.L2_reg, gamma=0.1)

        if opt_method == "Max Sharpe Ratio":
            weights = ef.max_sharpe()
        elif opt_method == "Max Return":
            weights = ef.max_quadratic_utility()
        else:
            weights = ef.min_volatility()

        cleaned_weights = ef.clean_weights()
        opt_df = pd.DataFrame({
            'Stock': list(cleaned_weights.keys()),
            'Weight %': [round(w * 100, 2) for w in cleaned_weights.values()]
        }).query("`Weight %` > 0")
        opt_df['Investment Amount (₹)'] = (opt_df['Weight %'] / 100) * investment_amount
        st.dataframe(opt_df)

        recommended_stocks = recommended_stocks[recommended_stocks['Stock'].isin(opt_df['Stock'])]
        recommended_stocks = recommended_stocks.drop(columns=['Weight %', 'Investment Amount (₹)'])
        recommended_stocks = recommended_stocks.merge(opt_df, on='Stock')

    # Earnings
    st.subheader("📈 Projected Earnings Scenarios")
    earnings = simulate_earnings(investment_amount, duration)
    st.line_chart(earnings.set_index("Year"))

    # Monte Carlo
    st.subheader("🧪 Monte Carlo Simulation (500 Runs)")
    avg_return = (recommended_stocks['Sharpe Ratio'] * recommended_stocks['Weight %'] / 100).sum()
    avg_vol = (recommended_stocks['Volatility'] * recommended_stocks['Weight %'] / 100).sum()
    mc = monte_carlo_simulation(investment_amount, avg_return, avg_vol, duration)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(min(100, mc.shape[0])):
        ax.plot(range(duration + 1), mc[i], color='gray', alpha=0.1)
    median = np.percentile(mc, 50, axis=0)
    p10 = np.percentile(mc, 10, axis=0)
    p90 = np.percentile(mc, 90, axis=0)
    ax.plot(median, label="Median", color='blue')
    ax.fill_between(range(duration + 1), p10, p90, color='blue', alpha=0.2, label='10%-90% Range')
    ax.set_title("Monte Carlo Simulation of Portfolio Value")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    st.pyplot(fig)

    # Backtest
    st.subheader("📉 Portfolio Backtest (Last 24 Months)")
    portfolio_weights = recommended_stocks.set_index("Stock")["Weight %"] / 100
    tickers = [TICKER_MAP[s] for s in portfolio_weights.index]

    price_bt = safe_yf_download(tickers, start_date, end_date)
    benchmark = safe_yf_download(["NIFTYBEES.NS"], start_date, end_date)

    if not price_bt.empty and not portfolio_weights.empty:
        norm = price_bt / price_bt.iloc[0]
        portfolio = (norm * portfolio_weights.values).sum(axis=1)
        bench = benchmark / benchmark.iloc[0]

        daily = portfolio.pct_change().dropna()
        sharpe = (daily.mean() / daily.std()) * np.sqrt(252)
        vol = daily.std() * np.sqrt(252)
        cum = (1 + daily).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()

        bt_df = pd.DataFrame({
            "Portfolio": portfolio,
            "NIFTYBEES ETF": bench.squeeze()
        })
        st.line_chart(bt_df)
        st.markdown(f"📊 **Portfolio Return**: {round((portfolio[-1] - 1) * 100, 2)}%")
        st.markdown(f"📈 **Sharpe Ratio**: {sharpe:.2f}")
        st.markdown(f"📉 **Volatility**: {vol:.2%}")
        st.markdown(f"💥 **Max Drawdown**: {max_dd:.2%}")
    else:
        st.error("🚫 No valid stock data available for backtest.")

    # Debug
    with st.expander("🛠 Debug Log"):
        st.write("Portfolio Weights:", portfolio_weights)
        st.write("Downloaded Price Columns:", price_bt.columns.tolist() if not price_bt.empty else [])
        st.write("Final Portfolio Shape:", price_bt.shape)

# ----------------------------
# Optional Historical Viewer
# ----------------------------
if st.checkbox("📜 Show Historical Stock Data (Last 3 Months)"):
    st.subheader("📜 Historical Stock Data")
    start_hist = datetime.today() - timedelta(days=90)
    end_hist = datetime.today()

    for stock, ticker in TICKER_MAP.items():
        st.markdown(f"### {stock} ({ticker})")
        try:
            hist = yf.download(ticker, start=start_hist, end=end_hist)
            if not hist.empty:
                st.dataframe(hist.tail(5))
            else:
                st.warning(f"No data for {stock}")
        except Exception as e:
            st.error(f"❌ Failed to fetch {stock}: {e}")
