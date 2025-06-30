import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
from nsepy import get_history

# ----------------------------
# Ticker Map
# ----------------------------
TICKER_MAP = {
    'TCS': 'TCS',
    'HDFC Bank': 'HDFCBANK',
    'Infosys': 'INFY',
    'Adani Enterprises': 'ADANIENT',
    'Eternal Limited': 'ETERNAL',
    'Reliance Industries': 'RELIANCE',
    'Bajaj Finance': 'BAJFINANCE',
    'IRCTC': 'IRCTC'
}

YFINANCE_MAP = {k: v + '.NS' for k, v in TICKER_MAP.items()}

# ----------------------------
# Live Stock Data (yfinance)
# ----------------------------
def fetch_live_data(stock_df):
    data = []
    for stock in stock_df['Stock']:
        ticker = YFINANCE_MAP.get(stock)
        if not ticker:
            continue
        try:
            info = yf.Ticker(ticker).info
            data.append({
                'Stock': stock,
                'Live Price (₹)': round(info.get('currentPrice', np.nan), 2),
                '52W High (₹)': round(info.get('fiftyTwoWeekHigh', np.nan), 2),
                '52W Low (₹)': round(info.get('fiftyTwoWeekLow', np.nan), 2),
                'Dividend Yield (%)': round(info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0, 2),
                'P/E Ratio': round(info.get('trailingPE', np.nan), 2),
                'Market Cap (₹ Cr)': round(info.get('marketCap', 0) / 1e7, 2),
                'Beta (Live)': round(info.get('beta', np.nan), 2)
            })
        except:
            data.append({
                'Stock': stock,
                'Live Price (₹)': np.nan,
                '52W High (₹)': np.nan,
                '52W Low (₹)': np.nan,
                'Dividend Yield (%)': np.nan,
                'P/E Ratio': np.nan,
                'Market Cap (₹ Cr)': np.nan,
                'Beta (Live)': np.nan
            })
    return pd.DataFrame(data)

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
# Recommender
# ----------------------------
def get_stock_list(risk_profile, investment_amount, diversify=False):
    df = pd.DataFrame({
        'Stock': list(TICKER_MAP.keys()),
        'Sharpe Ratio': [1.2, 1.0, 1.15, 0.85, 0.65, 1.05, 0.95, 0.75],
        'Beta': [0.9, 0.85, 1.1, 1.4, 1.8, 1.0, 1.2, 1.5],
        'Volatility': [0.18, 0.20, 0.19, 0.25, 0.30, 0.22, 0.21, 0.28],
        'Market Cap': ['Large', 'Large', 'Large', 'Mid', 'Small', 'Large', 'Mid', 'Mid'],
        'Risk Category': ['Conservative', 'Moderate', 'Moderate', 'Aggressive', 'Aggressive',
                          'Moderate', 'Moderate', 'Aggressive']
    })

    if diversify:
        portions = {'Conservative': 0.33, 'Moderate': 0.33, 'Aggressive': 0.34}
        dfs = []
        for cat, portion in portions.items():
            temp = df[df['Risk Category'] == cat].copy()
            temp['Score'] = temp['Sharpe Ratio'] / temp['Beta']
            temp['Weight %'] = temp['Score'] / temp['Score'].sum() * portion * 100
            temp['Investment Amount (₹)'] = (temp['Weight %'] / 100) * investment_amount
            dfs.append(temp)
        result = pd.concat(dfs)
    else:
        result = df[df['Risk Category'] == risk_profile].copy()
        if len(result) < 5:
            result = pd.concat([result, df[df['Risk Category'] != risk_profile].head(5 - len(result))])
        result['Score'] = result['Sharpe Ratio'] / result['Beta']
        result['Weight %'] = result['Score'] / result['Score'].sum() * 100
        result['Investment Amount (₹)'] = (result['Weight %'] / 100) * investment_amount

    return result.round(2).drop(columns='Score')

# ----------------------------
# NSEpy-Based 3-Month Backtest
# ----------------------------
def run_backtest(stocks_df):
    end_date = datetime.today().date()
    start_date = (end_date - timedelta(days=90))
    results = []

    for stock in stocks_df['Stock']:
        symbol = TICKER_MAP.get(stock)
        if not symbol:
            continue
        try:
            df = get_history(symbol=symbol, start=start_date, end=end_date)
            if df.empty:
                continue
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            cagr = ((end_price / start_price) ** (1 / (90 / 365)) - 1) * 100
            volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
            sharpe = cagr / (volatility if volatility else 1)
            results.append({
                'Stock': stock,
                'Start Price': round(start_price, 2),
                'End Price': round(end_price, 2),
                'CAGR (3M) %': round(cagr, 2),
                'Volatility %': round(volatility, 2),
                'Sharpe Ratio': round(sharpe, 2)
            })
        except:
            continue

    return pd.DataFrame(results)

# ----------------------------
# Earnings Simulation
# ----------------------------
def simulate_earnings(amount, years):
    df = pd.DataFrame({'Year': list(range(years + 1))})
    df['Bear (-5%)'] = amount * ((1 - 0.05) ** df['Year'])
    df['Base (8%)'] = amount * ((1 + 0.08) ** df['Year'])
    df['Bull (15%)'] = amount * ((1 + 0.15) ** df['Year'])
    return df

# ----------------------------
# Monte Carlo
# ----------------------------
def monte_carlo_simulation(initial, expected_return, volatility, years, n=500):
    np.random.seed(42)
    sim = np.zeros((n, years + 1))
    sim[:, 0] = initial
    for i in range(1, years + 1):
        rand = np.random.normal(loc=expected_return, scale=volatility, size=n)
        sim[:, i] = sim[:, i - 1] * (1 + rand)
    return sim

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📊 AI-Based Stock Recommender (NSEpy Backtest)")

st.sidebar.header("Client Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)

if st.button("Generate Recommendation"):
    profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{profile}**")

    selected_stocks = get_stock_list(profile, investment_amount, diversify=True)
    st.subheader("📈 Recommended Portfolio")
    st.dataframe(selected_stocks)

    st.subheader("📉 Live Data (via yfinance)")
    st.dataframe(fetch_live_data(selected_stocks))

    st.subheader("🕰️ Backtesting (3-Months via NSEpy)")
    bt = run_backtest(selected_stocks)
    if not bt.empty:
        st.dataframe(bt)
    else:
        st.warning("No data available for backtesting.")

    st.subheader("📈 Simulated Earnings")
    sim_df = simulate_earnings(investment_amount, duration)
    st.line_chart(sim_df.set_index("Year"))

    st.subheader("🧪 Monte Carlo Simulation")
    avg_ret = (selected_stocks['Sharpe Ratio'] * selected_stocks['Weight %'] / 100).sum()
    avg_vol = (selected_stocks['Volatility'] * selected_stocks['Weight %'] / 100).sum()
    mc = monte_carlo_simulation(investment_amount, avg_ret, avg_vol, duration)

    fig, ax = plt.subplots()
    for i in range(100):
        ax.plot(mc[i], color='grey', alpha=0.1)
    ax.plot(np.percentile(mc, 50, axis=0), color='blue', label='Median')
    ax.fill_between(range(duration + 1), np.percentile(mc, 10, axis=0), np.percentile(mc, 90, axis=0), color='blue', alpha=0.2, label='10-90% CI')
    ax.set_title("Monte Carlo Simulation")
    ax.set_xlabel("Year")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.legend()
    st.pyplot(fig)
