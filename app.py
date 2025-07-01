# Imports
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Constants
TICKER_MAP = {
    'TCS': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS',
    'Eternal Limited': 'ETERNAL.NS',
    'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'IRCTC': 'IRCTC.NS'
}

# Helper Functions
def fetch_live_data(stock_df):
    additional_data = []
    for stock in stock_df['Stock']:
        ticker_symbol = TICKER_MAP.get(stock)
        if not ticker_symbol:
            continue
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            additional_data.append({
                'Stock': stock,
                'Live Price (₹)': round(info.get('currentPrice', np.nan), 2),
                '52W High (₹)': round(info.get('fiftyTwoWeekHigh', np.nan), 2),
                '52W Low (₹)': round(info.get('fiftyTwoWeekLow', np.nan), 2),
                'Dividend Yield (%)': round(info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0, 2),
                'P/E Ratio': round(info.get('trailingPE', np.nan), 2),
                'Market Cap (₹ Cr)': round(info.get('marketCap', 0) / 1e7, 2),
                'Beta (Live)': round(info.get('beta', np.nan), 2)
            })
        except Exception:
            additional_data.append({
                'Stock': stock,
                'Live Price (₹)': np.nan,
                '52W High (₹)': np.nan,
                '52W Low (₹)': np.nan,
                'Dividend Yield (%)': np.nan,
                'P/E Ratio': np.nan,
                'Market Cap (₹ Cr)': np.nan,
                'Beta (Live)': np.nan
            })
    return pd.DataFrame(additional_data)

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

# ----------------- Start Streamlit -----------------
st.title("📊 AI-Based Stock Recommender + Backtester")

# Sidebar Inputs
st.sidebar.header("Client Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Highest Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)

# Generate Recommendation
if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{risk_profile}**")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=True)
    st.subheader("📈 Recommended Portfolio")
    st.dataframe(recommended_stocks)

    live_data = fetch_live_data(recommended_stocks)
    st.subheader("📉 Live Stock Data")
    st.dataframe(live_data)

    # Backtesting Section
    st.subheader("📊 Portfolio Backtesting (3 Months)")
    portfolio_weights = recommended_stocks.set_index("Stock")["Weight %"] / 100
    tickers = [TICKER_MAP[stock] for stock in portfolio_weights.index if stock in TICKER_MAP]

    end_date = datetime.today()
    start_date = end_date - timedelta(days=90)

    price_data = yf.download(tickers, start=start_date, end=end_date)['Close']
    price_data = price_data.dropna(axis=1)  # drop stocks with no data
    portfolio_weights = portfolio_weights[[s for s in portfolio_weights.index if TICKER_MAP[s] in price_data.columns]]
    normalized = price_data / price_data.iloc[0]
    portfolio_returns = (normalized * portfolio_weights).sum(axis=1)

    # Benchmark
    benchmark = yf.download("^NSEI", start=start_date, end=end_date)['Close']
    benchmark_norm = benchmark / benchmark.iloc[0]

    # Performance Metrics
    def compute_metrics(series):
        returns = series.pct_change().dropna()
        cumulative = series.iloc[-1]
        cagr = (cumulative / series.iloc[0]) ** (365 / len(series)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        drawdown = (series / series.cummax() - 1).min()
        return round(cagr*100,2), round(sharpe,2), round(drawdown*100,2), round(volatility*100,2)

    port_cagr, port_sharpe, port_dd, port_vol = compute_metrics(portfolio_returns)
    bench_cagr, bench_sharpe, bench_dd, bench_vol = compute_metrics(benchmark_norm)

    metrics_df = pd.DataFrame({
        "Metric": ["CAGR", "Sharpe Ratio", "Max Drawdown", "Volatility"],
        "Portfolio": [f"{port_cagr}%", port_sharpe, f"{port_dd}%", f"{port_vol}%"],
        "Market": [f"{bench_cagr}%", bench_sharpe, f"{bench_dd}%", f"{bench_vol}%"]
    })

    st.subheader("📊 Backtest Performance Comparison")
    st.dataframe(metrics_df)
