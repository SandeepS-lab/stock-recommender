# 📦 Imports
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions

# 📍 Ticker Map
TICKER_MAP = {
    'TCS': 'TCS.NS', 'HDFC Bank': 'HDFCBANK.NS', 'Infosys': 'INFY.NS',
    'Adani Enterprises': 'ADANIENT.NS', 'Reliance Industries': 'RELIANCE.NS',
    'Bajaj Finance': 'BAJFINANCE.NS', 'Asian Paints': 'ASIANPAINT.NS',
    'Larsen & Toubro': 'LT.NS', 'Axis Bank': 'AXISBANK.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS', 'Maruti Suzuki': 'MARUTI.NS',
    'ITC Ltd': 'ITC.NS', 'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'State Bank of India': 'SBIN.NS'
}

# 📋 Sidebar Inputs
st.sidebar.header("Client Profile")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)
diversify = st.sidebar.checkbox("Diversify Across Risk Categories", value=False)

# 📊 Risk Profile
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

# 🧠 Compute Metrics From YFinance
@st.cache_data
def compute_stock_metrics():
    start_date = datetime.today() - timedelta(days=730)
    end_date = datetime.today()
    tickers = list(TICKER_MAP.values())
    df = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1)
    df.dropna(axis=1, inplace=True)

    returns = df.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

    # Market proxy
    market_returns = returns.mean(axis=1)
    beta = {}
    for col in returns.columns:
        cov = np.cov(returns[col], market_returns)[0][1]
        beta[col] = cov / market_returns.var()

    stock_data = []
    for name, ticker in TICKER_MAP.items():
        if ticker in returns.columns:
            stock_data.append({
                'Stock': name,
                'Ticker': ticker,
                'Sharpe Ratio': round(sharpe[ticker], 2),
                'Volatility': round(volatility[ticker], 2),
                'Beta': round(beta[ticker], 2),
                'Risk Category': (
                    'Conservative' if beta[ticker] < 0.9 else
                    'Moderate' if beta[ticker] < 1.15 else
                    'Aggressive'
                )
            })

    return pd.DataFrame(stock_data)

# ✅ Stock Selection Function (Dynamic Scoring)
def get_stock_list(risk_profile, investment_amount, stock_metrics_df, diversify=False):
    df = stock_metrics_df.copy()

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
        selected['Score'] = selected['Sharpe Ratio'] / selected['Beta']
        selected['Weight %'] = selected['Score'] / selected['Score'].sum() * 100
        selected['Investment Amount (₹)'] = (selected['Weight %'] / 100) * investment_amount

    return selected.round(2).drop(columns=['Score'])

# 🧮 Earnings Projection
def simulate_earnings(amount, years):
    rates = {'Bear (-5%)': -0.05, 'Base (8%)': 0.08, 'Bull (15%)': 0.15}
    df = pd.DataFrame({'Year': list(range(years + 1))})
    for label, rate in rates.items():
        df[label] = amount * ((1 + rate) ** df['Year'])
    return df

# 🔁 Monte Carlo Simulation
def monte_carlo_simulation(initial_investment, expected_return, volatility, years, n_simulations=500):
    np.random.seed(42)
    results = np.zeros((n_simulations, years + 1))
    results[:, 0] = initial_investment
    for i in range(1, years + 1):
        random_returns = np.random.normal(expected_return, volatility, n_simulations)
        results[:, i] = results[:, i - 1] * (1 + random_returns)
    return results
# ✅ Main Action Button
if st.button("🚀 Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{risk_profile}**")

    stock_metrics_df = compute_stock_metrics()
    recommended_stocks = get_stock_list(risk_profile, investment_amount, stock_metrics_df, diversify)

    st.subheader("📊 Recommended Portfolio")
    st.dataframe(recommended_stocks)

    st.subheader("⚙️ Portfolio Optimization")
    opt_method = st.selectbox("Optimization Objective", ["Max Sharpe Ratio", "Max Return", "Min Volatility"])
    tickers = recommended_stocks['Ticker'].tolist()
    start_date = datetime.today() - timedelta(days=730)
    end_date = datetime.today()

    price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    price_data.dropna(axis=1, inplace=True)

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
            'Ticker': list(cleaned_weights.keys()),
            'Weight %': [round(w * 100, 2) for w in cleaned_weights.values()]
        }).query("`Weight %` > 0")
        opt_df['Investment Amount (₹)'] = (opt_df['Weight %'] / 100) * investment_amount
        opt_df['Stock'] = opt_df['Ticker'].map({v: k for k, v in TICKER_MAP.items()})
        st.dataframe(opt_df[['Stock', 'Ticker', 'Weight %', 'Investment Amount (₹)']])

        recommended_stocks = recommended_stocks[recommended_stocks['Ticker'].isin(opt_df['Ticker'])]
        recommended_stocks = recommended_stocks.drop(columns=['Weight %', 'Investment Amount (₹)'])
        recommended_stocks = recommended_stocks.merge(opt_df[['Ticker', 'Weight %', 'Investment Amount (₹)']], on='Ticker')

    # 📈 Projected Earnings
    st.subheader("📈 Projected Earnings Scenarios")
    earnings = simulate_earnings(investment_amount, duration)
    st.line_chart(earnings.set_index("Year"))

    # 🧪 Monte Carlo
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

    # 📉 Backtesting
    st.subheader("📉 Portfolio Backtest (Last 24 Months)")
    portfolio_weights = recommended_stocks.set_index("Ticker")["Weight %"] / 100
    tickers = portfolio_weights.index.tolist()
    price_bt = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    price_bt.dropna(axis=1, inplace=True)
    benchmark = yf.download("NIFTYBEES.NS", start=start_date, end=end_date)['Adj Close']

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
            "NIFTYBEES ETF": bench
        })
        st.line_chart(bt_df)
        st.markdown(f"📊 **Portfolio Return**: {round((portfolio[-1] - 1) * 100, 2)}%")
        st.markdown(f"📈 **Sharpe Ratio**: {sharpe:.2f}")
        st.markdown(f"📉 **Volatility**: {vol:.2%}")
        st.markdown(f"💥 **Max Drawdown**: {max_dd:.2%}")

        # 📋 Portfolio vs Market Comparison
        market_daily = bench.pct_change().dropna()
        market_sharpe = (market_daily.mean() / market_daily.std()) * np.sqrt(252)
        market_vol = market_daily.std() * np.sqrt(252)
        market_cum = (1 + market_daily).cumprod()
        market_dd = (market_cum - market_cum.cummax()) / market_cum.cummax()
        market_max_dd = market_dd.min()
        market_return = (bench[-1] - 1) * 100

        comparison_df = pd.DataFrame({
            "Metric": ["Cumulative Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"],
            "Portfolio": [
                round((portfolio[-1] - 1) * 100, 2),
                round(vol * 100, 2),
                round(sharpe, 2),
                round(max_dd * 100, 2)
            ],
            "Market (NIFTYBEES)": [
                round(market_return, 2),
                round(market_vol * 100, 2),
                round(market_sharpe, 2),
                round(market_max_dd * 100, 2)
            ]
        })
        st.subheader("📋 Portfolio vs Market Comparison")
        st.dataframe(comparison_df)

    else:
        st.error("🚫 No valid stock data available for backtest.")

    # 🛠 Debug
    with st.expander("🛠 Debug Log"):
        st.write("Portfolio Weights:", portfolio_weights)
        st.write("Downloaded Tickers:", tickers)
        st.write("Valid Price Data Columns:", price_bt.columns.tolist())

# 📜 Optional Historical Viewer
if st.checkbox("📜 Show Historical Stock Data (Last 3 Months)"):
    st.subheader("📜 Historical Stock Data")
    start_hist = datetime.today() - timedelta(days=90)
    end_hist = datetime.today()

    for name, ticker in TICKER_MAP.items():
        st.markdown(f"### {name} ({ticker})")
        try:
            hist = yf.download(ticker, start=start_hist, end=end_hist)
            if not hist.empty:
                st.dataframe(hist.tail(5))
            else:
                st.warning(f"No data for {name}")
        except Exception as e:
            st.error(f"❌ Failed to fetch {name}: {e}")
