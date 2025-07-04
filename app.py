# === Part 1 of 2 ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# ----------------------------
# Ticker Map (14 Stocks with 2-Year History)
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
# Compute Real Metrics from yfinance
# ----------------------------
def compute_financial_metrics(tickers):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=730)
    prices = yf.download(tickers, start=start_date, end=end_date)['Close']

    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(0, axis=1)

    prices = prices.dropna(axis=1, how='any')
    daily_returns = prices.pct_change().dropna()
    
    benchmark = yf.download("NIFTYBEES.NS", start=start_date, end=end_date)['Close']
    benchmark_returns = benchmark.pct_change().dropna()
    
    metrics = []
    for stock in prices.columns:
        returns = daily_returns[stock]
        excess_returns = returns - 0.07 / 252
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        volatility = returns.std() * np.sqrt(252)
        covariance = np.cov(returns, benchmark_returns.loc[returns.index])[0][1]
        benchmark_var = benchmark_returns.var()
        beta = covariance / benchmark_var if benchmark_var != 0 else np.nan
        metrics.append({
            'Stock': stock,
            'Sharpe Ratio': round(sharpe, 2),
            'Volatility': round(volatility, 4),
            'Beta': round(beta, 2)
        })

    return pd.DataFrame(metrics)

# ----------------------------
# Stock Recommendation Logic
# ----------------------------
def get_stock_list(risk_profile, investment_amount, diversify=False):
    tickers = list(TICKER_MAP.values())
    metric_df = compute_financial_metrics(tickers)
    reverse_map = {v: k for k, v in TICKER_MAP.items()}
    metric_df['Stock'] = metric_df['Stock'].map(reverse_map)

    risk_map = {
        'Conservative': (0, 0.95),
        'Moderate': (0.95, 1.1),
        'Aggressive': (1.1, float('inf'))
    }
    low, high = risk_map[risk_profile]
    filtered = metric_df[(metric_df['Sharpe Ratio'] >= low) & (metric_df['Sharpe Ratio'] < high)].copy()

    if diversify:
        portions = {'Conservative': 0.33, 'Moderate': 0.33, 'Aggressive': 0.34}
        result = []
        for profile, weight in portions.items():
            l, h = risk_map[profile]
            sub = metric_df[(metric_df['Sharpe Ratio'] >= l) & (metric_df['Sharpe Ratio'] < h)].copy()
            sub['Score'] = sub['Sharpe Ratio'] / sub['Beta']
            sub['Weight %'] = sub['Score'] / sub['Score'].sum() * weight * 100
            sub['Investment Amount (₹)'] = (sub['Weight %'] / 100) * investment_amount
            result.append(sub)
        final = pd.concat(result)
    else:
        if filtered.empty:
            filtered = metric_df.copy()
        filtered['Score'] = filtered['Sharpe Ratio'] / filtered['Beta']
        filtered['Weight %'] = filtered['Score'] / filtered['Score'].sum() * 100
        filtered['Investment Amount (₹)'] = (filtered['Weight %'] / 100) * investment_amount
        final = filtered

    return final.round(2).drop(columns=['Score'])
# === Part 2 of 2 ===

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
# Streamlit UI
# ----------------------------
st.title("📈 AI-Based Stock Recommender for Fund Managers")

st.sidebar.header("Client Profile Input")
age = st.sidebar.number_input("Age", 18, 100, 30)
income = st.sidebar.number_input("Monthly Income (₹)", 10000, 200000, 50000, step=5000)
dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
qualification = st.sidebar.selectbox("Highest Qualification", ["Undergraduate", "Postgraduate", "Professional"])
duration = st.sidebar.number_input("Investment Duration (Years)", 1, 30, 5)
investment_type = st.sidebar.selectbox("Investment Type", ["Lumpsum", "SIP"])
investment_amount = st.sidebar.number_input("Investment Amount (₹)", 10000, 10000000, 100000)
diversify = st.sidebar.checkbox("Diversify Across Risk Categories", value=False)

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"🧠 Risk Profile: **{risk_profile}**")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=diversify)
    st.subheader("📊 Recommended Portfolio")
    st.dataframe(recommended_stocks)

    st.subheader("📈 Projected Earnings Scenarios")
    earning_df = simulate_earnings(investment_amount, duration)
    st.line_chart(earning_df.set_index("Year"))

    st.subheader("🧪 Monte Carlo Simulation (500 Scenarios)")
    avg_return = (recommended_stocks['Sharpe Ratio'] * recommended_stocks['Weight %'] / 100).sum()
    avg_volatility = (recommended_stocks['Volatility'] * recommended_stocks['Weight %'] / 100).sum()
    mc_results = monte_carlo_simulation(investment_amount, avg_return, avg_volatility, duration)

    fig4, ax4 = plt.subplots(figsize=(10, 5))
    for i in range(min(100, mc_results.shape[0])):
        ax4.plot(range(duration + 1), mc_results[i], color='grey', alpha=0.1)
    median = np.percentile(mc_results, 50, axis=0)
    p10 = np.percentile(mc_results, 10, axis=0)
    p90 = np.percentile(mc_results, 90, axis=0)
    ax4.plot(median, color='blue', label='Median Projection')
    ax4.fill_between(range(duration + 1), p10, p90, color='blue', alpha=0.2, label='10%-90% Confidence Interval')
    ax4.set_title("Monte Carlo Simulation of Portfolio Value")
    ax4.set_xlabel("Year")
    ax4.set_ylabel("Portfolio Value (₹)")
    ax4.legend()
    st.pyplot(fig4)

    st.subheader("📉 Portfolio Backtest vs NIFTYBEES (Last 24 Months)")
    portfolio_weights = recommended_stocks.set_index("Stock")["Weight %"] / 100
    tickers = [TICKER_MAP[stock] for stock in portfolio_weights.index if stock in TICKER_MAP]

    start_date = datetime.today() - timedelta(days=730)
    end_date = datetime.today()

    try:
        price_data = yf.download(tickers + ["NIFTYBEES.NS"], start=start_date, end=end_date)['Close']

        if isinstance(price_data.columns, pd.MultiIndex):
            price_data = price_data.droplevel(0, axis=1)

        price_data.dropna(axis=1, how='any', inplace=True)
        tickers = [ticker for ticker in tickers if ticker in price_data.columns]
        portfolio_weights = portfolio_weights[[stock for stock in portfolio_weights.index if TICKER_MAP[stock] in price_data.columns]]

        portfolio_data = price_data[tickers]
        benchmark_data = price_data["NIFTYBEES.NS"]

        normalized = portfolio_data / portfolio_data.iloc[0]
        portfolio_returns = (normalized * portfolio_weights.values).sum(axis=1)
        benchmark_normalized = benchmark_data / benchmark_data.iloc[0]

        daily_returns = portfolio_returns.pct_change().dropna()
        benchmark_returns = benchmark_normalized.pct_change().dropna()

        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        benchmark_sharpe = (benchmark_returns.mean() / benchmark_returns.std()) * np.sqrt(252)
        volatility = daily_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_drawdown = (benchmark_cumulative - benchmark_cumulative.cummax()) / benchmark_cumulative.cummax()
        benchmark_max_drawdown = benchmark_drawdown.min()

        backtest_df = pd.DataFrame({
            "Portfolio": portfolio_returns,
            "NIFTYBEES (Benchmark)": benchmark_normalized
        })

        st.line_chart(backtest_df)
        st.markdown(f"📊 **Portfolio Return**: {round((portfolio_returns[-1]-1)*100, 2)}%")
        st.markdown(f"📉 **Benchmark Return**: {round((benchmark_normalized[-1]-1)*100, 2)}%")
        st.markdown(f"✨ **Sharpe Ratio**: {sharpe_ratio:.2f}")
        st.markdown(f"🔁 **Annualized Volatility**: {volatility:.2%}")
        st.markdown(f"💥 **Max Drawdown**: {max_drawdown:.2%}")

        comparison_data = {
            "Metric": ["Cumulative Return (%)", "Annualized Volatility (%)", "Sharpe Ratio", "Max Drawdown (%)"],
            "Portfolio": [
                round((portfolio_returns[-1] - 1) * 100, 2),
                round(volatility * 100, 2),
                round(sharpe_ratio, 2),
                round(max_drawdown * 100, 2)
            ],
            "NIFTYBEES (Benchmark)": [
                round((benchmark_normalized[-1] - 1) * 100, 2),
                round(benchmark_volatility * 100, 2),
                round(benchmark_sharpe, 2),
                round(benchmark_max_drawdown * 100, 2)
            ]
        }

        comparison_df = pd.DataFrame(comparison_data)
        st.subheader("📊 Portfolio vs Benchmark Comparison")
        st.table(comparison_df.set_index("Metric"))

    except Exception as e:
        st.error(f"⚠️ Backtest failed: {e}")
