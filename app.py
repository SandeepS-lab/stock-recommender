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
# Fetch Live Stock Data
# ----------------------------
def fetch_live_data(stock_df):
    additional_data = []
    for stock in stock_df['Stock']:
        ticker_symbol = TICKER_MAP.get(stock)
        if not ticker_symbol:
            continue
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info or {}
            additional_data.append({
                'Stock': stock,
                'Live Price (₹)': round(info.get('last_price', np.nan), 2),
                '52W High (₹)': round(info.get('year_high', np.nan), 2),
                '52W Low (₹)': round(info.get('year_low', np.nan), 2),
                'Dividend Yield (%)': np.nan,
                'P/E Ratio': np.nan,
                'Market Cap (₹ Cr)': np.nan,
                'Beta (Live)': np.nan
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

    live_data = fetch_live_data(recommended_stocks)
    st.subheader("📉 Live Stock Data (via yfinance)")
    st.dataframe(live_data)

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

    st.subheader("📉 Portfolio Backtest (Last 24 Months)")
    portfolio_weights = recommended_stocks.set_index("Stock")["Weight %"] / 100
    tickers = [TICKER_MAP[stock] for stock in portfolio_weights.index if stock in TICKER_MAP]

    start_date = datetime.today() - timedelta(days=730)
    end_date = datetime.today()

    try:
        price_data = yf.download(tickers, start=start_date, end=end_date)['Close']

        if isinstance(price_data.columns, pd.MultiIndex):
            price_data = price_data.droplevel(0, axis=1)

        price_data.dropna(axis=1, how='any', inplace=True)

        valid_stocks = [stock for stock in portfolio_weights.index if TICKER_MAP[stock] in price_data.columns]
        tickers = [TICKER_MAP[stock] for stock in valid_stocks]
        portfolio_weights = portfolio_weights[valid_stocks]
        price_data = price_data[tickers]

        normalized = price_data / price_data.iloc[0]
        portfolio_returns = (normalized * portfolio_weights.values).sum(axis=1)
        market_returns = normalized.mean(axis=1)

        daily_returns = portfolio_returns.pct_change().dropna()
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        volatility = daily_returns.std() * np.sqrt(252)
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        backtest_df = pd.DataFrame({
            "Portfolio": portfolio_returns,
            "Market Average": market_returns
        })

        st.line_chart(backtest_df)
        st.markdown(f"📊 **Portfolio Return**: {round((portfolio_returns[-1]-1)*100, 2)}%")
        st.markdown(f"📉 **Market Return**: {round((market_returns[-1]-1)*100, 2)}%")
        st.markdown(f"✨ **Sharpe Ratio**: {sharpe_ratio:.2f}")
        st.markdown(f"🔁 **Annualized Volatility**: {volatility:.2%}")
        st.markdown(f"💥 **Max Drawdown**: {max_drawdown:.2%}")

    except Exception as e:
        st.error(f"⚠️ Backtest failed: {e}")

if st.checkbox("📜 Show Historical Stock Data (Last 3 Months)"):
    st.subheader("📜 Historical Stock Data")
    start_date = datetime.today() - timedelta(days=90)
    end_date = datetime.today()

    for stock_name, ticker in TICKER_MAP.items():
        st.markdown(f"### {stock_name} ({ticker})")
        try:
            hist_data = yf.download(ticker, start=start_date, end=end_date)
            if not hist_data.empty:
                st.dataframe(hist_data.tail(5))
            else:
                st.warning(f"No historical data found for {stock_name} ({ticker})")
        except Exception as e:
            st.error(f"Error fetching data for {stock_name}: {e}")
