import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import yfinance as yf

# ----------------------------
# Ticker Map for Yahoo Finance
# ----------------------------
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
        'Stock': ['TCS', 'HDFC Bank', 'Infosys', 'Adani Enterprises', 'Zomato',
                  'Reliance Industries', 'Bajaj Finance', 'IRCTC'],
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
# Backtesting Logic
# ----------------------------
def backtest_portfolio(stocks, weights, start="2020-01-01", end=None):
    if end is None:
        end = pd.Timestamp.today().strftime('%Y-%m-%d')

    tickers = [TICKER_MAP.get(stock) for stock in stocks]
    valid = [t for t in tickers if t is not None]

    if not valid:
        return None, None, {"Error": "No valid tickers found for backtest."}, 0, 0, 0

    try:
        data = yf.download(valid, start=start, end=end)['Adj Close'].dropna()
        returns = data.pct_change().dropna()
        weights = weights[:len(data.columns)]
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative = (1 + portfolio_returns).cumprod()

        nifty = yf.download("^NSEI", start=start, end=end)['Adj Close'].pct_change().dropna()
        nifty_cumulative = (1 + nifty).cumprod()

        total_return = cumulative.iloc[-1] - 1
        years = (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25
        cagr = (cumulative.iloc[-1]) ** (1 / years) - 1
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        drawdown = (cumulative / cumulative.cummax()) - 1
        max_drawdown = drawdown.min()

        expected_value = (1 + cagr) ** years

        metrics = {
            "CAGR": f"{cagr*100:.2f}%",
            "Max Drawdown": f"{max_drawdown*100:.2f}%",
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Volatility": f"{volatility*100:.2f}%",
            "Expected Growth Multiple": f"{expected_value:.2f}x"
        }
        return cumulative, nifty_cumulative, metrics, cagr, sharpe, expected_value
    except Exception as e:
        return None, None, {"Error": str(e)}, 0, 0, 0

# ----------------------------
# Streamlit App
# ----------------------------
st.set_page_config(page_title="AI Stock Recommender", layout="centered")
st.title("AI-Based Stock Recommender for Mutual Fund Managers")

st.header("Client Profile")
age = st.slider("Age", 18, 75, 35)
income = st.number_input("Monthly Income (₹)", min_value=0, value=50000, step=5000)
investment_amount = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=10000)
dependents = st.selectbox("Dependents", [0, 1, 2, 3, 4])
qualification = st.selectbox("Qualification", ["Graduate", "Postgraduate", "Professional", "Other"])
duration = st.slider("Investment Duration (Years)", 1, 30, 5)
investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"])
diversify = st.checkbox("Diversify across risk levels")

if st.button("Generate Recommendation"):
    risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type)
    st.success(f"Risk Profile: {risk_profile}")

    recommended_stocks = get_stock_list(risk_profile, investment_amount, diversify=diversify)

    if not recommended_stocks.empty:
        st.markdown("### Recommended Portfolio")
        st.dataframe(recommended_stocks, use_container_width=True)

        fig1, ax1 = plt.subplots()
        ax1.pie(recommended_stocks['Investment Amount (₹)'], labels=recommended_stocks['Stock'], autopct='%1.1f%%')
        ax1.set_title("Investment Allocation")
        st.pyplot(fig1)

        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(recommended_stocks['Stock'], recommended_stocks['Investment Amount (₹)'], color='skyblue')
        ax_bar.set_ylabel("Investment Amount (₹)")
        ax_bar.set_title("Investment Amount by Stock")
        plt.xticks(rotation=45)
        st.pyplot(fig_bar)

        st.markdown("### Projected Earnings")
        earnings = simulate_earnings(investment_amount, duration)
        fig2, ax2 = plt.subplots()
        for col in earnings.columns[1:]:
            ax2.plot(earnings['Year'], earnings[col], label=col)
        ax2.set_title("Projected Growth")
        ax2.legend()
        st.pyplot(fig2)

        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            recommended_stocks.to_excel(writer, sheet_name='Portfolio', index=False)
            earnings.to_excel(writer, sheet_name='Projections', index=False)
        st.download_button("Download Excel Report", output.getvalue(), file_name="portfolio.xlsx")

        st.markdown("### Backtest Results")
        tickers = recommended_stocks['Stock'].tolist()
        weights = recommended_stocks['Weight %'].values / 100
        cum_ret, nifty_ret, metrics, cagr, sharpe, expected_value = backtest_portfolio(tickers, weights)

        if cum_ret is not None:
            fig_bt, ax_bt = plt.subplots()
            ax_bt.plot(cum_ret.index, cum_ret, label="Portfolio")
            ax_bt.plot(nifty_ret.index, nifty_ret, label="NIFTY 50", linestyle='--')
            ax_bt.set_title("Cumulative Returns")
            ax_bt.legend()
            st.pyplot(fig_bt)

            st.write("#### Backtest Summary")
            st.write(f"**CAGR**: {cagr * 100:.2f}%")
            st.write(f"**Sharpe Ratio**: {sharpe:.2f}")
            st.write(f"**Expected Portfolio Value in {duration} Years**: ₹{investment_amount * ((1 + cagr) ** duration):,.0f}")

            st.markdown("### Interpretation (AI Commentary)")
            interpretation = f"""
            Based on historical data, your selected portfolio has delivered a **CAGR of {cagr*100:.2f}%** over the last few years.
            If similar conditions persist, your ₹{investment_amount:,.0f} investment could grow to approximately ₹{investment_amount * ((1 + cagr) ** duration):,.0f} in {duration} years.

            A **Sharpe Ratio of {sharpe:.2f}** indicates that the portfolio has offered attractive risk-adjusted returns.
            This suggests a healthy balance of reward relative to volatility, making it suitable for your current risk profile.
            """
            st.info(interpretation)

            st.table(metrics)
        else:
            st.warning(metrics.get("Error", "Backtest failed."))
    else:
        st.warning("No suitable stocks found.")
