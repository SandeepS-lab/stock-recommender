import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt # Import matplotlib for plotting

# ----------------------------
# Constants and Configurations
# ----------------------------
LOOKBACK_PERIOD = "3y" # For historical data (e.g., 1 year, 3 years)
INTERVAL = "1wk"       # Data interval (e.g., "1d", "1wk", "1mo")
BENCHMARK_TICKER = "^NSEI" # Nifty 50 for Indian context
RISK_FREE_RATE_ANNUAL = 0.04 # Example risk-free rate (e.g., current FD rate)

# Map for risk scores to categories (adjust these based on your weighted scoring)
RISK_SCORE_THRESHOLDS = {
    "Conservative": 0.8,
    "Moderate": 1.8,
    "Aggressive": 100 # Effectively no upper limit for aggressive
}

# Example Indian Stock Tickers (Expand this list significantly in real app)
INDIAN_STOCK_TICKERS = [
    'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS', 'ICICIBANK.NS',
    'LT.NS', 'ITC.NS', 'SBIN.NS', 'ASIANPAINT.NS', 'KOTAKBANK.NS',
    'AXISBANK.NS', 'MARUTI.NS', 'ULTRACEMCO.NS', 'BHARTIARTL.NS',
    'ADANIENT.NS', 'BAJFINANCE.NS', 'IRCTC.NS', 'ZOMATO.NS',
    'DMART.NS', 'NYKAA.NS', 'PAYTM.NS' # More volatile/growth examples
]

# ----------------------------
# Risk Profiling Logic
# ----------------------------

def get_risk_profile(age, income, dependents, qualification, duration, investment_type, volatility_comfort):
    score = 0
    # Assign weights to factors - calibrated for more granular impact
    weights = {
        "age": 0.20,
        "income": 0.20,
        "dependents": 0.10,
        "qualification": 0.05,
        "duration": 0.25,
        "investment_type": 0.10,
        "volatility_comfort": 0.10 # New factor
    }

    # Age: Younger can take more risk
    if age < 30:
        score += 2 * weights["age"]
    elif age < 45:
        score += 1 * weights["age"]
    # 45+ gives 0 from age perspective

    # Income: Higher income, higher risk capacity
    if income > 150000: # Higher bracket
        score += 2 * weights["income"]
    elif income > 75000:
        score += 1 * weights["income"]

    # Dependents: More dependents, less risk capacity
    if dependents >= 3:
        score -= 1 * weights["dependents"]
    elif dependents == 2:
        score -= 0.5 * weights["dependents"]

    # Qualification: Indicates financial literacy/understanding of risk
    if qualification in ["Postgraduate", "Professional"]:
        score += 1 * weights["qualification"]

    # Duration: Longer duration, more capacity for risk
    if duration >= 10:
        score += 2 * weights["duration"]
    elif duration >= 5:
        score += 1 * weights["duration"]

    # Investment Type: SIP implies averaged entry, slightly lower risk profile
    if investment_type == "SIP":
        score += 0.5 * weights["investment_type"] # Small positive for SIP, as it smooths out volatility

    # Volatility Comfort: Direct measure of psychological risk tolerance
    # Scale 1 (Very Uncomfortable) to 5 (Very Comfortable)
    score += (volatility_comfort - 1) * 0.5 * weights["volatility_comfort"] # Scale 0 to 2 for weight

    # Determine risk profile based on total score
    if score <= RISK_SCORE_THRESHOLDS["Conservative"]:
        return "Conservative"
    elif score <= RISK_SCORE_THRESHOLDS["Moderate"]:
        return "Moderate"
    else:
        return "Aggressive"

# ----------------------------
# Stock Data & Recommendation Logic
# ----------------------------

@st.cache_data(ttl=timedelta(hours=4)) # Cache data for 4 hours to reduce API calls
def get_stock_metrics(tickers, period, interval, benchmark_ticker, risk_free_rate_annual):
    stock_data = {}
    st.info(f"Downloading historical data for {len(tickers)} stocks and benchmark ({benchmark_ticker}). This might take a moment...")

    try:
        benchmark_df = yf.download(benchmark_ticker, period=period, interval=interval)['Adj Close']
        benchmark_returns = benchmark_df.pct_change().dropna()
        if benchmark_returns.empty:
            st.warning(f"Could not get sufficient benchmark returns for {benchmark_ticker}.")
            return {} # Return empty if benchmark is critically flawed

        # Annualize risk-free rate to match interval frequency
        # For weekly data (52 weeks in a year), risk-free rate per period = (1 + annual_rate)^(1/52) - 1
        if interval == "1wk":
            risk_free_rate_per_period = (1 + risk_free_rate_annual)**(1/52) - 1
        elif interval == "1d":
            risk_free_rate_per_period = (1 + risk_free_rate_annual)**(1/252) - 1 # Approx 252 trading days
        else: # Fallback for other intervals (e.g., monthly)
            risk_free_rate_per_period = (1 + risk_free_rate_annual)**(1/12) - 1 if interval == "1mo" else risk_free_rate_annual / 252


    except Exception as e:
        st.error(f"Error downloading benchmark data for {benchmark_ticker}: {e}. Cannot calculate Sharpe/Beta without benchmark.")
        return {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval)['Adj Close']
            if df.empty or len(df) < 2:
                st.warning(f"Could not download sufficient data for {ticker}. Skipping metrics.")
                continue

            returns = df.pct_change().dropna()

            # Align stock and benchmark returns
            # Use inner join to ensure only common dates are considered
            aligned_returns = pd.concat([returns, benchmark_returns], axis=1, join='inner').dropna()
            if aligned_returns.empty or len(aligned_returns) < 2:
                st.warning(f"Not enough aligned data for Sharpe/Beta for {ticker}. Skipping.")
                continue

            stock_returns_aligned = aligned_returns.iloc[:, 0]
            benchmark_returns_aligned = aligned_returns.iloc[:, 1]

            # Annualize volatility based on the number of periods in a year
            annualization_factor = np.sqrt(len(stock_returns_aligned) / (benchmark_df.index[-1] - benchmark_df.index[0]).days * 365.25)
            if np.isinf(annualization_factor) or np.isnan(annualization_factor): # Handle cases where period is too short or date calculation fails
                 if interval == "1wk": annualization_factor = np.sqrt(52)
                 elif interval == "1d": annualization_factor = np.sqrt(252)
                 elif interval == "1mo": annualization_factor = np.sqrt(12)
                 else: annualization_factor = 1 # Fallback
            
            volatility = stock_returns_aligned.std() * annualization_factor
            
            beta = np.nan
            sharpe_ratio_annualized = np.nan

            if stock_returns_aligned.std() > 0 and benchmark_returns_aligned.std() > 0:
                covariance = stock_returns_aligned.cov(benchmark_returns_aligned)
                benchmark_variance = benchmark_returns_aligned.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan

                excess_returns_per_period = stock_returns_aligned.mean() - risk_free_rate_per_period
                sharpe_ratio_per_period = excess_returns_per_period / (stock_returns_aligned.std() if stock_returns_aligned.std() != 0 else np.nan)
                
                sharpe_ratio_annualized = sharpe_ratio_per_period * annualization_factor

            # Rough Market Cap approximation (requires external data for accuracy)
            # For a real app, integrate a financial data API that provides Market Cap.
            # Here, we'll assign 'Market Cap' based on a predefined assumption or Beta for simplicity.
            market_cap_category = "Mid"
            if beta < 0.9:
                market_cap_category = "Large"
            elif beta > 1.3:
                market_cap_category = "Small"

            stock_data[ticker] = {
                'Sharpe Ratio': sharpe_ratio_annualized if not np.isnan(sharpe_ratio_annualized) else 0, # Default to 0 if NaN
                'Beta': beta if not np.isnan(beta) else 1.0, # Default to 1.0 if NaN
                'Volatility': volatility if not np.isnan(volatility) else 0.3, # Default to average volatility
                'Market Cap': market_cap_category # Needs actual data for accuracy
            }

        except Exception as e:
            st.warning(f"Error processing data for {ticker}: {e}")
            continue
    return stock_data

def assign_risk_category_to_stock(beta, volatility, sharpe_ratio):
    # More nuanced assignment based on dynamically calculated metrics
    # Adjust thresholds as needed based on your data and desired categorization
    if beta < 0.85 and volatility < 0.18 and sharpe_ratio > 1.0:
        return "Conservative"
    elif 0.85 <= beta <= 1.2 and 0.18 <= volatility <= 0.25 and sharpe_ratio > 0.6:
        return "Moderate"
    elif beta > 1.2 or volatility > 0.25 or sharpe_ratio <= 0.6: # High beta or high volatility or low Sharpe
        return "Aggressive"
    else: # Fallback for anything in between
        return "Moderate"

def get_stock_recommendations(risk_profile, investment_amount):
    live_metrics = get_stock_metrics(INDIAN_STOCK_TICKERS, LOOKBACK_PERIOD, INTERVAL, BENCHMARK_TICKER, RISK_FREE_RATE_ANNUAL)

    if not live_metrics:
        st.error("Could not fetch stock metrics. Please try again later or check network connection.")
        return pd.DataFrame()

    df_list = []
    for ticker, metrics in live_metrics.items():
        # Ensure that calculated metrics are valid before adding to list
        if not np.isnan(metrics['Sharpe Ratio']) and not np.isnan(metrics['Beta']) and not np.isnan(metrics['Volatility']):
            risk_cat_stock = assign_risk_category_to_stock(metrics['Beta'], metrics['Volatility'], metrics['Sharpe Ratio'])
            df_list.append({
                'Stock': ticker.replace('.NS', ''), # Clean ticker name for display
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Beta': metrics['Beta'],
                'Volatility': metrics['Volatility'],
                'Market Cap': metrics['Market Cap'],
                'Risk Category (Stock)': risk_cat_stock
            })
    
    if not df_list:
        st.warning("No valid stock data could be processed from the selected tickers.")
        return pd.DataFrame()

    df = pd.DataFrame(df_list)

    # Filter stocks based on client's risk profile
    selected_stocks = df[df['Risk Category (Stock)'] == risk_profile].copy()

    if selected_stocks.empty:
        st.warning(f"No suitable stocks directly matching the '{risk_profile}' profile found from the fetched data. Trying a broader match.")
        # Fallback: If no direct match, try to find a slightly less strict match
        if risk_profile == "Conservative":
            selected_stocks = df[df['Risk Category (Stock)'].isin(["Conservative", "Moderate"])].copy()
        elif risk_profile == "Moderate":
            selected_stocks = df[df['Risk Category (Stock)'].isin(["Conservative", "Moderate", "Aggressive"])].copy()
        elif risk_profile == "Aggressive":
            selected_stocks = df[df['Risk Category (Stock)'].isin(["Moderate", "Aggressive"])].copy()
        
        if selected_stocks.empty:
            st.warning("Still no suitable stocks found after broadening the search. Consider adjusting your stock universe or risk category definitions.")
            return pd.DataFrame()

    # Sort based on Sharpe Ratio (higher is better)
    selected_stocks = selected_stocks.sort_values(by='Sharpe Ratio', ascending=False)

    # Portfolio Allocation Logic (more refined than simple inverse beta)
    if not selected_stocks.empty:
        # Limit to top N stocks for practical portfolio size
        num_stocks_to_recommend = 5 # Default
        if risk_profile == "Conservative":
            num_stocks_to_recommend = min(8, len(selected_stocks)) # More diversification for conservative
        elif risk_profile == "Moderate":
            num_stocks_to_recommend = min(7, len(selected_stocks))
        else: # Aggressive
            num_stocks_to_recommend = min(6, len(selected_stocks)) # More concentrated for aggressive

        selected_stocks = selected_stocks.head(num_stocks_to_recommend)

        # Allocate based on Sharpe Ratio and Risk Profile
        if risk_profile == "Conservative":
            # Higher weight to lower Beta, higher Sharpe
            selected_stocks['Allocation Score'] = selected_stocks['Sharpe Ratio'] / (selected_stocks['Beta'] + 0.1)
        elif risk_profile == "Moderate":
            # Balance between Sharpe and Beta
            selected_stocks['Allocation Score'] = selected_stocks['Sharpe Ratio'] * (1 / (selected_stocks['Beta']**0.5))
        else: # Aggressive
            # Higher weight to higher Sharpe, less penalty for Beta (or even positive correlation with Beta)
            selected_stocks['Allocation Score'] = selected_stocks['Sharpe Ratio'] * (selected_stocks['Beta'] + 0.5) # Add 0.5 to avoid very low multipliers
            
        # Ensure allocation score is positive to avoid issues with sum
        selected_stocks['Allocation Score'] = selected_stocks['Allocation Score'].apply(lambda x: max(x, 0.01))

        # Normalize weights
        total_allocation_score = selected_stocks['Allocation Score'].sum()
        if total_allocation_score > 0:
            selected_stocks['Weight %'] = selected_stocks['Allocation Score'] / total_allocation_score * 100
        else:
            selected_stocks['Weight %'] = 100 / len(selected_stocks) # Equal weight if total score is zero

        selected_stocks['Investment Amount (₹)'] = (selected_stocks['Weight %'] / 100) * investment_amount
        
        selected_stocks = selected_stocks.round({
            'Sharpe Ratio': 2,
            'Beta': 2,
            'Volatility': 2,
            'Weight %': 2,
            'Investment Amount (₹)': 0
        })
        # Drop the temporary 'Allocation Score' column
        selected_stocks = selected_stocks.drop(columns=['Allocation Score'])

    return selected_stocks

# ----------------------------
# Streamlit App UI
# ----------------------------

st.set_page_config(page_title="AI-Based Stock Recommender", layout="centered", initial_sidebar_state="expanded")
st.title("💼 AI-Based Stock Recommender for Fund Managers")
st.markdown("""
This intelligent assistant recommends stock allocations based on a client's risk profile using dynamically calculated financial metrics.
**Disclaimer:** This tool is for informational purposes only and does not constitute financial advice. Investment in securities markets are subject to market risks, read all the related documents carefully before investing.
""")

st.header("📋 Enter Client Profile")

with st.expander("Client Demographics & Financials", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Client Age", 18, 75, 35, help="Client's current age. Younger clients generally have a higher risk tolerance.")
        income = st.number_input("Monthly Income (₹)", min_value=0, value=75000, step=5000, help="Client's current monthly income.")
        dependents = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4, 5], help="More dependents typically mean lower risk capacity.")
    with col2:
        qualification = st.selectbox("Highest Qualification", ["Graduate", "Postgraduate", "Professional", "Other"], help="Higher qualification may suggest better financial literacy.")
        duration = st.slider("Investment Duration (Years)", 1, 30, 5, help="Longer investment horizons generally allow for more risk.")
        investment_type = st.radio("Investment Type", ["Lumpsum", "SIP"], horizontal=True, help="SIP can smooth out market volatility compared to a lumpsum investment.")

with st.expander("Investment Details & Risk Tolerance", expanded=True):
    investment_amount = st.number_input("Total Investment Amount (₹)", min_value=1000, value=100000, step=10000, help="The total amount the client intends to invest.")
    volatility_comfort = st.slider("How comfortable is the client with market volatility (potential ups and downs)?", 1, 5, 3, help="1: Very Uncomfortable, 5: Very Comfortable. A direct measure of psychological risk tolerance.")

# Recommendation button
if st.button("Generate Recommendation", type="primary"):
    with st.spinner("Generating recommendations... This may take a moment as we fetch live stock data and calculate metrics."):
        risk_profile = get_risk_profile(age, income, dependents, qualification, duration, investment_type, volatility_comfort)
        st.success(f"📊 Client Risk Profile: **{risk_profile}**")
        st.info(f"💰 Proposed Investment Allocation for ₹{investment_amount:,.0f}")

        recommended_stocks = get_stock_recommendations(risk_profile, investment_amount)

        if not recommended_stocks.empty:
            st.markdown("### 📈 Recommended Stock Portfolio")
            st.dataframe(recommended_stocks, use_container_width=True)

            # Optional: Add a pie chart for visual allocation
            st.markdown("### 📊 Portfolio Allocation Overview")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(recommended_stocks['Weight %'], labels=recommended_stocks['Stock'], autopct='%1.1f%%', startangle=90, pctdistance=0.85)
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title("Recommended Portfolio Weight Distribution")
            st.pyplot(fig)

            st.markdown("---")
            st.markdown("#### Understanding the Metrics:")
            st.markdown("""
            * **Sharpe Ratio:** Measures risk-adjusted return. Higher is better. It indicates the amount of return an investor receives for each unit of risk.
            * **Beta:** Measures a stock's volatility in relation to the overall market (benchmark). A Beta of 1 means the stock moves with the market. Beta < 1 indicates lower volatility than the market; Beta > 1 indicates higher volatility.
            * **Volatility:** Measures the fluctuation of returns (standard deviation of returns). Lower volatility generally means lower risk.
            * **Market Cap:** Indicates company size (Large, Mid, Small). Generally, larger market cap companies are considered less volatile.
            """)
        else:
            st.warning("No suitable stocks found for this risk profile with the current data. Please adjust inputs or expand the stock universe.")

# Add a simple footer
st.markdown("---")
st.markdown("Built by Your Name/Organization")
