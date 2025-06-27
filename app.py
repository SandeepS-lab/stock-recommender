import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------
# Backtesting Function
# ----------------------------

def backtest_portfolio(stocks, weights, start='2019-01-01', benchmark='^NSEI'):
    tickers = stocks.tolist() + [benchmark]
    data = yf.download(tickers, start=start)['Adj Close'].dropna()
    
    # Normalize
    normalized = data / data.iloc[0]
    
    # Portfolio Return
    portfolio = (normalized[stocks] * weights).sum(axis=1)
    
    # Cumulative returns
    portfolio_returns = portfolio.pct_change().dropna()
    benchmark_returns = normalized[benchmark].pct_change().dropna()

    # Metrics
    def calculate_metrics(returns):
        cagr = (portfolio.iloc[-1] / portfolio.iloc[0]) ** (1 / ((len(returns)) / 252)) - 1
        volatility = np.std(returns) * np.sqrt(252)
        sharpe = (np.mean(returns) - 0.04/252) / np.std(returns) * np.sqrt(252)
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio / running_max) - 1
        max_dd = drawdown.min()
        return {
            "CAGR": round(cagr * 100, 2),
            "Volatility": round(volatility * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Max Drawdown": round(max_dd * 100, 2)
        }

    metrics = calculate_metrics(portfolio_returns)
    benchmark_metrics = calculate_metrics(benchmark_returns)

    return portfolio, normalized[benchmark], metrics, benchmark_metrics

# ----------------------------
# Commentary Generator
# ----------------------------

def generate_ai_commentary(risk_profile, selected_stocks, duration):
    sectors = {
        "TCS": "Technology", "Infosys": "Technology", "HDFC Bank": "Financials",
        "Adani Enterprises": "Conglomerate", "Zomato": "Consumer", "Reliance Industries": "Energy",
        "Bajaj Finance": "Financials", "IRCTC": "Travel"
    }
    dominant_sector = selected_stocks['Stock'].map(sectors).mode().values[0]
    
    risk_summary = {
        "Conservative": "risk-averse with a focus on capital protection.",
        "Moderate": "balanced with growth opportunities and reasonable stability.",
        "Aggressive": "growth-focused with high return potential and associated risks."
    }

    return (
        f"Your portfolio is {risk_summary[risk_profile]} "
        f"It has a strong tilt towards the **{dominant_sector}** sector, "
        f"which is expected to perform well over a {duration}-year horizon. "
        f"This portfolio is designed to align with your client's financial goals and risk appetite."
    )

# ----------------------------
# Plotting Function
# ----------------------------

def plot_cumulative_returns(portfolio, benchmark):
    fig, ax = plt.subplots()
    portfolio.plot(label="Portfolio", ax=ax)
    benchmark.plot(label="Nifty 50", ax=ax)
    ax.set_title("Cumulative Returns: Portfolio vs Nifty 50")
    ax.set_ylabel("Growth (Indexed)")
    ax.legend()
    return fig
