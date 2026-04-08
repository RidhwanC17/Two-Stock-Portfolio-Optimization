import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = ['RELIANCE.NS', 'INFY.NS']

start_date = '2020-01-01'
end_date = '2026-03-31'

data = yf.download(stocks, start=start_date, end=end_date, auto_adjust=True)
prices = data['Close'].dropna()

returns = prices.pct_change().dropna()

split_date = '2024-03-31'
train_returns = returns[returns.index < split_date]
test_returns = returns[returns.index >= split_date]

mean_returns = train_returns.mean() * 252

cov_matrix = train_returns.cov() * 252

risk_free_rate = 0.05

mu = mean_returns.values
Sigma = cov_matrix.values
ones = np.ones(len(mu))

excess_mu = mu - risk_free_rate * ones

Sigma_inv = np.linalg.inv(Sigma)

w_unnormalized = Sigma_inv @ excess_mu

w = w_unnormalized / (ones @ w_unnormalized)

optimal_allocation = pd.Series(w, index=train_returns.columns)

def portfolio_return(weights):
    return np.dot(weights, mu)

def portfolio_volatility(weights):
    return np.sqrt(weights.T @ Sigma @ weights)

ret = portfolio_return(w)
vol = portfolio_volatility(w)
sharpe = (ret - risk_free_rate) / vol

portfolio_test_returns = test_returns.dot(optimal_allocation)
cumulative_return = (1 + portfolio_test_returns).prod() - 1

num_simulations = 5000
num_days = 252

simulated_final_returns = []

for _ in range(num_simulations):
    simulated_returns = np.random.multivariate_normal(
        mu / 252,
        Sigma / 252,
        num_days
    )
    
    portfolio_returns = simulated_returns @ w
    cumulative = np.prod(1 + portfolio_returns) - 1
    simulated_final_returns.append(cumulative)

simulated_final_returns = np.array(simulated_final_returns)

mc_mean = simulated_final_returns.mean()
mc_std = simulated_final_returns.std()
mc_p5 = np.percentile(simulated_final_returns, 5)
mc_p95 = np.percentile(simulated_final_returns, 95)

print("Optimal Weights:")
print(f"Reliance: {optimal_allocation['RELIANCE.NS']:.4f}")
print(f"Infosys: {optimal_allocation['INFY.NS']:.4f}")

print("\nSharpe Ratio (train):")
print(f"{sharpe:.4f}")

print("\nMonte Carlo Results:")
print(f"Expected Return: {mc_mean:.4f}")
print(f"Volatility: {mc_std:.4f}")
print(f"Worst Case (5%): {mc_p5:.4f}")
print(f"Best Case (95%): {mc_p95:.4f}")

plt.hist(simulated_final_returns, bins=50)
plt.title("Monte Carlo Distribution (1-Year Return)")
plt.xlabel("Return")
plt.ylabel("Frequency")

plt.savefig("mc_plot.png")
plt.show()

print("\nOut-of-Sample (OOS) Return:")
print(f"{cumulative_return * 100:.2f}%")