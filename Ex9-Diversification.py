import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
N = 50  # Number of stocks
months = 60
mean_beta = 1.0
rf = 0.02  # Risk-free rate

# Simulate the excess returns on the tangent portfolio, r_T - r_f:
mu_T = 0.08 / 12
sigma_T = 0.20 / np.sqrt(12)
factor_returns = np.random.normal(mu_T, sigma_T, months)

# Simulate betas and error terms
betas = np.random.uniform(mean_beta - 0.75, mean_beta + 0.75, N)
z_std_devs = np.random.uniform(0.05, 0.15, N)  #

# Simulate stock returns
excess_returns = np.zeros((months, N))
for i in range(N):
    z_i = np.random.normal(0, z_std_devs[i], months)
    excess_returns[:, i] = betas[i] * factor_returns + z_i

# Create equally weighted portfolio
portfolio_returns = np.mean(excess_returns, axis=1)

# Plot time-series of individual stock returns and portfolio return
plt.figure(figsize=(14, 7))
for i in range(N):
    plt.plot(excess_returns[:, i], label=f'Stock {i+1}', alpha=0.5)
plt.plot(portfolio_returns, label='Equally Weighted Portfolio', color='black', linewidth=2)
plt.title('Time-Series of Simulated Stock Returns and Portfolio Return')
plt.xlabel('Months')
plt.ylabel('Excess Return')
plt.legend()
plt.show()

# Calculate the variance of the portfolio return
portfolio_variance = np.var(portfolio_returns)
print(f'Variance of the Portfolio Return: {portfolio_variance:.6f}')
check = np.mean(betas) * sigma_F**2
print(f'Average Beta Times Varaiance of Tangent Portfolio : {portfolio_variance:.6f}')

