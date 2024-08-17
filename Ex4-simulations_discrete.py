import numpy as np
import matplotlib.pyplot as plt

# Define the discrete PDF
outcomes = [-0.05, 0.00, 0.05, 0.10]  # Possible return outcomes
probabilities = [0.2, 0.3, 0.4, 0.1]  # Corresponding probabilities

# Number of simulations
num_simulations = 1000

# Simulate returns
simulated_returns = np.random.choice(outcomes, size=num_simulations, p=probabilities)

# Define bin edges to center the bars on outcomes
bin_edges = np.array(outcomes) - 0.025  # Shift each edge by half the width of the bin
bin_edges = np.append(bin_edges, outcomes[-1] + 0.025)  # Add the upper edge of the last bin

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(simulated_returns, bins=bin_edges, edgecolor='k', alpha=0.7, align='mid')
plt.xticks(outcomes)  # Set x-ticks to be exactly on the outcomes
plt.title('Histogram of Simulated Returns')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
