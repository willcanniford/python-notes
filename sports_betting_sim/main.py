import matplotlib.pyplot as plt
import numpy as np

from code.helpers import simulate_n_bets, bank_over_time

# Simulate 100 bets of 4/9 staking £5 each time
n_100 = np.array(simulate_n_bets(100, 4 / 9, 5))

# Calculate loss/gain
n_100_lg = n_100 - 5
n_100_lg = np.insert(n_100_lg, 0, 50)

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
x = np.arange(16)

for _ in np.arange(1500):
    plt.plot(x, bank_over_time(50, 15, 1, 5), c='grey', alpha=0.25, zorder=2)

ax.axhline(50, c='red', zorder=1, lw=1, alpha=0.7)
ax.set_xlim(0, None)
ax.set_ylim(0, None)
plt.show()
