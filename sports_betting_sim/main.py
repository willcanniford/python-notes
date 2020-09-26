from code.helpers import simulate_n_bets

# Simulate 100 bets of 4/9 staking £5 each time
n_100 = simulate_n_bets(100, 4 / 9, 5)

# Calculate total winnings
print(sum(n_100))

# Calculate number of bets won / 100
print(len([i for i in n_100 if i > 0]))
