# Maximum Likelihood Estimation (MLE)
import numpy as np
from scipy.optimize import minimize

def logistic(x):
    return 1 / (1 + np.exp(-x))

def likelihood(params, data):
    mu = params[:-1]
    sigma = params[-1]

    p = np.zeros(len(data))
    for i, (player_a, player_b, result) in enumerate(data):
        diff = mu[player_a] - mu[player_b]
        p[i] = logistic(diff) if result == 1 else logistic(-diff)

    likelihood = np.prod(p)
    return -np.log(likelihood)

def optimize(data, num_players):
    init_params = np.zeros(num_players + 1)
    bounds = [(None, None) for i in range(num_players)] + [(0, None)]
    result = minimize(likelihood, init_params, args=(data,), bounds=bounds, method='L-BFGS-B')
    return result.x[:-1]

data = [(0, 1, 1), (0, 2, 1), (1, 2, 1)]
num_players = 3

ratings = optimize(data, num_players)
print(ratings)
