import numpy as np


def odds_to_prob(odds):
    """
    Convert the odds given to the expected probability of the event
    :param odds: numeric: the odds given for the event
    :return: decimal: the probability of the event occurring
    """
    return 1 / (1 + odds)


def simulate_bet(odds, stake):
    """
    Simulate the bet taking place assuming the odds accurately represent the probability of the event
    :param odds: numeric: the odds given for the event
    :param stake: numeric: the amount of money being staked
    :return: decimal: the returns from the bet
    """
    probability = odds_to_prob(odds)

    if np.random.rand() <= probability:
        return stake * (1 + odds)
    else:
        return 0


def simulate_n_bets(n, odds, stake):
    """
    Simulate multiple bets with the same odds/stake parameters
    :param n: int: the number of bets to simulate
    :param odds: numeric: the odds given for the event
    :param stake: numeric: the amount of money being staked
    :return: list: the returns for n simulated bets
    """
    returns = []

    for i in np.arange(n):
        returns.append(simulate_bet(odds, stake))

    return returns
