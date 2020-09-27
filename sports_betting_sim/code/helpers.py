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


def bank_over_time(bank, n, odds, stake):
    """
    Generate an array tracking total account bank over n bets
    :param bank: the starting amount in the account
    :param n: int: the number of bets to simulate
    :param odds: numeric: the odds given for the event
    :param stake: numeric: the amount of money being staked
    :return: np.array: bank tracking array, length of n+1
    """
    array = np.array([])
    array = np.insert(array, 0, bank)
    bank = bank

    for _ in np.arange(n):
        if bank <= 0:
            array = np.append(array, 0)
        else:
            if bank < stake:
                returns = round(simulate_bet(odds, bank), 2)
            else:
                returns = round(simulate_bet(odds, stake), 2)

            if returns == 0:
                bank = bank - stake
            else:
                bank = bank + returns

            array = np.append(array, bank)

    return array
