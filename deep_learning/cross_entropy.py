import numpy as np

# This is for two classes for cross entropy
# Write a function that takes as input two lists Y, P,
# and returns the float corresponding to their cross-entropy.
# Y represents the events, and P the probabilities


def cross_entropy(Y, P):
    # Make sure that the vectors are floats
    Y = np.float_(Y)
    P = np.float_(P)
    # Note that if the event is 1 it takes the first part
    # If the event is 0 it takes the value for q given that p and adds nothing
    # for the first segment with p
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
