# the softmax function is equivalent to the sigmoid but is capable of taking more than two classes
# it is an important step in having a continuous activation function inside a perceptron

import numpy as np

def softmax(L):
    # we take the exponentials to make sure that all the linear scores are positive
    expL = np.exp(L)
    sumExpL = sum(expL)
    results = []
    # Loop through and take the proportions against the sum to give the probabilities of the given classes
    for i in expL:
        # Make sure that we are multiplying by a float to get the right output type
        results.append(i*1.0/sumExpL)
    return results
