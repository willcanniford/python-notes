# First example
# Import the libraries that are going to be needed
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialise the model
model = LinearRegression()

# Define values; note that X must be a 2D array with each entry a
# representation of observations
X = [[3], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]

# Outcome variable
y = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

# Fit the model
model.fit(X, y)

# Make a prediction and print to the console
print(model.predict([[2]]))


# Second example from the documentation
# Import numpy for this example

# Generate observations and check the structure
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
print(X)

# Generate outcome variable and check structure using defined mapping function
y = np.dot(X, np.array([1, 2])) + 3
print(y)

# Initialise and fit the linear regression model
reg = LinearRegression().fit(X, y)

# View various model features
print("Model score: {}".format(reg.score(X, y)))
print("Model coefficients: {}".format(reg.coef_))
print("Model intercept: {}".format(reg.intercept_))
print("Model predictions: {}".format(reg.predict(np.array([[3, 5], [2, 3]]))))
