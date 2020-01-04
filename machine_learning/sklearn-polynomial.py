# Import the libraries and functions that we are going to need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load in some fake data for comparison
data = pd.read_csv('./data/polynomial.csv')
X = data[['Var_X']].values.reshape(-1, 1)
y = data[['Var_Y']].values

# Assign the data to predictor and outcome variables for another theoretical set
# X = np.array([1, 3, 4, 5, 2, 8, 7, 6, 5, 4, 2]).reshape(11, 1)
# y = np.array(list(map(lambda x: 2*x**3 - 3*x**2 + 5*x + 9, X))).flatten()

# Create polynomial features
# Create a PolynomialFeatures object, then fit and transform the predictor
# feature to use these polynomial features
poly_feat = PolynomialFeatures(degree=4)
X_poly = poly_feat.fit_transform(X)

# Make and fit the polynomial regression model
# Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept=False).fit(X_poly, y)

# Make predictions using the linear model with poly features
data['Predictions'] = poly_model.predict(X_poly)
# Sort by the values of the X variable to fix model line plotting
data.sort_values('Var_X', inplace=True)

# Visualise the predictions against the real values
plt.scatter(data[['Var_X']].values, data[['Var_Y']].values, c='Blue')
plt.plot(data[['Var_X']].values, data[['Predictions']], c='Red')
plt.title('4 Degree Polynomial predictions using sklearn')
plt.show()
