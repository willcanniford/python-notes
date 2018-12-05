# Imports
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# Read in the data
data = pd.read_csv('./data/polynomial.csv')
print(data.head())

# The benefits of Lasso regression is that it is L1 regularised, meaning that it performs feature selection for you to a degree as it is capable of altering a feature's coefficient to 0 and thus removing it from the model
# This doesn't make a difference in our example, but would if we had numerous features 
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

print(X)
print(y)

# Create a Lasso regression model from sklearn
lasso_reg = Lasso()
print(lasso_reg)
print(type(lasso_reg))

# Fit the model with our data
lasso_model = lasso_reg.fit(X, y)

# We can access the coeficients through the `coef_` property
# Note that this is a linear model, and isn't suited to this polynomial example data
print(lasso_model.coef_)


# Scaling data with sklearn
from sklearn.preprocessing import StandardScaler

# It is important when we are using regularisation or algorithms that use distance metrics to scale and standardise our features
# This is to avoid the algorithm treating one feature more harshly than another based purely on the magnitude of the units it is recorded in

# Create a StandardScaler object
scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)

# As our X variable wasn't that spread out, we don't see a massive difference in the resulting values
print(scaled_X)
print(X)

# Let's try with the response variable purely for comparison
scaled_y = scaler.fit_transform(pd.DataFrame(y))

# We can see that the impact on y was much greater, and it now reflects the same scale seen in X
# You wouldn't do this for your response variable, this was purely to demonstrate the impact of scaling the data
print(scaled_y)
print(y)