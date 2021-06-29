# Regression
This is used when the target variables is a continuously distributed variable, such as the price of a house. It uses independent variables to determine a value of the target variable. 

When using `sklearn` we are going to separate the data that we are predicting into feature and target arrays; we select the columns we are interested in for both the feature and target and then use the `.values` attribute to return a `numpy.array`.  

It will also be necessary to `reshape` the data so that the arrays are the defined correct shapes, you can do this using `.reshape(-1, 1)` on both the features array and the target. Performing this step is an important step in making sure the data is ready for `sklearn`. 

- - -  
## Linear Regression with `sklearn`
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data.csv')

X = df.drop(['target']).values.reshape(-1, 1)
y = df['target'].values.reshape(-1, 1)

# Initiate the regression object
reg = LinearRegression()
reg.fit(X, y) # Train using features/target
```

We can expand on the above by using a training and test split to avoid the overfitting and make sure that our model generalises well to unseen data. 

```python 
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
```

- - - 
## Basics of Linear Regression 
- We want to fit a line to the data using some model parameters, a constant and a coefficient for each feature variable 
- We are going to select these parameters by minimising a loss/cost/error function 
    - We want the line to be as close to the points as possible, and thus we want to minimise the vertical distance between each point and the line itself. These distances are called residuals. 
    - We could use the sum, but then positive and negatives cancel eachother out.
    - We, therefore, minimise the sum of the squares of the residuals: this is called Orindary Least Squares (OLS).

R squared: intuitavely this shows you the amount of variance in the target variable that can be predicted by the feature variables. 

__Note__: You're unlikely to use linear regression like this and are more likely to use regularisation that adds further constraints to the models. 

- - -

## Regularised Regression
When you are creating the model, you are finding a coefficient for each parameter or feature. If you allow these to become too large then this can lead to drastic overfitting. The more features that you have, then the higher the risk of this happening. To avoid this you can regularise: 

> You want to penalise the larger coefficients to prevent them from getting too large. Altering the loss function enables this. This is regularisation. 

### Ridge Regression 
`{OLS loss function} + sum(squared value of each coefficient) * alpha`

This effectively means that we are minimising a function now that will penalise both large negative and large positive coefficients in the regression. 

`alpha` is a parameter that we can tune to improve our model and can be tuned as such. It controls model complexity as a value of 0 means that we have a simple OLS error function again. Conversely, a high value of `alpha` has a heavily penalty for large coefficients and can lead to underfitting the data. 

```python 
from sklearn.linear_model import Ridge

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Note this normalises the scales of all variables 
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(X_test, y_test)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
```

Example from course: 

```python
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_value_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
```

### Lasso Regression 
`{OLS loss function} + sum(absolute coefficient) * constant`

On the face of it, this doesn't penalise larger coefficients as heavily as ridge regression. Lasso regression actually can be used for feature selection as it tends to shrink the coefficients of less important features to exactly 0, at which point they have no predictive power on the target variable. 

```python
from sklearn.linear_model import Lasso 

lasso = Lasso(alpha=0.3, normalize=True)
lasso.fit(X, y)
lasso_coef = lasso.coef_
print(lasso_coef)
```

Through using the above code you should be able to see which features have shrunk to 0 and which hold the most predictive power. 

### Elastic Net Regularisation
There is a another type of regularisation for regression called 'Elastic Net' where the penalty term is a combination of 'l1' (lasso) and 'l2' (ridge). 

> a * L1 + b * L2

```python 
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state = 42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))
```

