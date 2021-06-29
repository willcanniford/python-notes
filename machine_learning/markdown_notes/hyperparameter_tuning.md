# Hyperparameter tuning
Hyperparameters are the values for certain parameters for the algorithm that you have to specify when the model is created, and before it is trained. These are parameters that cannot be learned when fitting the model. 

|Algorithm|Hyperparameters|
|-|-|
|Ridge/Lasso regression|Alpha|
|k-Nearest Neighbours|n_neighbours|
|Support Vector Machines|sigma and C|
|Neural Networks|Learning rate|
<br>

## Tuning the hyperparameters
You should go through potential values for the hyperparameters and then build a model from each one, comparing the output performance metrics and then select the highest performing ones to proceed with.  

It is essential that you use cross-validation for this stage, as if you just use a train/test split then you run the risk of selecting a hyperparameter based on an overfit to the training data set. Using cross-validation will give you a much more generalised set of values for your hyperparameters. You can do this using **grid search cross-validation**: `from sklearn.model_selection import GridSearchCV`

Something else to consider is to have an additional test set that can be used once the hyperparameters have been set i.e.  
- Train A/Test A  
- Train A -> cross-validation splits -> within each subset, a train and test are taken and metrics reported 
  - **nb: all data is contained within Train A during this phase**
- Hyperparameters are chosen and a model is fit using all of Train A. 
- Test performance on Test A.  


## Logistic Regression 
Logistic regression does have a regularisation parameter that can be controlled as a hyperparameter: C. C controls the inverse of the regularisation strength, a larger value can lead to overfitting, and too small can lead to underfitting. 

```python 
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))
```

We can tie the concepts of training and test sets outlined above, and then tie that in with the grid search tuning to produce an example as per below: 

```python 
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))
```

## `RandomizedSearchCV`
GridSearchCV can be computationally expensive, especially with multiple parameters and a large parameter space; `RandomizedSearchCV` samples a fixed number of hyperparameter settings from specified distributions. 

```python 
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
```

_Note: `RandomizedSearchCV` will never outperform `GridSearchCV` but it is valuable because it saves computation._