# Preprocessing with `sklearn`
## Categorical features 
Categorical features will need to be preprocessed to be made numerical ahead of the model training process. You can do this in a few ways, but the creation of dummy variables is a good place to start. 

### Dummy variables 
These dummy variables are a way of encoding categorical data. We create a number of binary variables, with each one representing a class of the categorical variable that we are encoding i.e. colour: ['red','blue'] column might become colour_red, colour_blue with boolean values depending on the value for that observation. 

We don't want to duplicate information, so we can employ an n-1 (n being the number of distinct categories) process for the columns i.e. we only need colour_red because if that value is 0 then we know it is blue, the only other option. 

|Library|Encoding Process|
|-|-|
|`sklearn`|`OneHotEncoder()`|
|`pandas`|`get_dummies()`|

You can use `drop_first` in `get_dummies()` to remove the first category from the split out encoded columns. 

You can view categorical variables against the target easily using a `pd.DataFrame` with the `.boxplot()` method. 

```python 
# Import pandas
import pandas as pd 

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=25)

# Show the plot
plt.show()
```

## Handling Missing Data
### Dropping missing data 
This isn't functional when you end up decreasing the size of shape of the data. 

### Imputing data and building pipelines
This ranges from computing the inputting the mean column-wise. 

There are a number of objects within `sklearn` that make imputation and model building within a pipeline possible: `sklearn.Pipeline`. Each step within the pipeline but the last must be a transformer, and then the last must be an estimator (model in practice).  

```python 
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
```

## Centering and scaling
### Scaling your data 
Many models use distance to inform the model. You want all your variables from the input to be on a similar scale, otherwise ones with larger scales will have a greater impact on the model; take k-NN for example, that uses distance explicitly when making the classifications. 

### Normalising 
Standardising: subtract the mean and divide by the variance, creating a set centered around 0 with a variance of 1. 

You can substract the min and divide by the range, meaning the data ranges from 0 to 1. 

```python 
# Import scale
from sklearn.preprocessing import scale 

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
```

The impact of scaling can be seen through running the data through a pipeline containing a `StandardScaler` and without one: 

```python 
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))
```


```python 
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))
```

Another example: 
Note the naming convention of paramters when using a grid search over a pipeline is `step_name__parameter_name`. 

```python 
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
```
