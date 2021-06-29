# Cross-validation
Model performance with a singular split is dependent on the split itself. Any metric that you calculate to see how your model is performing is at the mercy of the split that is performed randomly. This might mean that a bad split will result in metrics that don't properly reflect the model's ability to generalise to unseen data. 

It would be much better to perform multiple splits and then see how the model performs when the training and test sets differ; this should help identify the performance of the method regardless of the split that is taken during training. 

Cross-validation splits the data into `n` sets and then takes the first set as the test set, training on the `n-1` other sets. Then it takes the second as the test set, and so on. 

## `sklearn` implementation
```python
from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression

# Create a regression instance 
reg = LinearRegression()
cv_results = cross_val_score(reg, X, y, cv=5)

# Print results, an array of length = folds
print(cv_results)
```