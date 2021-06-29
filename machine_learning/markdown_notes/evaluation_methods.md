# Evaluation methods
The way in which we evaluate models depends on the type of model, and the type of response that we obtain from our predictions. The main two types of focus here are _regression_ and _classification_.  

Our goal is to determine which model we've created is the best for our needs; how can you tell whether a model is good or not?  We should make sure the model can cope with unseen data, and that we are using the right metrics to measure its performance.  

- - - 

## Making sure the model generalises well
To get an idea about the true 'real-world' value of our model, it is good practice to test the model on data that it hasn't seen before, i.e. that data was not involved in the training process; this reduces overfit of our model and, as previously illuded to, gives a better indication of a model's performance if it was to be used in a production setting. 

> We want to generalise the data sufficiently well that its predictive power extends to unseen observations.

We, therefore, don't mind sacrificing a level of accuracy on the training set to improve model performance on the testing set. 

The usual split is 80 train / 20 test. We can perform cross-validation that gives us a number of out-of-sample metrics before re-training the model again with all the data. This provides the best of both worlds by looping through splits in the data and training on _n_ training sets, while testing on the remainder and recording the performance metrics. At the end we have the average performance for each loop through the cross-validation and a final model that has been trained on all of the data available to us. 

### Train-test splits in Python
We can create train-test splits easily in Python using the `sklearn.cross_validation` package and importing `train_test_split`. 

An example of generating training and testing datasets using `train_test_split`, with 20% being retained for testing:
```python 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
``` 

We can also ensure that our stochastic processes are the same between model trainings by specifying a `random_state` argument in the call to `train_test_split`; this ensures that our results are reproducible, an important part of the process. 

- - - 
## Measuring model performance 
### Classification metrics 
Typically you will look to measure the performance using **accuracy** but this isn't always the best solution; you calculate the fraction of correctly classified examples to measure the performance of your model. 

**Accuracy** doesn't really work with class imbalance problems where we are working with a dominant class; where one class is much more frequent. 

#### Confusion Matrix:
||Predicted: Class 1| Predicted: Class 2|
|:--:|:--:|:--:|
|**Actual: Class 1**|True Positive|False Negative|
|**Actual: Class 2**|False Positive|True Negative|

We can use a combination of the above results from the confusion matrix to calculate new metrics that are more applicable to class imbalance problems. 

Accuracy: `tp + tn / tp + tn + fp + fn`  
Precision: `tp / tp + fp` (correctly labelled examples for the 'positive' class)  
Recall/Sensitivity: `tp / tp + fn` (hit rate/true positive rate whereby how many of 'positive' class were predicted successfully)  
F1 score: `2 * ((precision*recall)/precision+recall)` (the harmonic mean of precision and recall)  

High precision, not many class 2 examples where predicted as being class 1.  
High recall, predicted most class 1 examples correctly. 

```python 
from sklearn.metrics import classification_report, confusion_matrix 

confusion_matrix(y_test, y_pred)
classification_report(y_test, y_pred)
```
