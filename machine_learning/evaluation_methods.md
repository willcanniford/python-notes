# Evaluation methods
The way in which we evaluate models depends on the type of model, and the type of response that we obtain from our predictions. The main two types of focus here are _regression_ and _classification_.  

Ultimately our goal is to determine which of the models that we have created is the best for our needs. To get an idea about the _true 'real-world'_ value of our model, it is good practice to test the model on data that it hasn't seen before, i.e. that data was not involved in the training process; this reduces overfit of our model and, as previously illuded to, gives a better indication of a model's performance if it was to be used in a production setting. 

> We want to generalise the data sufficiently well that its predictive power extends to unseen observations.

We, therefore, don't mind sacrificing a level of accuracy on the training set to improve model performance on the testing set. 

The usual split is 80 train / 20 test. We can perform cross-validation that gives us a number of out-of-sample metrics before re-training the model again with all the data. This provides the best of both worlds by looping through splits in the data and training on _n_ training sets, while testing on the remainder and recording the performance metrics. At the end we have the average performance for each loop through the cross-validation and a final model that has been trained on all of the data available to us. 

## Train-test splits in Python
We can create train-test splits easily in Python using the `sklearn.cross_validation` package and importing `train_test_split`. 

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)` would be an example of how we can generate train and test sets using this function for a 20/80 split for training to test sets. 

We can also ensure that our stochastic processes are the same between model trainings by specifying a `random_state` argument in the call to `train_test_split`. 
