import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load data
data = pd.read_csv('../data/Dataset_spine.csv').iloc[:, :-1]

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1000)

print(len(X_train) == len(y_train))

print(np.unique(y_test, return_counts=True))
print(np.unique(y_train, return_counts=True))

# Train models
poly_model = SVC(kernel = 'poly').fit(X_train, y_train)
rbf_model = SVC(kernel = 'rbf', gamma = 10).fit(X_train, y_train)

# See accuracies
poly_pred = poly_model.predict(X_test)
print("Polynomial kernel accuracy: {}".format(accuracy_score(y_test, poly_pred)))

rbf_pred = rbf_model.predict(X_test)
print("RBF kernel accuracy: {}".format(accuracy_score(y_test, rbf_pred)))

gamma_values = range(1,100, 2)

gamma_set = []
for value in gamma_values:
    model = SVC(gamma = value).fit(X_train, y_train)
    gamma_set.append([value, accuracy_score(y_test, model.predict(X_test))])

print(gamma_set)
