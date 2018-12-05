# Imports 
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Load our data
# sklearn comes with datasets, but I've found this one on Kaggle that is quite good and interesting
raw_data = pd.read_csv('./data/Dataset_spine.csv')

# Remove redundant column at the end that contains meta information
data = raw_data.iloc[:, :-1]

# View the data that we have
print(data.head())

# Separate the features from the response variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.head())
print(y.head())

# Scale the input
X_scaled = StandardScaler().fit_transform(X)
print(X_scaled[0])

# Make a DecisionTreeClassifier with some of the available hyperparameters:
# max_depth: The maximum number of levels in the tree.
# min_samples_leaf: The minimum number of samples allowed in a leaf.
# min_samples_split: The minimum number of samples required to split an internal node.
# max_features : The number of features to consider when looking for the best split.
tree_model = DecisionTreeClassifier(max_depth = 5, min_samples_leaf = 5, max_features = 3)

# Fit the tree model using the scaled X data that we created above
tree_model.fit(X_scaled, y)

# Predict using the scaled data
predictions = tree_model.predict(X_scaled)

# Print the model's accuracy
# Depending on the seed of the system will alter the final accuracy, but I get ~85-91% which is pretty good 
# The model is probably overfit since we haven't done a train/test split though...
print(round(accuracy_score(predictions, y), 2))