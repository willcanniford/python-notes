# Imports
import numpy as np
import pandas as pd
from sklearn import datasets

# Load iris dataset
iris = datasets.load_iris()
data1 = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                     columns=iris['feature_names'] + ['target'])

# print(data1)
# print(iris['target_names'])

# Mean gives a strong impression but is impacted by outliers
print(data1.apply(np.mean, axis=0))

# Median isn't as heavily impacted by outliers
print(data1.apply(np.median, axis=0))

# Custom percentiles can give you more information 
percentiles = np.percentile(data1["sepal length (cm)"], [10,25,50,75,90])
print(percentiles)
