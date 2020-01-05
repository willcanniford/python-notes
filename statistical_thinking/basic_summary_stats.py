# Imports 
import numpy as np
import pandas as pd 
from sklearn import datasets

# Load iris dataset 
iris = datasets.load_iris()
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# print(data1)
# print(iris['target_names'])

# Mean gives a strong impression but is impacted by outliers 
print(data1.apply(np.mean, axis = 0))

# Median isn't as heavily impacted by outliers 
print(data1.apply(np.median, axis = 0))