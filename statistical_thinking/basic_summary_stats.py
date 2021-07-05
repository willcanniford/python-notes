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


# Calculating the variance 'by hand' and using numpy 
# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)
diff_sq = differences ** 2

# Compute the mean square difference: this is the 'by hand' method 
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Compare the two resulting figures
print(variance_explicit, variance_np)

# Note the same can be done here with the standard deviation
# Print the square root of the variance
print(np.sqrt(variance_np))

# Print the standard deviation
print(np.std(versicolor_petal_length))

# Covariance calculations 
# Compute the covariance matrix: covariance_matrix
# Note that where covariance_matrix[i, i] this represents the variance of that variable 
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0, 1]

# Print the length/width covariance
print(petal_cov)

# Specifying ddof=0 means that variance matches np.var(xi)
np.cov(versicolor_petal_length, versicolor_petal_width, ddof=0)

