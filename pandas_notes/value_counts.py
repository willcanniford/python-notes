import pandas as pd

# This will return raw counts
data.categorical_column.value_counts()

# This will return proportions of the data
data.categorical_column.value_counts(normalize=True)
