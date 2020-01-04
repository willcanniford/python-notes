# Example of creating an ordered variable in pandas
# Splice the flag columns into a new dataframe for summarising
WT = weather.loc[:, 'WT01':'WT22']

# Calculate the sum of each row in 'WT'
# By specifying the axis here we are summing across the columns - rowwise
weather['bad_conditions'] = WT.sum(axis='columns')

# Replace missing values in 'bad_conditions' with '0'
weather['bad_conditions'] = weather.bad_conditions.fillna(0).astype('int')

# Count the unique values in 'bad_conditions' and sort the index
print(weather.bad_conditions.value_counts().sort_index())

# Create a dictionary that maps integers to strings
mapping = {
    0: 'good',
    1: 'bad',
    2: 'bad',
    3: 'bad',
    4: 'bad',
    5: 'worse',
    6: 'worse',
    7: 'worse',
    8: 'worse',
    9: 'worse'}

# Convert the 'bad_conditions' integers to strings using the 'mapping'
weather['rating'] = weather.bad_conditions.map(mapping).astype(str)

# Create a list of weather ratings in logical order
cats = ['good', 'bad', 'worse']

# Change the data type of 'rating' to category
weather['rating'] = weather.rating.astype(
    'category', ordered=True, categories=cats)

# Examine the head of 'rating'
# You'll now note that Series has an order to it
print(weather['rating'].head())
