import pandas as pd 

# Concatenate 'stop_date' and 'stop_time' (separated by a space)
# Note that in order to concatentate we must make sure that the 'stop_date' column is str 
combined = ri.stop_date.str.cat(ri.stop_time, ' ')

# Convert 'combined' to datetime format
ri['stop_datetime'] = pd.to_datetime(combined)

# Examine the data types of the DataFrame and check that 'stop_datetime' is a datetime64[ns]
print(ri.dtypes)