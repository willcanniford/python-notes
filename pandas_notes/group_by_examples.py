# You can group by given column values prior to summarising the value of another column
# This is similar to R's group_by(x,y) %>% summarise(avg=mean(search_conducted))
# Calculating the search rate for each combination of gender and violation
print(ri.groupby((ri.driver_gender, ri.violation)).search_conducted.mean())

# If we have an index that is a date or datetime then we can group by the index.hour
# Calculate the hourly arrest rate
print(ri.groupby(ri.index.hour).is_arrested.mean())

# Or we can use resample('A') for annual rates
annual = ri.search_conducted.resample('A').mean()