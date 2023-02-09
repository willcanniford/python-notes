import pandas as pd
import numpy as np

# Constants for the Elo rating system
K = 32 # The K-factor, which controls the sensitivity of the ratings to the results
INITIAL_RATING = 1500 # The initial rating for each team

# Create a sample dataset of results
results = pd.DataFrame({
    'team_a': ['A', 'B', 'C', 'A', 'C'],
    'team_b': ['B', 'C', 'A', 'C', 'B'],
    'result': [1, 0, 1, 0, 1]
})

# Create a mapping from team name to integer
team_map = {team: i for i, team in enumerate(set(results['team_a'].unique().tolist() + results['team_b'].unique().tolist()))}

# Convert team names to integers
results['team_a'] = results['team_a'].map(team_map)
results['team_b'] = results['team_b'].map(team_map)

# Initialize the ratings for each team
ratings = {i: INITIAL_RATING for i in range(len(team_map))}

# Iterate through each game in the results dataframe
for i, row in results.iterrows():
    team_a = row['team_a']
    team_b = row['team_b']
    result = row['result']

    # Calculate the expected outcome for each team
    team_a_expected = 1 / (1 + 10**((ratings[team_b] - ratings[team_a]) / 400))
    team_b_expected = 1 / (1 + 10**((ratings[team_a] - ratings[team_b]) / 400))

    # Update the ratings for each team based on the result
    ratings[team_a] = ratings[team_a] + K * (result - team_a_expected)
    ratings[team_b] = ratings[team_b] + K * ((1 - result) - team_b_expected)

# Convert the ratings to a pandas dataframe
ratings_df = pd.DataFrame.from_dict(ratings, orient='index', columns=['rating'])

print(results)
print(ratings_df)
