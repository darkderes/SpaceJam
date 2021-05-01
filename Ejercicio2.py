import pandas as pd

player_df_final = pd.read_csv('player_data_final.csv')
#print(player_df_final)

# Create a DataFrame of only Tune Squad players.
ts_df = player_df_final.iloc[26: , :]
#ts_df

# Import Tune Squad player names.
ts_name_df = pd.read_csv('tune_squad.csv', sep='\t')
#print(ts_name_df)

ts_df = pd.merge(ts_df, ts_name_df, on='ID', how='left', suffixes=('_type', '_name'))
#print(ts_df.head())

# Rearrange the columns to put the ID and player_name columns next to each other.
column_list = list(ts_df)

player_name = column_list.pop()
column_list[1] = player_name

ts_df = ts_df[column_list]
#print(ts_df.head())

# Create a list of only the column names we are interested in.
game_stat_cols = list(ts_df.iloc[:, 7:-1])
game_stat_stdevs = []

# Create a list of standard deviations for each stat.
for stat in game_stat_cols:
    game_stat_stdevs.append(ts_df[stat].std())

# Create a Series of the standard deviations, with the stat names as the index.
stdev_s = pd.Series(game_stat_stdevs, index=game_stat_cols)
#print(stdev_s)

# Import the models and libraries we need.
from sklearn.linear_model import LinearRegression
from numpy.random import randn

# Get the dependent and independent variables for modeling the PER.
X = player_df_final.iloc[:, 7:-1].to_numpy()
y = player_df_final.iloc[:, -1]

# Define and fit the model.
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Print the player with the highest and lower PER for each iteration.
print('Iteration # \thigh PER \tlow PER')

# Run the simulation 10 times.
for i in range(10):

    # Define an empty temporary DataFrame for each iteration.
    # The columns of this DataFrame are the player stats and the index is the players' names.
    game_df = pd.DataFrame(columns=game_stat_cols, index=list(ts_df['player_name']))
    
    # Loop through each stat.
    for stat in game_stat_cols:
        
        # Each player's stats are used to generate a random value for each iteration.
        game_df[stat] = list(ts_df[stat] + randn(len(ts_df)) * stdev_s[stat])
    
    # Use the fitted model to predict players' PERs based on the randomized data.
    game_df['PER'] = lin_reg.predict(game_df)

    # Print the player with the highest and lower PER for each iteration.
    print('Iteration {}'.format(i+1) + ' \t' + game_df['PER'].idxmax() + ' \t' + game_df['PER'].idxmin())

    # Initialize four empty DataFrames, one for each 12-minute period.
number_of_iterations = 4
df_list = [pd.DataFrame(columns=game_stat_cols, index=list(ts_df['player_name'])) for i in range(number_of_iterations)]

# For each period, generate randomized player data and predict the PER.
# Use the model fitted earlier.
for df in df_list:
    for stat in game_stat_cols:
        df[stat] = list(ts_df[stat] + randn(len(ts_df)) * stdev_s[stat])
    df['PER'] = lin_reg.predict(df)

# Concatenate the DataFrames and make the players' names the index.
game_df = pd.concat(df_list)
game_df.rename_axis('player_name', inplace=True)

# Create another index for the period in question.
minutes = [(x // len(ts_df)) * 12 for x in range(len(game_df))]
game_df['minutes'] = minutes
game_df.set_index('minutes', append=True, inplace=True)
game_df = game_df.swaplevel()

print(game_df)

# Export the finished DataFrame to CSV.
game_df.to_csv('game_stats.csv')