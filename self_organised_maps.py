import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
import os

warnings.filterwarnings('ignore')



# # Simulated dataset with enough data and variation
# np.random.seed(42)

# # Generate data for three teams and three seasons
# teams = ['Team1', 'Team2', 'Team3']
# seasons = ['2021', '2022', '2023']
# races_per_season = 5

# data = []
# for team in teams:
#     for season in seasons:
#         for race in range(1, races_per_season + 1):
#             lap_times = np.random.uniform(80, 100, size=50)  # Simulated lap times
#             sector1 = lap_times * np.random.uniform(0.3, 0.4, size=50)
#             sector2 = lap_times * np.random.uniform(0.3, 0.4, size=50)
#             sector3 = lap_times - (sector1 + sector2)
#             avg_speed = np.random.uniform(180, 220, size=50)

#             for i in range(50):  # 50 laps per race
#                 data.append([team, season, race, lap_times[i], sector1[i], sector2[i], sector3[i], avg_speed[i]])

# # Create DataFrame
# columns = ['Team', 'Season', 'Race', 'LapTime', 'Sector1', 'Sector2', 'Sector3', 'AverageSpeed']
# df = pd.DataFrame(data, columns=columns)

# # Function to preprocess data for Apriori
# def preprocess_winner_laps(winner_laps):
#     """
#     Preprocess winner laps for Apriori by converting numerical features into categorical bins.
#     """
#     discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  # Binning into 3 categories
#     binned_data = discretizer.fit_transform(winner_laps[['LapTime', 'Sector1', 'Sector2', 'Sector3', 'AverageSpeed']])

#     # Create meaningful labels for bins (e.g., Low, Medium, High)
#     binned_df = pd.DataFrame(binned_data, columns=['LapTime', 'Sector1', 'Sector2', 'Sector3', 'AverageSpeed'])
#     for column in binned_df.columns:
#         binned_df[column] = binned_df[column].replace({0: f'{column}_Low', 1: f'{column}_Medium', 2: f'{column}_High'})

#     # Convert to transactions format (binary encoding for Apriori)
#     transactions = pd.get_dummies(binned_df)
    
#     print('----------------------------')
#     print("Transactions")
#     print(transactions)
#     return transactions

# # Function to apply Apriori
# def apply_apriori(transactions, min_support=0.2, min_confidence=0.5, min_lift=1.0):
#     """
#     Apply Apriori algorithm to find frequent itemsets and generate association rules.
#     """
#     # Compute frequent itemsets
#     frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)

#     # Ensure support column exists for association_rules
#     if 'support' not in frequent_itemsets.columns:
#         frequent_itemsets['support'] = frequent_itemsets['itemsets'].apply(
#             lambda itemset: transactions.loc[:, itemset].all(axis=1).mean()
#         )

#     # Generate association rules
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets = len(transactions))
#     rules = rules[rules['lift'] >= min_lift]
#     return frequent_itemsets, rules

# # SOM Training and Apriori Analysis
# for team in df['Team'].unique():
#     for season in df['Season'].unique():
#         # Filter data for team and season
#         team_season_data = df[(df['Team'] == team) & (df['Season'] == season)]
#         if team_season_data.empty:
#             continue

#         # Extract features
#         features = team_season_data[['LapTime', 'Sector1', 'Sector2', 'Sector3', 'AverageSpeed']].values

#         # Normalize features
#         scaler = StandardScaler()
#         features_scaled = scaler.fit_transform(features)

#         # Initialize SOM
#         som_size = int(np.sqrt(features_scaled.shape[0]))
#         som = MiniSom(som_size, som_size, features_scaled.shape[1], sigma=1.0, learning_rate=0.5)
#         som.random_weights_init(features_scaled)

#         # Train SOM
#         som.train_random(features_scaled, num_iteration=200)
#         print(f"Trained SOM for Team: {team}, Season: {season}")

#         # Visualize SOM Distance Map
#         plt.figure(figsize=(8, 8))
#         plt.title(f"SOM for {team} - {season}")
#         plt.pcolor(som.distance_map().T, cmap='coolwarm')
#         plt.colorbar(label='Distance')
#         plt.show()

#         # Use SOM to find "winner laps" and preprocess for Apriori
#         winner_laps = team_season_data  # Use all data for simplicity (adjust with your criteria)
#         transactions = preprocess_winner_laps(winner_laps)

#         # Apply Apriori and print results
#         frequent_itemsets, rules = apply_apriori(transactions)
#         print(f"Frequent Itemsets for {team} - {season}:\n", frequent_itemsets)
#         print(f"Association Rules for {team} - {season}:\n", rules)



# Function to preprocess data for Apriori (this assumes the features have been binned or one-hot encoded)
def preprocess_winner_laps(winner_laps):
    """
    Preprocess winner laps for Apriori by converting numerical features into categorical bins.
    """
    # Selecting relevant columns (numeric ones for analysis)
    relevant_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'AvgSectorTime', 
                        'PositionGained', 'PositionLost', 'PositionUnchanged', 'HighFuelLoad', 'MediumFuelLoad', 
                        'LowFuelLoad', 'Soft', 'Medium', 'Hard', 'Intermediate', 'Wet']
    
    # We focus only on these features
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')  # Binning into 3 categories
    binned_data = discretizer.fit_transform(winner_laps[relevant_columns])

    # Create meaningful labels for bins (e.g., Low, Medium, High)
    binned_df = pd.DataFrame(binned_data, columns=relevant_columns)
    for column in binned_df.columns:
        binned_df[column] = binned_df[column].replace({0: f'{column}_Low', 1: f'{column}_Medium', 2: f'{column}_High'})

    # Convert to transactions format (binary encoding for Apriori)
    transactions = pd.get_dummies(binned_df)
    return transactions

# Function to apply Apriori
def apply_apriori(transactions, min_support=0.2, min_confidence=0.5, min_lift=1.0):
    """
    Apply Apriori algorithm to find frequent itemsets and generate association rules.
    """
    # Compute frequent itemsets
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)

    # Ensure support column exists for association_rules
    if 'support' not in frequent_itemsets.columns:
        frequent_itemsets['support'] = frequent_itemsets['itemsets'].apply(
            lambda itemset: transactions.loc[:, itemset].all(axis=1).mean()
        )

    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(transactions))
    rules = rules[rules['lift'] >= min_lift]
    return frequent_itemsets, rules

# SOM Training and Apriori Analysis
teams = ['Ferrari', 'Mercedes', 'Redbull']
years = ['2018', '2019', '2021', '2022', '2023']

for team in teams:
    # Loop through each year for the team
    for year in years:
        # Path to the combined encoded CSV file for the team and year
        file_path = f"TransformedData/{team}/DriverLapTimes/{year}_combined_encoded.csv"
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping...")
            continue
        
        # Load the dataset for the specific year and team
        df = pd.read_csv(file_path)
        print(f"Processing file for {team} - {year}")

        # For SOM, select features to be used (we'll use continuous numerical columns)
        features = df[['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'AvgSectorTime', 
                       'PositionGained', 'PositionLost', 'PositionUnchanged', 'HighFuelLoad', 'MediumFuelLoad', 
                       'LowFuelLoad', 'Soft', 'Medium', 'Hard', 'Intermediate', 'Wet']].values

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Initialize SOM
        som_size = int(np.sqrt(features_scaled.shape[0]))  # Set size of the SOM grid based on data size
        som = MiniSom(som_size, som_size, features_scaled.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(features_scaled)

        # Train SOM
        som.train_random(features_scaled, num_iteration=200)
        print(f"Trained SOM for {team} - {year}")

        # Visualize SOM Distance Map
        plt.figure(figsize=(8, 8))
        plt.title(f"SOM for {team} - {year}")
        plt.pcolor(som.distance_map().T, cmap='coolwarm')
        plt.colorbar(label='Distance')
        
        # Save the figure as a PNG file
        output_filename = f"SOM_{team}_{year}.png"  # Use team and year in the filename
        plt.savefig(output_filename)
        print(f"Saved SOM Distance Map as {output_filename}")

        # Preprocess data for Apriori
        transactions = preprocess_winner_laps(df)

        # Apply Apriori and print results
        frequent_itemsets, rules = apply_apriori(transactions)
        print(f"Frequent Itemsets for {team} - {year}:\n", frequent_itemsets)
        print(f"Association Rules for {team} - {year}:\n", rules)
