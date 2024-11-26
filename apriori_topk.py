import numpy as np
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# # Define the teams and years
# teams = ['Ferrari', 'Mercedes', 'Redbull']
# years = [2018, 2019, 2021, 2022, 2023]

# # Define the path to the folder where the files are stored
# folder_path = 'TransformedData'

# # Loop through each team and year, and print the column names of each file
# for team in teams:
#     for year in years:
#         # Construct the file path
#         file_path = os.path.join(folder_path, team, f'AssociationRules_{year}.csv')
#         print(file_path)
        
#         # Check if the file exists
#         if os.path.exists(file_path):
#             # Read the CSV file into a DataFrame
#             df = pd.read_csv(file_path)
            
#             # Print the file name and the column names
#             print(f"Columns in {team} {year} file:")
#             print(df.columns.tolist())  # Print columns as a list
#             print('-' * 50)
#         else:
#             print(f"File for {team} {year} not found.")


# Define the teams and years to focus on
teams_years = {
    'Ferrari': 2021,
    'Mercedes': 2019,
    'Redbull': 2018
}

# Define the path to the folder where the files are stored
folder_path = 'TransformedData'

# Folder to save the association rules
output_folder = 'association_rules'

# Create the output folder if it does not exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Function to filter and rank association rules
def filter_and_rank_association_rules(df):
    # Filter based on conditions such as high support, confidence, lift, and leverage
    filtered_df = df[
        (df['support'] > 0.01) &  # Support greater than 2%
        (df['confidence'] > 0.3) &  # Confidence greater than 30%
        (df['lift'] > 1.0) &  # Lift greater than 1.0
        (df['leverage'] > 0.0005)  # Leverage greater than 0.05%
    ]
    
    # Sort by Lift (descending) and Confidence (descending)
    ranked_df = filtered_df.sort_values(by=['lift', 'confidence'], ascending=False)
    
    # Return top 10 rows
    return ranked_df.head(100)

# Loop through the selected teams and years, process the CSV file and extract top 10 relevant rows
for team, year in teams_years.items():
    # Construct the file path (remove leading slash before file name)
    file_path = os.path.join(folder_path, team, f'AssociationRules_{year}.csv')
    
    # Check if the file exists
    if os.path.exists(file_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Filter and rank the DataFrame based on chosen metrics
        top_10_df = filter_and_rank_association_rules(df)
        
        # If the top 10 DataFrame is not empty, save to a new CSV file
        if not top_10_df.empty:
            # Construct the output file name
            output_file = os.path.join(output_folder, f'{team}_{year}_top_100_rules.csv')
            
            # Save the top 10 rows to the CSV file
            top_10_df[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage']].to_csv(output_file, index=False)
            print(f"Saved top 10 rules for {team} {year} to {output_file}")
        else:
            print(f"No relevant rules found for {team} {year} based on the criteria.")
    else:
        print(f"File for {team} {year} not found.")
