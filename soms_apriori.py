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

# # Redirect print statements to a file
# output_file = "output_log.txt"
# sys.stdout = open(output_file, 'w')
# print("Starting")

# New evaluation and visualization function
def evaluate_and_visualize_som_clusters(som, data, features_scaled, team, year):
    """
    Evaluate the quality of SOM clusters and visualize the results, with explanations of metrics and visualizations.
    Saves the visualizations as image files.
    """

    # Step 1: Get Best Matching Units (BMUs) and Assign Cluster Labels
    bmus = np.array([som.winner(x) for x in features_scaled])
    unique_bmus = {bmu: i for i, bmu in enumerate(set(tuple(bmu) for bmu in bmus))}
    cluster_labels = np.array([unique_bmus[tuple(bmu)] for bmu in bmus])

    # Step 2: Calculate Metrics
    metrics = {}
    if len(set(cluster_labels)) > 1:
        metrics['Silhouette Score'] = silhouette_score(features_scaled, cluster_labels)
        metrics['Calinski-Harabasz Index'] = calinski_harabasz_score(features_scaled, cluster_labels)
        metrics['Davies-Bouldin Index'] = davies_bouldin_score(features_scaled, cluster_labels)
    else:
        metrics['Silhouette Score'] = None
        metrics['Calinski-Harabasz Index'] = None
        metrics['Davies-Bouldin Index'] = None

    cluster_sizes = np.unique(cluster_labels, return_counts=True)[1]
    metrics['Cluster Balance (std)'] = np.std(cluster_sizes)
    metrics['Number of Clusters'] = len(set(cluster_labels))

    # Print Metrics with Explanations
    print("\n--- Clustering Evaluation Metrics for {} - {} ---".format(team, year))
    if metrics['Silhouette Score'] is not None:
        print("Silhouette Score: {} (Close to 1 is ideal; indicates good separation of clusters).".format(metrics['Silhouette Score']))
    else:
        print("Silhouette Score: Not applicable (Only one cluster detected).")
    print("Calinski-Harabasz Index: {} (Higher is better; indicates compact and well-separated clusters).".format(metrics['Calinski-Harabasz Index']))
    print("Davies-Bouldin Index: {} (Lower is better; indicates less overlap among clusters).".format(metrics['Davies-Bouldin Index']))
    print("Cluster Balance (std): {} (Lower is better; indicates more balanced cluster sizes).".format(metrics['Cluster Balance (std)']))
    print("Number of Clusters: {} (Should be a reasonable number based on data complexity).".format(metrics['Number of Clusters']))
    
    # Identify Optimal Laps (Winner Laps)
    cluster_avg_lap_time = {}
    for cluster in set(cluster_labels):
        avg_lap_time = data[cluster_labels == cluster]['LapTime'].mean()
        cluster_avg_lap_time[cluster] = avg_lap_time
    
    optimal_cluster = min(cluster_avg_lap_time, key=cluster_avg_lap_time.get)
    winner_laps = data[cluster_labels == optimal_cluster]
    
    print("\nOptimal Cluster: {} (Avg Lap Time: {})".format(optimal_cluster, cluster_avg_lap_time[optimal_cluster]))
    print("Winner Laps (Top Performances) for {} - {}:\n".format(team, year), winner_laps[['LapNumber','Sector1Time','Sector2Time','Sector3Time','TyreLife','FreshTyre','AvgSectorTime','PositionGained','PositionLost','PositionUnchanged','Soft','Medium','Hard','Intermediate','Wet','HighFuelLoad','MediumFuelLoad','LowFuelLoad']])
    
    win_df = winner_laps[['LapNumber','Sector1Time','Sector2Time','Sector3Time','TyreLife','FreshTyre','AvgSectorTime','PositionGained','PositionLost','PositionUnchanged','Soft','Medium','Hard','Intermediate','Wet','HighFuelLoad','MediumFuelLoad','LowFuelLoad']]
    win_df.to_csv(f"TransformedData/{team}/DriverLapTimes/{year}_optimal_laps.csv")

    # Create folder to save plots
    save_dir = "plots/{}/{}".format(team, year)
    os.makedirs(save_dir, exist_ok=True)

    # Step 3: Visualizations with Explanations
    # U-Matrix with Cluster Assignments
    plt.figure(figsize=(10, 8))
    plt.title("U-Matrix with Cluster Assignments for {} - {}".format(team, year))
    plt.pcolor(som.distance_map().T, cmap='coolwarm', edgecolors='k', shading='auto')
    for idx, (bmu, label) in enumerate(zip(bmus, cluster_labels)):
        plt.text(bmu[0] + 0.5, bmu[1] + 0.5, str(label), color='black', fontsize=8, ha='center', va='center')
    plt.colorbar(label='Distance')
    plt.savefig("{}/U-Matrix_{}_{}.png".format(save_dir, team, year), dpi=300, bbox_inches='tight')
    print("Saved U-Matrix Visualization for {} - {}: {}/U-Matrix_{}_{}.png".format(team, year, save_dir, team, year))
    plt.close()

    # Bar Plot of Data Points per Cluster
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(unique_bmus.values()), y=cluster_sizes, palette='viridis')
    plt.title("Data Points per Cluster for {} - {}".format(team, year))
    plt.xlabel("Cluster")
    plt.ylabel("Number of Data Points")
    plt.savefig("{}/ClusterSizes_{}_{}.png".format(save_dir, team, year), dpi=300, bbox_inches='tight')
    print("Saved Cluster Size Visualization for {} - {}: {}/ClusterSizes_{}_{}.png".format(team, year, save_dir, team, year))
    plt.close()

    # Node Activation Map (Data Density)
    plt.figure(figsize=(10, 8))
    plt.title("Node Activation (Data Density) for {} - {}".format(team, year))
    grid_shape = som.get_weights().shape[:2]  # Get grid size (x, y)
    som_grid = np.zeros(grid_shape, dtype=int)  # Create a zero matrix with the grid dimensions
    for bmu in bmus:
        som_grid[bmu] += 1
    plt.pcolor(som_grid.T, cmap='Blues', edgecolors='k', shading='auto')
    plt.colorbar(label='Number of Data Points')
    plt.savefig("{}/NodeActivation_{}_{}.png".format(save_dir, team, year), dpi=300, bbox_inches='tight')
    print("Saved Node Activation Map for {} - {}: {}/NodeActivation_{}_{}.png".format(team, year, save_dir, team, year))
    plt.close()

    return metrics, win_df

# Preprocess data for Apriori
def preprocess_winner_laps(winner_laps):
    # Relevant columns to process
    relevant_columns = ['Sector1Time', 'Sector2Time', 'Sector3Time', 'AvgSectorTime', 
                        'PositionGained', 'PositionLost', 'PositionUnchanged', 'HighFuelLoad', 'MediumFuelLoad', 
                        'LowFuelLoad', 'Soft', 'Medium', 'Hard', 'Intermediate', 'Wet']
    
    # Discretize continuous data into bins
    discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    binned_data = discretizer.fit_transform(winner_laps[relevant_columns])
    
    # Map binned data into categorical labels
    binned_df = pd.DataFrame(binned_data, columns=relevant_columns)
    for column in binned_df.columns:
        binned_df[column] = binned_df[column].map({
            0: column + "_Low",
            1: column + "_Medium",
            2: column + "_High"
        })
    
    # Create transactions in a one-hot encoded format
    transactions = pd.get_dummies(binned_df)
    return transactions

# Apply Apriori and generate association rules
def apply_apriori(transactions, min_support=0.2, min_confidence=0.5, min_lift=1.0):
    # Generate frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets = len(transactions))
    
    # Filter rules to compare across columns only
    filtered_rules = rules[
        rules['antecedents'].apply(lambda x: len(set(map(lambda item: item.split("_")[0], x))) > 1) &
        rules['consequents'].apply(lambda x: len(set(map(lambda item: item.split("_")[0], x))) > 1)
    ]
    
    return frequent_itemsets, filtered_rules

# Main Loop
teams = ['Ferrari', 'Mercedes', 'Redbull']
years = ['2018', '2019', '2021', '2022', '2023']

for team in teams:
    for year in years:
        print("\n--- Analyzing data for Team: {}, Year: {} ---".format(team, year))
        file_path = f"TransformedData/{team}/DriverLapTimes/{year}_combined_encoded.csv"
        
        if not os.path.exists(file_path):
            print("File {} does not exist. Skipping...".format(file_path))
            continue
        
        df = pd.read_csv(file_path)
        features = df[['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'AvgSectorTime', 
                       'PositionGained', 'PositionLost', 'PositionUnchanged', 'HighFuelLoad', 'MediumFuelLoad', 
                       'LowFuelLoad', 'Soft', 'Medium', 'Hard', 'Intermediate', 'Wet']].values

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        som_size = max(4, int(np.sqrt(features_scaled.shape[0]) * 0.4))  # Less strict SOM
        som = MiniSom(som_size, som_size, features_scaled.shape[1], sigma=0.7, learning_rate=0.5)
        som.random_weights_init(features_scaled)
        som.train_random(features_scaled, num_iteration=400)

        metrics, win_df = evaluate_and_visualize_som_clusters(som, df, features_scaled, team, year)
        transactions = preprocess_winner_laps(win_df)
        frequent_itemsets, rules = apply_apriori(transactions)

        # Save Frequent Itemsets to CSV
        frequent_itemsets_file = f"TransformedData/{team}/FrequentItemsets_{year}.csv"
        frequent_itemsets.to_csv(frequent_itemsets_file, index=False)
        print(f"Saved Frequent Itemsets for {team} - {year} to {frequent_itemsets_file}")
        
        # Save Association Rules to CSV
        rules_file = f"TransformedData/{team}/AssociationRules_{year}.csv"
        rules.to_csv(rules_file, index=False)
        print(f"Saved Association Rules for {team} - {year} to {rules_file}")
        
        print("Frequent Itemsets for {} - {}:\n{}".format(team, year, frequent_itemsets))
        print("Association Rules for {} - {}:\n{}".format(team, year, rules))


# # Close the log file after all operations
# sys.stdout.close()
# print(f"Results have been saved to {output_file}.")