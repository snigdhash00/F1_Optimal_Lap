import os
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Path to the TransformedData folder
base_path = "TransformedData"

# Binary Encoding Function
def binary_encode(column, num_bits):
    """
    Encodes a numeric column into binary with fixed number of bits.
    """
    return column.apply(lambda x: list(map(int, bin(x)[2:].zfill(num_bits))))

# Process each team
for team in os.listdir(base_path):
    team_path = os.path.join(base_path, team)
    if not os.path.isdir(team_path):
        continue

    driver_lap_times_path = os.path.join(team_path, "DriverLapTimes")
    if not os.path.exists(driver_lap_times_path):
        continue

    for file in os.listdir(driver_lap_times_path):
        if file.endswith("_combined.csv"):  # Process only combined CSV files
            file_path = os.path.join(driver_lap_times_path, file)
            print(f"Processing file: {file_path}")

            # Load the CSV file
            df = pd.read_csv(file_path)

            # One-hot encode the 'driver' column, but use generic labels (driver_1, driver_2, etc.)
            drivers = df['driver'].unique()  # Get unique driver names
            driver_mapping = {driver: f"{i+1}" for i, driver in enumerate(drivers)}  # Create mapping
            df['driver'] = df['driver'].map(driver_mapping)  # Map drivers to generic names

            # One-hot encode the 'driver' column
            onehot_encoder = OneHotEncoder(sparse_output=False)
            driver_encoded = onehot_encoder.fit_transform(df[['driver']])
            driver_columns = onehot_encoder.get_feature_names_out(['driver'])
            driver_df = pd.DataFrame(driver_encoded, columns=driver_columns)

            # Binary encode the 'race_number' column
            # Convert to Python int to use bit_length()
            max_race_number = int(df['race_number'].max())  # Convert to Python int
            num_bits = max_race_number.bit_length()  # Determine number of bits needed
            binary_encoded = pd.DataFrame(binary_encode(df['race_number'], num_bits).tolist(),
                                          columns=[f"race_bit_{i}" for i in range(num_bits)])

            # Combine encoded columns with the original DataFrame
            df = pd.concat([df.drop(columns=['driver', 'race_number']), driver_df, binary_encoded], axis=1)

            # Save the updated CSV
            output_file = os.path.join(driver_lap_times_path, f"{file.split('.')[0]}_encoded.csv")
            df.to_csv(output_file, index=False)
            print(f"Encoded file saved at: {output_file}")
