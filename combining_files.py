import os
import pandas as pd

# Path to the TransformedData folder
base_path = "TransformedData"

# Combine all files for each year
for team in os.listdir(base_path):  # Loop through teams
    team_path = os.path.join(base_path, team)
    if not os.path.isdir(team_path):
        continue

    driver_lap_times_path = os.path.join(team_path, "DriverLapTimes")
    if not os.path.exists(driver_lap_times_path):
        continue

    for year in os.listdir(driver_lap_times_path):  # Loop through years
        year_path = os.path.join(driver_lap_times_path, year)
        if not os.path.isdir(year_path):
            continue

        yearly_combined_df = pd.DataFrame()
        for race in os.listdir(year_path):  # Loop through race numbers
            race_path = os.path.join(year_path, race)
            if not os.path.isdir(race_path):
                continue

            # Process each driver CSV file in the race folder
            for file in os.listdir(race_path):
                if file.endswith("_LapTimes.csv"):  # Check for valid CSV files
                    file_path = os.path.join(race_path, file)
                    print(f"Processing file: {file_path}")

                    # Extract driver name from the file name
                    driver_name = file.split("_")[0]

                    # Read the CSV and add driver and race_number columns
                    df = pd.read_csv(file_path)
                    df['driver'] = driver_name  # Add driver column
                    df['race_number'] = int(race)  # Add race number column
                    yearly_combined_df = pd.concat([yearly_combined_df, df], ignore_index=True)

        # Save the yearly combined CSV file
        if not yearly_combined_df.empty:
            output_file = os.path.join(driver_lap_times_path, f"{year}_combined.csv")
            yearly_combined_df.to_csv(output_file, index=False)
            print(f"Combined CSV for {team}, {year} saved at {output_file}.")
        else:
            print(f"No valid data found for {team}, {year}. Skipping...")
