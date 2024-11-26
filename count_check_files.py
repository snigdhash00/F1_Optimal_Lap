import os
import pandas as pd

# def count_rows_in_original_files(team_path):
#     """Counts total rows in the original CSV files for a team."""
#     total_rows = 0
#     driver_lap_times_path = os.path.join(team_path, "DriverLapTimes")
    
#     if not os.path.exists(driver_lap_times_path):
#         return total_rows

#     for year in os.listdir(driver_lap_times_path):  # Loop through years
#         year_path = os.path.join(driver_lap_times_path, year)
#         if not os.path.isdir(year_path):
#             continue

#         for race in os.listdir(year_path):  # Loop through race numbers
#             race_path = os.path.join(year_path, race)
#             if not os.path.isdir(race_path):
#                 continue

#             # Count rows in each file for the race
#             for file in os.listdir(race_path):
#                 if file.endswith("_LapTimes.csv"):
#                     file_path = os.path.join(race_path, file)
#                     df = pd.read_csv(file_path)
#                     total_rows += len(df)
#     return total_rows

# def count_rows_in_combined_file(team_path, year):
#     """Counts total rows in the combined CSV file for a year."""
    # combined_file_path = os.path.join(team_path, "DriverLapTimes", f"{year}_combined.csv")
#     if os.path.exists(combined_file_path):
#         combined_df = pd.read_csv(combined_file_path)
#         return len(combined_df)
#     return 0

# # Base path is the current directory (TransformedData folder)
# base_path = "TransformedData"

# # Verify for each team
# for team in os.listdir(base_path):  # Loop through teams
#     team_path = os.path.join(base_path, team)
#     if not os.path.isdir(team_path):
#         continue

#     print(f"Verifying rows for team: {team}")
#     driver_lap_times_path = os.path.join(team_path, "DriverLapTimes")

#     if not os.path.exists(driver_lap_times_path):
#         print(f"No DriverLapTimes folder found for {team}. Skipping...")
#         continue

#     for year in os.listdir(driver_lap_times_path):  # Loop through years
#         year_path = os.path.join(driver_lap_times_path, year)
#         if not os.path.isdir(year_path):
#             continue

#         # Count rows in original files
#         original_rows = count_rows_in_original_files(team_path)

#         # Count rows in combined file
#         combined_rows = count_rows_in_combined_file(team_path, year)

#         # Compare and print the results
#         print(f"Year: {year}")
#         print(f"  Total rows in original CSV files: {original_rows}")
#         print(f"  Total rows in combined CSV file: {combined_rows}")

#         if original_rows == combined_rows:
#             print(f"  ✅ Verification Passed: No rows omitted for {team}, {year}.")
#         else:
#             print(f"  ❌ Verification Failed: {original_rows - combined_rows} rows omitted for {team}, {year}.")













path = "TransformedData/Ferrari/DriverLapTimes/"
count=0

for i in range (1,23):
    race_path = f'{path}2023/{i}'
    print(race_path)
    if os.path.exists(f'{race_path}/LEC_LapTimes.csv'):
        data1 = pd.read_csv(f'{race_path}/LEC_LapTimes.csv')
        print("Data 1: ", len(data1))
        count += len(data1)
    if os.path.exists(f'{race_path}/SAI_LapTimes.csv'):
        data2 = pd.read_csv(f'{race_path}/SAI_LapTimes.csv')
        print("Data 2: ", len(data2))
        count += len(data2)
    print("Count: " ,count)
print("Final: ", count)


data = pd.read_csv(f'{path}2023_combined.csv')
print("Combined: ", len(data))

if count == len(data):
    print("Same")
else:
    print("Diff: ", len(data)-count)











# import os
# import pandas as pd

# # Path to the TransformedData folder
# path = "TransformedData/Ferrari/DriverLapTimes/"
# count = 0

# # Debugging: Store row counts for each race
# individual_counts = {}

# # Loop through all race folders (1 to 21)
# for i in range(1, 23):
#     race_path = f'{path}2023/{i}'
#     print(f"Processing race: {race_path}")

#     race_count = 0

#     # Check if LEC_LapTimes.csv exists and count rows
#     if os.path.exists(f'{race_path}/LEC_LapTimes.csv'):
#         data1 = pd.read_csv(f'{race_path}/LEC_LapTimes.csv')
#         print(f"Data 1 (LEC): {len(data1)} rows")
#         race_count += len(data1)
#     else:
#         print(f"LEC_LapTimes.csv not found in {race_path}")

#     # Check if SAI_LapTimes.csv exists and count rows
#     if os.path.exists(f'{race_path}/SAI_LapTimes.csv'):
#         data2 = pd.read_csv(f'{race_path}/SAI_LapTimes.csv')
#         print(f"Data 2 (SAI): {len(data2)} rows")
#         race_count += len(data2)
#     else:
#         print(f"SAI_LapTimes.csv not found in {race_path}")

#     individual_counts[i] = race_count  # Store count for the current race
#     count += race_count  # Update total count for all races
#     print(f"Race {i} total count: {race_count}")
#     print(f"Running total count: {count}")

# # After processing all races
# print(f"Final count from individual files: {count}")

# # Load the combined CSV to check against
# combined_file_path = f'{path}2023_combined.csv'
# data_combined = pd.read_csv(combined_file_path)
# print(f"Rows in combined CSV: {len(data_combined)}")

# # Verify if the total row count matches the combined CSV
# if count == len(data_combined):
#     print("✅ Row counts match!")
# else:
#     print(f"❌ Row counts do not match. Difference: {len(data_combined) - count} rows.")

# # Check for rows in the combined file that have race_numbers not found in individual files
# for race, race_count in individual_counts.items():
#     race_rows = data_combined[data_combined['race_number'] == race]
#     print(f"Race {race}: {race_count} rows processed | {len(race_rows)} rows in combined CSV")

#     if len(race_rows) != race_count:
#         print(f"❌ Mismatch for Race {race}. Expected {race_count} rows, but found {len(race_rows)}.")
#     else:
#         print(f"✅ Race {race}: Row count matches.")

# # Find rows in the combined CSV that have race numbers not in the processed individual files
# all_race_numbers = list(individual_counts.keys())
# unexpected_rows = data_combined[~data_combined['race_number'].isin(all_race_numbers)]

# # Check for duplicate rows in the combined CSV
# duplicates = data_combined[data_combined.duplicated()]

# # Output results
# print(f"\nRows in combined CSV with unexpected race numbers:")
# print(unexpected_rows)

# print(f"\nDuplicate rows in the combined CSV:")
# print(duplicates)

# # Optionally, remove duplicates and check if the row count matches after cleanup
# data_combined_clean = data_combined.drop_duplicates()
# print(f"Rows in combined CSV after removing duplicates: {len(data_combined_clean)}")

# if len(data_combined_clean) == count:
#     print("✅ Row counts match after removing duplicates.")
# else:
#     print(f"❌ Row counts still do not match after removing duplicates. Difference: {len(data_combined_clean) - count} rows.")

