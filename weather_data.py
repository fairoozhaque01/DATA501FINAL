import pandas as pd
import glob

path = "weather_data/"

# Get all CSV files from the folder
all_files = glob.glob(path + "*.csv")

df_list = []

for file in all_files:
    df = pd.read_csv(file)  # Read each CSV
    df_list.append(df)  # Append to list

# Combine all DataFrames into one
merged_weather = pd.concat(df_list, ignore_index=True)

# Save the merged dataset
merged_weather.to_csv("calgary_weather_2021_2024.csv", index=False)

print("Merged dataset saved as calgary_weather_2021_2024.csv")
