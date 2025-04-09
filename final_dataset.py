# Re-import necessary libraries and reload files after kernel reset
import pandas as pd

# Load datasets
crime_df = pd.read_csv("calgary_crime_cleaned.csv")
population_df = pd.read_csv("calgary_population_estimated.csv")
weather_df = pd.read_csv("calgary_weather_2021_2024.csv")
community_ward_map = pd.read_csv("Communities_by_Ward_20250404.csv")
income_df = pd.read_csv("ward_estimated_income.csv")

# Clean community names
crime_df["Community"] = crime_df["Community"].str.lower().str.strip()
population_df["Community"] = population_df["Community"].str.lower().str.strip()
community_ward_map["Community"] = community_ward_map["NAME"].str.lower().str.strip()
community_ward_map = community_ward_map.rename(columns={"WARD_NUM": "Ward"})

# Merge crime data with ward info
crime_df = pd.merge(crime_df, community_ward_map[["Community", "Ward"]], on="Community", how="left")

# Merge population data
population_df = pd.merge(population_df, community_ward_map[["Community", "Ward"]], on="Community", how="left")

# Filter population data for 2021 to 2024 only
population_df = population_df[population_df["Year"].isin([2021, 2022, 2023, 2024])]

# Merge crime with population
crime_df = pd.merge(crime_df, population_df, on=["Community", "Year", "Ward"], how="left")

# Calculate crime rate per 1000
crime_df["Crime Rate per 1K"] = crime_df["Crime Count"] / crime_df["Population"] * 1000

# Merge income
income_df = income_df.rename(columns={"Ward": "Ward", "Estimated Mean Income": "Estimated Mean Income"})
crime_df = pd.merge(crime_df, income_df, on="Ward", how="left")

# Preprocess weather: average weather per month-year
weather_df["Date/Time"] = pd.to_datetime(weather_df["Date/Time"])
weather_df["Year"] = weather_df["Date/Time"].dt.year
weather_df["Month"] = weather_df["Date/Time"].dt.month

monthly_weather = weather_df.groupby(["Year", "Month"]).agg({
    "Mean Temp (°C)": "mean",
    "Total Rain (mm)": "sum",
    "Total Snow (cm)": "sum",
    "Total Precip (mm)": "sum"
}).reset_index()

# Merge weather with crime data
final_df = pd.merge(crime_df, monthly_weather, on=["Year", "Month"], how="left")

# Save final dataset
crime_df.to_csv("final_crime_dataset.csv", index=False)

