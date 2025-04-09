import pandas as pd
import seaborn as sns


population_df = pd.read_csv("Census_by_Community_2019_20250302.csv")  

# Preview data
print(population_df.head())

print(population_df.columns)


population_df["NAME"] = population_df["NAME"].str.lower().str.strip()

population_df.rename(columns={"NAME": "Community", "CNSS_YR": "Year", "RES_CNT": "Population"}, inplace=True)

population_df["Community"] = population_df["Community"].str.lower().str.strip()

population_df = population_df[["Year", "Community", "Population"]]

population_df = population_df[~population_df["Community"].str.match(r'^\d', na=False)]  # Removes names starting with a digit


population_df.to_csv("calgary_population_cleaned.csv", index=False)

print("Cleaned population dataset saved as calgary_population_cleaned.csv")



population_df = pd.read_csv("calgary_population_cleaned.csv")

# Assume a 1.07% annual growth rate
growth_rate = 1.07

# Extend population estimates for 2021-2024
for year in range(2021, 2024):
    new_year_data = population_df.copy()
    new_year_data["Year"] = year
    new_year_data["Population"] = (new_year_data["Population"] * growth_rate).astype(int)
    population_df = pd.concat([population_df, new_year_data], ignore_index=True)

population_df.to_csv("calgary_population_estimated.csv", index=False)

print("Updated population dataset saved as calgary_population_estimated.csv")


crime_df = pd.read_csv("calgary_crime_cleaned.csv")

population_df = pd.read_csv("calgary_population_estimated.csv")

crime_merged = crime_df.merge(population_df, on=["Year", "Community"], how="left")

crime_merged["Crime Rate per 1K"] = (crime_merged["Crime Count"] / crime_merged["Population"]) * 1000

crime_merged.to_csv("calgary_crime_merged.csv", index=False)

print("Merged dataset saved as calgary_crime_merged.csv")


import matplotlib.pyplot as plt

crime_by_year = crime_merged.groupby("Year")["Crime Count"].sum()

plt.figure(figsize=(10, 5))
plt.plot(crime_by_year.index, crime_by_year.values, marker="o", linestyle="-", color="blue")

plt.title("Crime Trends in Calgary (2021-2025)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Crime Count", fontsize=12)
plt.grid(True)

plt.show()




# Get top 5 high-crime communities
top_communities = crime_merged.groupby("Community")["Crime Count"].sum().nlargest(5).index

crime_top_5 = crime_merged[crime_merged["Community"].isin(top_communities)]

plt.figure(figsize=(12, 6))
sns.lineplot(x="Year", y="Crime Count", hue="Community", data=crime_top_5, marker="o")

plt.title("Crime Trends in Top 5 High-Crime Communities (2021-2025)", fontsize=14)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Crime Count", fontsize=12)
plt.legend(title="Community")
plt.grid(True)

plt.show()

