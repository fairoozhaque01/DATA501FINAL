import pandas as pd
crime_df = pd.read_csv("calgary_crime_cleaned.csv")  # already cleaned
population_df = pd.read_csv("calgary_population_estimated.csv")
ward_income = pd.read_csv("Household_Income.csv", thousands=",")
community_ward_map = pd.read_csv("Communities_by_Ward_20250404.csv")  # contains 'NAME', 'WARD_NUM'


crime_df["Community"] = crime_df["Community"].str.lower().str.strip()
community_ward_map["Community"] = community_ward_map["NAME"].str.lower().str.strip()

community_ward_map.rename(columns={"WARD_NUM": "Ward"}, inplace=True)
print(community_ward_map.head())

ward_income["Ward_Clean"] = ward_income["Ward"].astype(str).str.extract(r"(\d+)")[0].astype(int)
print(ward_income.head())
crime_df = pd.merge(crime_df, community_ward_map[["Community", "Ward"]], on="Community", how="left")
population_df = pd.merge(population_df, community_ward_map[["Community", "Ward"]], on="Community", how="left")

print(crime_df.head())
print(population_df.head())


crime_by_ward = crime_df.groupby("Ward")["Crime Count"].sum().reset_index()
pop_by_ward = population_df.groupby("Ward")["Population"].sum().reset_index()
print(crime_by_ward.head())
print(pop_by_ward.head())

crime_by_ward = pd.merge(crime_by_ward, pop_by_ward, on="Ward", how="left")

crime_by_ward["Crime Rate"] = crime_by_ward["Crime Count"] / crime_by_ward["Population"] * 1000
print(crime_by_ward.head())
income_df = pd.read_csv("ward_estimated_income.csv")

combined_df = pd.merge(crime_by_ward, income_df, on="Ward", how="left")
print(combined_df.head())
combined_df.to_csv("crime_population_income_ward.csv", index=False)
