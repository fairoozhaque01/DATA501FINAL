import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


crime_df = pd.read_csv("calgary_crime_cleaned.csv") 
population_df = pd.read_csv("calgary_population_estimated.csv")
ward_income = pd.read_csv("Household_Income.csv", thousands=",")
community_ward_map = pd.read_csv("Communities_by_Ward_20250404.csv")  
crime_df["Community"] = crime_df["Community"].str.lower().str.strip()
community_ward_map["Community"] = community_ward_map["NAME"].str.lower().str.strip()

community_ward_map.rename(columns={"WARD_NUM": "Ward"}, inplace=True)
print(community_ward_map.head())

# Create new clean Ward column from string like 'WARD 1'
ward_income["Ward_Clean"] = ward_income["Ward"].astype(str).str.extract(r"(\d+)")[0].astype(int)
print(ward_income.head())
# Merging 
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




print("\n📊 Correlation: Population vs Total Crime Count")
print(crime_by_ward[["Crime Rate", "Population"]].corr())

sns.scatterplot(data=crime_by_ward, x="Population", y="Crime Count")
plt.title("Total Crime vs Population by Ward")
plt.xlabel("Population")
plt.ylabel("Crime Count")
plt.grid(True)
plt.show()
sns.lmplot(data=crime_by_ward, x="Population", y="Crime Rate", height=5, aspect=1.5)
plt.title("Crime Rate per 1K vs Population (with regression line)")
plt.xlabel("Population")
plt.ylabel("Crime Rate per 1K")
plt.grid(True)
plt.show()

# POPULATION ALONE DO NOT EXPLAIN CRIME RATE CLEARLY

# Convert income columns to numeric
bracket_cols = [
    "Under $20,000", "$20,000 to $39,999", "$40,000 to $59,999", "$60,000 to $79,999",
    "$80,000 to $99,999", "$100,000 to $124,999", "$125,000 to $149,999",
    "$150,000 to $199,999", "$200,000 and over",
    "Total - Household total income groups in 2015 for private households - 25% sample data"
]

for col in bracket_cols:
    ward_income[col] = pd.to_numeric(ward_income[col], errors='coerce').fillna(0)

# Midpoints
brackets = {
    "Under $20,000": 10000,
    "$20,000 to $39,999": 30000,
    "$40,000 to $59,999": 50000,
    "$60,000 to $79,999": 70000,
    "$80,000 to $99,999": 90000,
    "$100,000 to $124,999": 112500,
    "$125,000 to $149,999": 137500,
    "$150,000 to $199,999": 175000,
    "$200,000 and over": 225000
}

for col, mid in brackets.items():
    ward_income[col + " (weighted)"] = ward_income[col] * mid

ward_income["Estimated Mean Income"] = ward_income[
    [col + " (weighted)" for col in brackets]
].sum(axis=1) / ward_income["Total - Household total income groups in 2015 for private households - 25% sample data"]
print(ward_income.head())
# Final cleaned income dataset with integer ward
ward_income_final = ward_income[["Ward_Clean", "Estimated Mean Income"]].copy()
ward_income_final.rename(columns={"Ward_Clean": "Ward"}, inplace=True)
print(ward_income_final.head())

crime_by_ward["Ward"] = crime_by_ward["Ward"].astype(int)
crime_by_ward = pd.merge(crime_by_ward, ward_income_final, on="Ward", how="left")

crime_by_ward.to_csv("final_crime_dataset_wo_weather.csv", index=False)


print("\n📊 Correlation Matrix:")
print(crime_by_ward[["Crime Rate", "Estimated Mean Income"]].corr())

sns.heatmap(crime_by_ward[["Crime Rate", "Estimated Mean Income"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation: Crime Rate vs Estimated Income (Ward Level)")
plt.show()

print("\n📌 Data Preview:")
print(crime_by_ward[["Ward", "Crime Rate", "Estimated Mean Income"]].head(15))


plot_df = crime_by_ward.dropna(subset=["Crime Rate", "Estimated Mean Income"])

sns.scatterplot(data=crime_by_ward, x="Estimated Mean Income", y="Crime Rate")
plt.title("Crime Rate vs Estimated Income by Ward")
plt.xlabel("Estimated Mean Household Income ($)")
plt.ylabel("Crime Rate (per 1,000 residents)")
plt.grid(True)
plt.show()
sns.lmplot(data=crime_by_ward, x="Estimated Mean Income", y="Crime Rate", height=5, aspect=1.5)
plt.title("Crime Rate vs Estimated Mean Income (with regression line)")
plt.xlabel("Esimated Mean Income")
plt.ylabel("Crime Rate per 1,000 residents")
plt.grid(True)
plt.show()





#multivariate linear regression
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

multi_df = crime_by_ward.replace([np.inf, -np.inf], np.nan).dropna(subset=["Crime Rate", "Estimated Mean Income", "Population"])

# Feature matrix (X) and target variable (y)
X = multi_df[["Estimated Mean Income", "Population"]]
y = multi_df["Crime Rate"]


model = LinearRegression()
model.fit(X, y)

intercept = model.intercept_
coefficients = model.coef_
feature_names = X.columns.tolist()

print("📈 Multivariate Linear Regression Results:")
print(f"Intercept: {intercept:.4f}")
for i, coef in enumerate(coefficients):
    print(f"Coefficient for {feature_names[i]}: {coef:.6f}")


# Predictions and R-squared
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f"\n🔍 R-squared: {r2:.3f} — this means that {r2*100:.1f}% of the variation in crime rate is explained by income and population.")

residuals = y - y_pred
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, linestyle='--', color='gray')
plt.xlabel("Predicted Crime Rate")
plt.ylabel("Residuals")
plt.title("Residual Plot: Crime Rate vs Predicted Values")
plt.grid(True)
plt.show()





import statsmodels.api as sm

X = multi_df[["Estimated Mean Income", "Population"]]
y = multi_df["Crime Rate"]

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())


import seaborn as sns
import matplotlib.pyplot as plt

heatmap_data = crime_by_ward[["Crime Rate", "Estimated Mean Income", "Population"]]

corr_matrix = heatmap_data.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap: Crime Rate, Income, and Population")
plt.show()

crime_by_ward.to_csv("final_crime_dataset2.csv", index=False)
