import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("Final_LSTM_Dataset.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Linear Regression 
X = df[["Estimated Mean Income", "Population", "Mean Temp (°C)"]]
y = df["Crime Rate"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n📊 Linear Regression Results:")
print("Intercept:", round(lr_model.intercept_, 4))
print("Coefficients:", dict(zip(X.columns, lr_model.coef_)))
print("MAE:", round(mean_absolute_error(y_test, y_pred_lr), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred_lr), 2))
mse = mean_squared_error(y_test, y_pred_lr)
rmse = round(mse**0.5, 2)
print("RMSE:", rmse)
print("R² Score:", round(r2_score(y_test, y_pred_lr), 2))

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n🌳 Random Forest Regression Results:")
print("MAE:", round(mean_absolute_error(y_test, y_pred_rf), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred_rf), 2))
mse = mean_squared_error(y_test, y_pred_rf)
rmse = round(mse**0.5, 2)
print("RMSE:", rmse)
print("R² Score:", round(r2_score(y_test, y_pred_rf), 2))

#linear regression plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", marker="o")
plt.plot(y_pred_lr, label="Predicted (Linear Regression)", marker="x")
plt.title("📈 Linear Regression: Actual vs Predicted Crime Rate")
plt.xlabel("Test Sample Index")
plt.ylabel("Crime Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#random forest plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual", marker="o")
plt.plot(y_pred_rf, label="Predicted (Random Forest)", marker="s")
plt.title("🌳 Random Forest: Actual vs Predicted Crime Rate")
plt.xlabel("Test Sample Index")
plt.ylabel("Crime Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#linear vs random forest plot

plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual", marker='o')
plt.plot(y_pred_lr, label="Predicted (Linear Regression)", marker='x')
plt.plot(y_pred_rf, label="Predicted (Random Forest)", marker='s')
plt.title("📈 Actual vs Predicted Crime Rate: Regression Models")
plt.xlabel("Test Sample Index")
plt.ylabel("Crime Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
