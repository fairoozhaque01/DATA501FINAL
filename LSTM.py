import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


df = pd.read_csv("Final_LSTM_Dataset.csv")

# Ensure proper data types
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
df["Crime Count"] = pd.to_numeric(df["Crime Count"], errors="coerce")
df["Population"] = pd.to_numeric(df["Population"], errors="coerce")
df["Estimated Mean Income"] = pd.to_numeric(df["Estimated Mean Income"], errors="coerce")
df["Mean Temp (°C)"] = pd.to_numeric(df["Mean Temp (°C)"], errors="coerce")
df["Total Rain (mm)"] = pd.to_numeric(df["Total Rain (mm)"], errors="coerce")
df["Total Snow (cm)"] = pd.to_numeric(df["Total Snow (cm)"], errors="coerce")
df["Total Precip (mm)"] = pd.to_numeric(df["Total Precip (mm)"], errors="coerce")

# Drop rows with NaN
df.dropna(inplace=True)

# Aggregate monthly data
monthly_crime = df.groupby(["Year", "Month"]).agg({
    "Crime Count": "sum",
    "Population": "mean",
    "Estimated Mean Income": "mean",
    "Mean Temp (°C)": "mean",
    "Total Rain (mm)": "mean",
    "Total Snow (cm)": "mean",
    "Total Precip (mm)": "mean"
}).reset_index()

monthly_crime["Date"] = pd.to_datetime(monthly_crime[["Year", "Month"]].assign(DAY=1))
monthly_crime.sort_values("Date", inplace=True)

features = ["Crime Count", "Population", "Estimated Mean Income", "Mean Temp (°C)", "Total Rain (mm)", "Total Snow (cm)", "Total Precip (mm)"]
scaler = MinMaxScaler()
scaled = scaler.fit_transform(monthly_crime[features])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length][0])  # Crime Count
    return np.array(X), np.array(y)

sequence_length = 6
X, y = create_sequences(scaled, sequence_length)

X_train, X_test = X[:-12], X[-12:]
y_train, y_test = y[:-12], y[-12:]

# LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2]), return_sequences=False),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("🔁 LSTM Forecasting Results (Last 12 months):")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2:.2f}")



plot_dates = monthly_crime["Date"].iloc[-12:]  # last 12 months

plt.figure(figsize=(10, 5))
plt.plot(y_test, label="Actual (Scaled)", marker="o")
plt.plot(y_pred, label="Predicted (LSTM)", marker="^")
plt.title("🔁 LSTM (Scaled): Actual vs Predicted Crime Rate")
plt.xlabel("Test Sample Index")
plt.ylabel("Crime Rate (Scaled)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

n_features = scaled.shape[1]

def inverse_crime_only(scaled_col):
    padded = np.zeros((len(scaled_col), n_features))
    padded[:, 0] = scaled_col  
    return scaler.inverse_transform(padded)[:, 0]  # Return only crime count column

y_test_unscaled = inverse_crime_only(y_test)
y_pred_unscaled = inverse_crime_only(y_pred)
mae = mean_absolute_error(y_test_unscaled, y_pred_unscaled)
mse = mean_squared_error(y_test_unscaled, y_pred_unscaled)
rmse = np.sqrt(mse)

print(f"\n🔁 LSTM Forecasting Results (Unscaled):")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
r2 = r2_score(y_test_unscaled, y_pred_unscaled)
print(f"R² Score: {r2:.2f}")



plt.figure(figsize=(10, 5))
plt.plot(y_test_unscaled, label="Actual Crime Count", marker="o")
plt.plot(y_pred_unscaled, label="Predicted Crime Count (LSTM)", marker="^")
plt.title("🔁 LSTM Forecast: Actual vs Predicted Monthly Crime Count (Unscaled)")
plt.xlabel("Test Sample Index")
plt.ylabel("Crime Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

