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

df = pd.read_csv("final_crime_dataset.csv")
df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Linear Regression 
X = df[["Estimated Mean Income", "Population"]]
y = df["Crime Rate per 1K"]

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

# LSTM Time-Series Forecasting
df_sorted = df.sort_values(["Year", "Month"])
monthly_series = df_sorted.groupby(["Year", "Month"])["Crime Rate per 1K"].sum().reset_index()
monthly_series["Date"] = pd.to_datetime(monthly_series[["Year", "Month"]].assign(DAY=1))
monthly_series.set_index("Date", inplace=True)

scaler = MinMaxScaler()
scaled_crime = scaler.fit_transform(monthly_series[["Crime Rate per 1K"]])

def create_sequences(data, window=12):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_crime)
X_train_lstm, X_test_lstm = X_lstm[:-6], X_lstm[-6:]
y_train_lstm, y_test_lstm = y_lstm[:-6], y_lstm[-6:]

lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)

y_pred_lstm = lstm_model.predict(X_test_lstm)
y_pred_rescaled = scaler.inverse_transform(y_pred_lstm)
y_test_rescaled = scaler.inverse_transform(y_test_lstm)

print("\n🔁 LSTM Forecasting Results:")
print("MAE:", round(mean_absolute_error(y_test_rescaled, y_pred_rescaled), 2))
print("MSE:", round(mean_squared_error(y_test_rescaled, y_pred_rescaled), 2))
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = round(mse**0.5, 2)
print("RMSE:", rmse)
