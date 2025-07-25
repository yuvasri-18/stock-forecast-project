import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Load data
df = pd.read_csv("data/AAPL_stock.csv", index_col=0)

# Reset index and rename columns
df.reset_index(inplace=True)
if 'index' in df.columns:
    df.rename(columns={'index': 'ds'}, inplace=True)
elif 'Date' in df.columns:
    df.rename(columns={'Date': 'ds'}, inplace=True)
else:
    df.columns.values[0] = 'ds'

df.rename(columns={'Close': 'y'}, inplace=True)

# Clean data
df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
df['y'] = pd.to_numeric(df['y'], errors='coerce')
df.dropna(inplace=True)

# Train/test split
train_size = int(len(df) * 0.9)
train, test = df[:train_size], df[train_size:]

# Prophet model
model = Prophet()
model.fit(train)

# Forecast
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# ✅ Safe evaluation
merged = pd.merge(test, forecast[['ds', 'yhat']], on='ds', how='inner')
rmse = np.sqrt(mean_squared_error(merged['y'], merged['yhat']))
print(f"📉 Prophet RMSE: {rmse:.2f}")

# Plot forecast
plt.figure(figsize=(14,6))
plt.plot(train['ds'], train['y'], label='Train', color='blue')
plt.plot(test['ds'], test['y'], label='Actual', color='green')
plt.plot(merged['ds'], merged['yhat'], label='Predicted (Prophet)', color='purple')
plt.title("Prophet Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/prophet_forecast.png")
plt.show()
