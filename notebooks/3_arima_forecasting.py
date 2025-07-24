import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# === Load and clean the data ===
file_path = "../data/AAPL_stock.csv"
df = pd.read_csv(file_path, index_col=0)

# Remove rows with 'Ticker' in index
df = df[~df.index.str.contains("Ticker", na=False)]

# Convert index to datetime
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[~df.index.isna()]

# Keep only the 'Close' column and clean it
df = df[['Close']]
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)

# === Train/Test Split ===
train_size = int(len(df) * 0.9)
train, test = df[:train_size], df[train_size:]

# === Build and Train ARIMA model ===
model = ARIMA(train['Close'], order=(5, 1, 0))  # ARIMA(p, d, q)
model_fit = model.fit()

# === Forecast ===
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# === Evaluation ===
rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
print(f"📉 RMSE: {rmse:.2f}")

# === Plotting ===
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['Close'], label='Train', color='blue')
plt.plot(test.index, test['Close'], label='Actual', color='green')
plt.plot(forecast.index, forecast, label='Predicted (ARIMA)', color='red')

plt.title("ARIMA Forecast vs Actual", fontsize=16)
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Save and Show Plot ===
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/arima_forecast.png")
plt.show()
