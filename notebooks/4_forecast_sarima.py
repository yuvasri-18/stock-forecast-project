import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import os

# Load and preprocess data
df = pd.read_csv("data/AAPL_stock.csv", index_col=0)
df = df[~df.index.str.contains("Ticker", na=False)]
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[~df.index.isna()]
df = df[['Close']]
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(inplace=True)

# Train/test split
train_size = int(len(df) * 0.9)
train, test = df[:train_size], df[train_size:]

# Fit SARIMA model
model = SARIMAX(train['Close'], order=(5,1,0), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=len(test))
forecast.index = test.index

# Evaluate
rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
print(f"📉 SARIMA RMSE: {rmse:.2f}")

# Plot forecast
plt.figure(figsize=(14,6))
plt.plot(train.index, train['Close'], label='Train', color='blue')
plt.plot(test.index, test['Close'], label='Actual', color='green')
plt.plot(forecast.index, forecast, label='Predicted (SARIMA)', color='orange')
plt.title("SARIMA Forecast vs Actual")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/sarima_forecast.png")
plt.show()
