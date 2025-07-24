import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# === Load CSV ===
file_path = "../data/AAPL_stock.csv"
df = pd.read_csv(file_path, index_col=0)

# Remove bad rows with 'Ticker' in the index
df = df[~df.index.str.contains("Ticker", na=False)]

# Convert index to datetime
df.index = pd.to_datetime(df.index, errors='coerce')
df = df[~df.index.isna()]  # Remove any invalid dates

# Keep only valid numeric stock columns
valid_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
df = df[[col for col in df.columns if col in valid_cols]]
df.dropna(inplace=True)

# === Plot ===
plt.figure(figsize=(16, 9))  # Wider figure
plt.plot(df.index, df['Close'], color='darkgreen', linewidth=2, label='Close Price')

plt.title('AAPL Closing Price Over Time', fontsize=18, weight='bold')
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price ($)', fontsize=14)

# Format Y-axis: reduce ticks, clean currency format
ymin, ymax = plt.ylim()
yticks = np.linspace(ymin, ymax, num=10)
plt.yticks(yticks, [f'${tick:,.0f}' for tick in yticks], fontsize=12)

# X-axis formatting
plt.xticks(rotation=45, fontsize=12)

# Grid, legend, save
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(fontsize=12)
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/close_price_plot.png")
plt.show()
