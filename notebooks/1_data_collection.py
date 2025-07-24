import yfinance as yf
import os

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

# Download stock data
ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2024-12-31")

# Save to CSV in notebooks/data/
data.to_csv("data/AAPL_stock.csv")
print("✅ Data saved to notebooks/data/AAPL_stock.csv")
