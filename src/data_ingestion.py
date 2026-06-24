import os
import pandas as pd
import yfinance as yf

def fetch_market_data():
    print("[SYSTEM INFO] Initializing Automated Data Ingestion Layer...")
    
    # Cleaned, highly-liquid macro assets: S&P500, Long-bonds, Gold, Crude Oil, Emerging Markets
    tickers = ["SPY", "TLT", "GLD", "USO", "EEM"]
    
    print(f"[DATA INFRA] Fetching historical data for: {tickers}")
    
    # Download 5 years of daily data
    raw_data = yf.download(tickers, period="5y", interval="1d")
    
    # Clean extraction method that works across all multi-ticker structures
    if 'Adj Close' in raw_data.columns.levels[0]:
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']
        
    # Check for missing values and forward-fill
    data = data.ffill().dropna()
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV for the forecasting engines to ingest
    output_path = "data/market_universe.csv"
    data.to_csv(output_path)
    print(f"[SUCCESS] Data Ingestion Complete. Asset matrix saved to '{output_path}' ({len(data)} rows).")

if __name__ == "__main__":
    fetch_market_data()
