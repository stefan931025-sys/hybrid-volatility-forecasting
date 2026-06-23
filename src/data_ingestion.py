import os
import pandas as pd
import yfinance as yf

def fetch_market_data():
    print("[SYSTEM INFO] Initializing Automated Data Ingestion Layer...")
    
    # Define a cross-asset universe: Equities, Fixed Income, Commodities, FX
    ticker_universe = {
        "SPY": "S&P 500 Index (Equities)",
        "TLT": "20+ Year Treasury Bond ETF (Fixed Income)",
        "GLD": "Gold Trust (Safe Haven Commodity)",
        "USO": "United States Oil Fund (Energy/Macro)",
        "EEM": "MSCI Emerging Markets ETF (Macro Risk Asset)"
    }
    
    tickers = list(ticker_universe.keys())
    
    # Download 5 years of daily adjusted closing prices
    print(f"[DATA INFRA] Fetching historical data for: {tickers}")
    data = yf.download(tickers, period="5y", interval="1d")['Adj Close']
    
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
