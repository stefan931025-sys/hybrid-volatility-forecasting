import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def generate_fx_dataset(ticker="EURUSD=X", period="5y", output_file="sample_fx_data.csv"):
    """
    Downloads historical daily foreign exchange price data up to the current date,
    computes log returns, cleans missing values, and saves to CSV for model consumption.
    """
    print(f"[*] Fetching live market data for {ticker} over the last {period}...")
    
    # Download daily market data using yfinance
    df = yf.download(ticker, period=period, interval="1d", progress=False)

    if df.empty:
        raise ValueError(f"Failed to retrieve data for ticker {ticker}.")

    # Flatten MultiIndex columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Clean missing values forward and backward
    df = df.ffill().bfill()

    # Calculate Daily Logarithmic Returns: ln(Pt / Pt-1)
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Calculate Rolling Historical Volatility (21-day annualized) as a secondary benchmark
    df['Realized_Vol_21D'] = df['Log_Return'].rolling(window=21).std() * np.sqrt(252)

    # Drop the initial NaN row created by the shift operation
    df_clean = df.dropna().reset_index()

    # Select and rename relevant columns for clean pipeline ingestion
    df_final = df_clean[['Date', 'Close', 'Log_Return', 'Realized_Vol_21D']].copy()
    df_final.columns = ['date', 'close_price', 'log_return', 'realized_vol_21d']

    # Export to CSV
    df_final.to_csv(output_file, index=False)
    
    print(f"[+] Successfully generated sample dataset: '{output_file}'")
    print(f"[+] Total Observations: {len(df_final)} rows")
    print(f"[+] Date Range: {df_final['date'].min().strftime('%Y-%m-%d')} to {df_final['date'].max().strftime('%Y-%m-%d')}")
    print("\nFirst 5 rows of sample dataset:")
    print(df_final.head())

if __name__ == "__main__":
    generate_fx_dataset()
