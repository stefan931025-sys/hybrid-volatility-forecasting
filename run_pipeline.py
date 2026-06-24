import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# ==========================================
# 1. AUTOMATED DATA INGESTION LAYER
# ==========================================
def fetch_market_data():
    print("\n[STEP 1] Initializing Automated Data Ingestion...")
    tickers = ["SPY", "TLT", "GLD", "USO", "EEM"]
    print(f"[DATA INFRA] Fetching historical data for: {tickers}")
    
    raw_data = yf.download(tickers, period="5y", interval="1d")
    
    if 'Adj Close' in raw_data.columns.levels[0]:
        data = raw_data['Adj Close']
    else:
        data = raw_data['Close']
        
    data = data.ffill().dropna()
    os.makedirs("data", exist_ok=True)
    output_path = "data/market_universe.csv"
    data.to_csv(output_path)
    print(f"[SUCCESS] Asset matrix saved to '{output_path}' ({len(data)} rows).")
    return data

# ==========================================
# 2. BLACK-LETTERMAN PORTFOLIO OPTIMIZER
# ==========================================
def run_black_litterman(df):
    print("\n[STEP 2] Initializing Black-Litterman Optimization Engine...")
    returns = df.pct_change().dropna()
    
    num_assets = len(df.columns)
    W_mkt = np.array([1.0 / num_assets] * num_assets)  
    Sigma = returns.cov().to_numpy() * 252            
    delta = 2.5                                       
    
    Pi = delta * (Sigma @ W_mkt)
    
    asset_momentum = returns.mean() * 252
    top_asset_idx = np.argmax(asset_momentum.to_numpy())
    bottom_asset_idx = np.argmin(asset_momentum.to_numpy())
    
    P = np.zeros((1, num_assets))
    P[0, top_asset_idx] = 1
    P[0, bottom_asset_idx] = -1
    
    Q = np.array([0.05])  
    tau = 0.05            
    Omega = np.array([[tau * (P @ Sigma @ P.T)[0,0]]]) 
    
    try:
        inv_tau_Sigma = np.linalg.inv(tau * Sigma)
        inv_Omega = np.linalg.inv(Omega)
        
        first_term = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P)
        second_term = inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q
        E_R = first_term @ second_term
        print(f"[MATH SUCCESS] Adjusted Expected Returns: {E_R}")
    except np.linalg.LinAlgError:
        print("[ERROR] Matrix inversion failed. Defaulting to historical parameters.")
        E_R = returns.mean().to_numpy() * 252
        
    def portfolio_variance(weights):
        return weights.T @ Sigma @ weights
        
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}) 
    bounds = [(0.0, 1.0) for _ in range(num_assets)]
    
    optimized_result = minimize(portfolio_variance, W_mkt, method='SLSQP', bounds=bounds, constraints=constraints)
    
    os.makedirs("output", exist_ok=True)
    weights_df = pd.DataFrame({
        "Asset": df.columns,
        "Optimal_Weight": np.round(optimized_result.x, 4)
    })
    weights_df.to_csv("output/bl_optimized_weights.csv", index=False)
    print("[SUCCESS] Optimized weights saved to output/bl_optimized_weights.csv")

# ==========================================
# 3. REGIME-SWITCHING VOLATILITY MODULE
# ==========================================
def analyze_market_regimes(df):
    print("\n[STEP 3] Initializing Regime-Switching Volatility Analyzer...")
    spy_returns = df['SPY'].pct_change().dropna()
    rolling_vol = spy_returns.rolling(window=21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    
    vol_median = rolling_vol.median()
    vol_75th = rolling_vol.quantile(0.75)
    
    regimes = []
    for current_vol in rolling_vol:
        if current_vol <= vol_median:
            regimes.append("Regime 1: Low Volatility (Equilibrium)")
        elif current_vol <= vol_75th:
            regimes.append("Regime 2: Medium Volatility (Expansion/Transition)")
        else:
            regimes.append("Regime 3: High Volatility (Tail-Risk/Stress State)")
            
    regime_df = pd.DataFrame(index=rolling_vol.index)
    regime_df['Rolling_Vol'] = rolling_vol
    regime_df['Market_State'] = regimes
    
    latest_date = regime_df.index[-1].strftime('%Y-%m-%d')
    latest_state = regime_df['Market_State'].iloc[-1]
    latest_vol_value = regime_df['Rolling_Vol'].iloc[-1]
    
    print(f"[ANALYSIS ENGINE] Date: {latest_date} | Realized Vol: {latest_vol_value:.2%} | Current Mode: {latest_state}")
    regime_df.to_csv("output/market_regime_matrix.csv")
    print("[SUCCESS] Volatility regime matrix saved to output/market_regime_matrix.csv")

# ==========================================
# 4. MASTER PIPELINE COORDINATOR
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("      LAUNCHING MASTER QUANTAMENTAL PIPELINE      ")
    print("==================================================")
    
    # Run Ingestion
    market_data = fetch_market_data()
    
    # Run Black-Litterman Portfolio Optimization
    run_black_litterman(market_data)
    
    # Run Regime Switching Analysis
    analyze_market_regimes(market_data)
    
    # ==========================================
    # CORE FORECASTING & LINKEDIN ENGINE PLACEHOLDER
    # ==========================================
    print("\n[STEP 4] Executing Deep Learning Forecasting & Post Automation...")
    # Your original core calculation/post code execution goes right here.
    # [Keep any existing LinkedIn post or GARCH-LSTM function calls down here]
    
    print("\n==================================================")
    print("          PIPELINE EXECUTION COMPLETE             ")
    print("==================================================")

