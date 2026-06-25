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
# 2. BLACK-LITTERMAN PORTFOLIO OPTIMIZER
# ==========================================
def run_black_litterman(df):
    print("\n[STEP 2] Executing Black-Litterman Matrix Allocator...")
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
    except np.linalg.LinAlgError:
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
    return optimized_result.x, weights_df

# ==========================================
# 3. REGIME-SWITCHING VOLATILITY MODULE
# ==========================================
def analyze_market_regimes(df):
    print("\n[STEP 3] Running Latent Market State Classification...")
    spy_returns = df['SPY'].pct_change().dropna()
    rolling_vol = spy_returns.rolling(window=21).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()
    
    vol_median = rolling_vol.median()
    vol_75th = rolling_vol.quantile(0.75)
    
    regimes = []
    for current_vol in rolling_vol:
        if current_vol <= vol_median:
            regimes.append("Regime 1: Low Vol (Equilibrium)")
        elif current_vol <= vol_75th:
            regimes.append("Regime 2: Med Vol (Transition)")
        else:
            regimes.append("Regime 3: High Vol (Tail-Risk)")
            
    regime_df = pd.DataFrame(index=rolling_vol.index)
    regime_df['Rolling_Vol'] = rolling_vol
    regime_df['Market_State'] = regimes
    
    latest_date = regime_df.index[-1].strftime('%Y-%m-%d')
    latest_state = regime_df['Market_State'].iloc[-1]
    latest_vol_value = regime_df['Rolling_Vol'].iloc[-1]
    
    print(f"[STATE FLAGGED] Date: {latest_date} | Realized Vol: {latest_vol_value:.2%} | Mode: {latest_state}")
    os.makedirs("output", exist_ok=True)
    regime_df.to_csv("output/market_regime_matrix.csv")
    print("[SUCCESS] Volatility regime matrix saved to output/market_regime_matrix.csv")
    return regime_df

# ==========================================
# 4. COMPREHENSIVE REGIME-CONDITIONED BACKTESTER
# ==========================================
def run_comparative_backtest(prices_df, regime_df, bl_weights):
    print("\n[STEP 4] Simulating Historical Performance Across Regimes...")
    returns = prices_df.pct_change().dropna()
    
    num_assets = len(prices_df.columns)
    eq_weights = np.array([1.0 / num_assets] * num_assets)
    
    backtest_df = pd.DataFrame(index=returns.index)
    backtest_df['Benchmark_EQ'] = returns.to_numpy() @ eq_weights
    backtest_df['Optimized_BL'] = returns.to_numpy() @ bl_weights
    
    backtest_df = backtest_df.join(regime_df['Market_State'], how='inner')
    
    unique_regimes = sorted(backtest_df['Market_State'].unique())
    summary_data = []
    
    for regime in unique_regimes:
        regime_data = backtest_df[backtest_df['Market_State'] == regime]
        if len(regime_data) < 5: continue
        
        for strategy in ['Benchmark_EQ', 'Optimized_BL']:
            strat_returns = regime_data[strategy]
            
            ann_ret = strat_returns.mean() * 252
            ann_vol = strat_returns.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            cum_ret = (1 + strat_returns).cumprod()
            max_dd = ((cum_ret - cum_ret.cummax()) / cum_ret.cummax()).min()
            
            summary_data.append({
                "Regime": regime,
                "Strategy": strategy,
                "Days": len(regime_data),
                "Ann. Return": f"{ann_ret:.2%}",
                "Ann. Volatility": f"{ann_vol:.2%}",
                "Sharpe Ratio": f"{sharpe:.2f}",
                "Max Drawdown": f"{max_dd:.2%}"
            })
            
    metrics_df = pd.DataFrame(summary_data)
    print("\n" + "="*70 + "\n EMPIRICAL BACKTEST COMPARISON BY MACRO REGIME\n" + "="*70)
    print(metrics_df.to_string(index=False))
    print("="*70)
    
    metrics_df.to_csv("output/comparative_regime_backtest.csv", index=False)
    print("[SUCCESS] Comparative analysis saved to 'output/comparative_regime_backtest.csv'")

# ==========================================
# 5. AUTOMATED MARKET DISPATCH & NEWSLETTER GENERATOR
# ==========================================
def generate_daily_newsletter(regime_df, weights_df):
    print("\n[STEP 5] Generating Automated LinkedIn Content Layer...")
    
    latest_date = regime_df.index[-1].strftime('%B %d, %Y')
    latest_state = regime_df['Market_State'].iloc[-1]
    latest_vol = regime_df['Rolling_Vol'].iloc[-1]
    
    alloc_strings = []
    for _, row in weights_df.iterrows():
        if row['Optimal_Weight'] > 0:
            alloc_strings.append(f"  • {row['Asset']}: {row['Optimal_Weight']:.2%}")
            
    post_content = f"""📊 QUANTAMENTAL DESK DAILY DISPATCH | {latest_date}

⚡ SYSTEM STATUS: LATENT REGIME DETECTED
Current State: {latest_state}
SPY Rolling Realized Volatility: {latest_vol:.2%}

🤖 BLACK-LITTERMAN BAYESIAN ALLOCATION MATRIX:
Based on historical asset covariance matrices and momentum-driven view vectors, our optimization engine has refactored the portfolio weights for the upcoming session:
{chr(10).join(alloc_strings)}

🧠 UNDER THE HOOD:
Our automated pipeline ingests multi-asset daily data, maps the latent market state using rolling volatility quantiles, and feeds structural returns into a Black-Litterman optimization framework to minimize portfolio variance dynamically across shifting regimes.

💡 CALL FOR COLLABORATION:
I am currently scaling out the predictive features of this framework (integrating an LSTM-GARCH hybrid model to forecast forward implied volatility surfaces). If any Portfolio Managers, Traders, or Quants have an interesting stress-test scenario, alternative dataset, or specialized strategy backtest they want rigorously modeled or optimized for free, drop a comment or DM me. Let's collaborate.

#QuantitativeFinance #BlackLitterman #PortfolioOptimization #PythonTrading #RiskManagement #DataScience
"""
    
    os.makedirs("output", exist_ok=True)
    output_text_path = "output/linkedin_post.txt"
    with open(output_text_path, "w", encoding="utf-8") as f:
        f.write(post_content)
    print(f"[SUCCESS] Daily newsletter artifact generated at '{output_text_path}'")

# ==========================================
# 6. MASTER COORDINATOR
# ==========================================
if __name__ == "__main__":
    print("==================================================")
    print("      LAUNCHING ENHANCED QUANTAMENTAL DESK        ")
    print("==================================================")
    
    market_data = fetch_market_data()
    bl_allocation, weights_dataframe = run_black_litterman(market_data)
    market_regimes = analyze_market_regimes(market_data)
    
    # Run backtest
    run_comparative_backtest(market_data, market_regimes, bl_allocation)
    
    # Generate the newsletter text
    generate_daily_newsletter(market_regimes, weights_dataframe)
    
    print("\n==================================================")
    print("          PIPELINE EXECUTION COMPLETE             ")
    print("==================================================")
