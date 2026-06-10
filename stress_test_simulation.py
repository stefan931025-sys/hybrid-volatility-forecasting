import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t

def run_monte_carlo_stress_test(csv_path='out_of_sample_results.csv', steps=30, paths=10000, df_degrees=4):
    """
    Simulates 10,000 future return paths using GARCH-LSTM predicted volatility 
    under a heavy-tailed Student's t-distribution to map maximum tail-loss.
    """
    try:
        data = pd.read_csv(csv_path)
        # Grab the latest forward-looking volatility forecast
        initial_vol = data['Predicted_Vol'].iloc[-1]
    except FileNotFoundError:
        print("Data layer not found. Initializing baseline stress parameters...")
        initial_vol = 0.02 # 2% baseline daily conditional volatility proxy

    np.random.seed(42)
    
    # Initialize simulation matrix (steps x paths)
    simulated_paths = np.zeros((steps, paths))
    
    # Generate shocks using Student's t-distribution to capture fat tails
    # Scaling the random shocks by the forecasted conditional volatility
    for p in range(paths):
        shocks = t.rvs(df=df_degrees, size=steps) * initial_vol
        simulated_paths[:, p] = np.cumsum(shocks) # Cumulative returns path
        
    # Calculate Risk Metrics at the terminal step
    terminal_returns = simulated_paths[-1, :]
    var_99 = np.percentile(terminal_returns, 1)
    expected_shortfall_99 = terminal_returns[terminal_returns <= var_99].mean()
    
    print("="*50)
    print("        MONTE CARLO TAIL-RISK STRESS AUDIT        ")
    print("="*50)
    print(f"Forward Horizon:         {steps} Trading Days")
    print(f"99% Value-at-Risk (VaR):  {var_99*100:.3f}%")
    print(f"99% Expected Shortfall:   {expected_shortfall_99*100:.3f}%")
    print("="*50)
    
    # Plotting the Risk Horizon Distribution
    plt.figure(figsize=(12, 6), dpi=300)
    plt.hist(terminal_returns, bins=100, color='darkslateblue', alpha=0.7, edgecolor='black', density=True)
    plt.axvline(var_99, color='crimson', linestyle='--', linewidth=2, label=f'99% VaR Threshold ({var_99*100:.2f}%)')
    plt.axvline(expected_shortfall_99, color='darkorange', linestyle='-', linewidth=2, label=f'99% Expected Shortfall ({expected_shortfall_99*100:.2f}%)')
    
    plt.title(f'30-Day Forward Return Projection Distribution (GARCH-LSTM Scaled Shock Engine)', fontsize=12, fontweight='bold')
    plt.xlabel('Cumulative Simulated Portfolio Returns')
    plt.ylabel('Probability Density')
    plt.legend(loc='upper left', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('tail_risk_stress_test.png', dpi=300)
    print("Stress test complete. Asset 'tail_risk_stress_test.png' successfully written to asset pipeline.")

if __name__ == "__main__":
    run_monte_carlo_stress_test()
