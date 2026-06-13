<div align="center">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 460" width="100%" style="background:#0d1117; border: 1px solid #30363d; border-radius: 8px; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">
    <!-- Title and Institutional Branding -->
    <text x="25" y="35" fill="#f0f6fc" font-size="16" font-weight="700" letter-spacing="0.5">GARCH-LSTM HYBRID VOLATILITY FORECAST ENGINE</text>
    <text x="25" y="55" fill="#8b949e" font-size="12">Out-of-Sample Parametric Risk &amp; Value-at-Risk (VaR) Backtest Suite</text>
    <rect x="710" y="23" width="115" height="22" rx="4" fill="#21262d" stroke="#30363d"/>
    <text x="722" y="37" fill="#58a6ff" font-size="10" font-weight="600" letter-spacing="1">PRODUCTION</text>

    <!-- Legend -->
    <g transform="translate(480, 75)">
      <line x1="0" y1="10" x2="25" y2="10" stroke="#8b949e" stroke-width="1.5"/>
      <text x="32" y="14" fill="#c9d1d9" font-size="11">Asset Log Returns</text>
      
      <line x1="150" y1="10" x2="175" y2="10" stroke="#f85149" stroke-width="2" stroke-dasharray="4,3"/>
      <text x="182" y="14" fill="#c9d1d9" font-size="11">95% VaR Threshold</text>
      
      <circle cx="315" cy="10" r="4.5" fill="#f85149" stroke="#ffffff" stroke-width="1"/>
      <text x="325" y="14" fill="#f85149" font-size="11" font-weight="600">Model Breaches (n=2)</text>
    </g>

    <!-- PANEL 1: VaR BACKTEST (Y: 90 to 280) -->
    <rect x="25" y="90" width="800" height="190" fill="#161b22" rx="6" stroke="#30363d" stroke-width="0.5"/>
    <text x="35" y="112" fill="#bc8cff" font-size="11" font-weight="600">PANEL 1: 95% Dynamic Risk Thresholding</text>
    
    <!-- Y-Axis Grid & Labels Panel 1 -->
    <g stroke="#30363d" stroke-width="1" stroke-dasharray="3,4">
      <line x1="75" y1="135" x2="810" y2="135"/><text x="40" y="139" fill="#8b949e" font-size="10" stroke="none">+2.5%</text>
      <line x1="75" y1="185" x2="810" y2="185"/><text x="40" y="189" fill="#8b949e" font-size="10" stroke="none"> 0.0%</text>
      <line x1="75" y1="235" x2="810" y2="235"/><text x="40" y="239" fill="#8b949e" font-size="10" stroke="none">-2.5%</text>
    </g>

    <!-- Data Render: Returns & VaR Line -->
    <!-- Path representing model variance tracking and returns -->
    <path d="M 75 180 L 110 160 L 145 195 L 180 150 L 215 210 L 250 140 L 285 242 L 320 170 L 355 190 L 390 155 L 425 200 L 460 165 L 495 185 L 530 145 L 565 248 L 600 175 L 635 190 L 670 160 L 705 205 L 740 170 L 775 180 L 810 165" fill="none" stroke="#8b949e" stroke-width="1.2" opacity="0.75"/>
    <path d="M 75 215 Q 110 205 145 220 T 215 225 T 250 205 T 285 235 T 355 215 T 425 220 T 495 210 T 565 240 T 635 220 T 705 230 T 810 215" fill="none" stroke="#f85149" stroke-width="2" stroke-dasharray="5,4"/>
    
    <!-- Breach Points -->
    <circle cx="285" cy="242" r="5" fill="#f85149" stroke="#ffffff" stroke-width="1.5"/>
    <circle cx="565" cy="248" r="5" fill="#f85149" stroke="#ffffff" stroke-width="1.5"/>


    <!-- PANEL 2: REGIME SWITCHING STATE HEATMAP (Y: 300 to 410) -->
    <rect x="25" y="300" width="800" height="110" fill="#161b22" rx="6" stroke="#30363d" stroke-width="0.5"/>
    <text x="35" y="322" fill="#39d353" font-size="11" font-weight="600">PANEL 2: Volatility Regime Classification State Machine</text>

    <!-- Regime Shading Bars -->
    <rect x="75" y="335" width="160" height="50" fill="#2ea44f" opacity="0.15"/>
    <rect x="235" y="335" width="110" height="50" fill="#f85149" opacity="0.15"/>
    <rect x="345" y="335" width="180" height="50" fill="#2ea44f" opacity="0.15"/>
    <rect x="525" y="335" width="90" height="50" fill="#f85149" opacity="0.15"/>
    <rect x="615" y="335" width="195" height="50" fill="#2ea44f" opacity="0.15"/>

    <!-- Inside Content Panel 2 -->
    <g stroke="#30363d" stroke-width="1" stroke-dasharray="2,3">
      <line x1="75" y1="335" x2="810" y2="335"/>
      <line x1="75" y1="385" x2="810" y2="385"/>
    </g>
    <text x="38" y="340" fill="#8b949e" font-size="9">HIGH</text>
    <text x="40" y="388" fill="#8b949e" font-size="9">LOW</text>

    <!-- Regime Label Indicators -->
    <text x="120" y="365" fill="#39d353" font-size="10" font-weight="600" opacity="0.7">LOW VOLATILITY STATE</text>
    <text x="250" y="365" fill="#ff7b72" font-size="10" font-weight="600" opacity="0.9">HIGH VOL REGIME</text>
    <text x="390" y="365" fill="#39d353" font-size="10" font-weight="600" opacity="0.7">LOW VOLATILITY STATE</text>
    <text x="535" y="365" fill="#ff7b72" font-size="10" font-weight="600" opacity="0.9">HIGH VOL</text>
    <text x="670" y="365" fill="#39d353" font-size="10" font-weight="600" opacity="0.7">LOW VOLATILITY STATE</text>

    <!-- Timeline Timeline Scale Label -->
    <text x="75" y="435" fill="#8b949e" font-size="10">t-250 (Inception)</text>
    <text x="420" y="435" fill="#8b949e" font-size="10">t-125 Out-of-Sample Window</text>
    <text x="765" y="435" fill="#8b949e" font-size="10">t-0 (Current)</text>
  </svg>
</div>


# Hybrid Volatility Forecasting 

## 🛠️ Institutional Audit & Quantitative Refactoring

This repository was systematically updated to transition the core forecasting pipeline from a basic machine learning script into an institutional-grade, production-ready risk management framework. 

### 1. Elimination of Look-Ahead Bias (`model_main.py` → `model_v2.py`)
* **Legacy Vulnerability:** The initial iteration calibrated the GARCH(1,1) parameters across the entire historical data sample before generating volatility inputs for the LSTM neural network, inadvertently leaking future variance structures into past data points.
* **Production Refactor:** Implemented a **strict rolling-window calibration loop**. The GARCH model is now re-calibrated dynamically at each time step $t$ using *only* historical information available up to that day, simulating an authentic live trading desk environment.

### 2. Resolution of Data & Scaler Leakage
* **Legacy Vulnerability:** Data preprocessing via `MinMaxScaler` was applied globally across the dataset before isolating the train and test sets, allowing out-of-sample statistical boundaries (minimum and maximum volatility peaks) to influence the training data.
* **Production Refactor:** Enforced absolute data isolation. The scaling parameters are now fitted strictly on the training partition and merely applied as a passive transform to the out-of-sample validation sequence, ensuring zero data leakage.

### 3. Integration of a Parametric Risk Backtesting Engine (`backtest_engine.py`)
To validate the predictive power of the refactored hybrid model, a formal risk management suite was integrated to translate volatility forecasts into actionable metrics:
* Generates daily **95% and 99% Parametric Value at Risk (VaR)** thresholds using dynamic predicted conditional standard deviations ($\hat{\sigma}_{t+1}$).
* Establishes the mathematical baseline for regulatory backtesting (such as the Kupiec Proportion of Failures coverage test) to analyze the frequency and independence of VaR exceptions during market stress.


## 🧠 Model Architecture & Deep-Learning Pipeline

To effectively capture both structural volatility clustering and non-linear regime shifts, this repository implements a two-stage hybrid time-series pipeline.

### 1. Parametric Baseline Estimation (GARCH)
The engine first fits a classical, mean-reverting GARCH(1,1) process to historical asset log returns. The conditional variance is modeled parametrically to capture long-term baseline volatility clustering and standard heteroskedasticity.

### 2. Multi-Dimensional Feature Engineering
Once the baseline parameters are calculated, the engine isolates the conditional variance and standardized residuals. These vectors are transformed using a rolling look-back window and shaped into multi-dimensional arrays matching deep recurrent sequence demands: `[Samples, Time_Steps, Features]`.

### 3. Non-Linear Residual Tracking (LSTM)
The engineered tensors are passed directly into a Long Short-Term Memory (LSTM) network architecture. While the GARCH baseline masters long-term variance reversion, the LSTM isolates latent, non-linear dependencies and structural regime shifts within the residual structures, providing an advanced toolkit optimized for risk mitigation and tail-hedging parameters.


## 4. Production Orchestration & Stress-Testing Framework

To operationalize the GARCH-LSTM forecasting pipeline for an institutional environment, the architecture was refactored into a modular, production-grade risk management suite.

### Core Architecture Components:
* **`run_pipeline.py` (Master Orchestrator):** Automates the end-to-end execution loop. It dynamic-links the data validation layer, tracks processing latency down to the millisecond, manages memory allocations, and isolates sequential failure points.
* **`stress_test_simulation.py` (Fat-Tail Monte Carlo Engine):** Simulates 10,000 forward-looking return paths across a 30-day trading horizon. Rather than relying on restrictive standard normal distribution assumptions, the engine applies a heavy-tailed **Student's t-distribution** to scale random price shocks by the GARCH-LSTM conditional volatility forecast.
* **Risk Metrics Evaluated:** Calculates a 99% parametric Value-at-Risk (VaR) alongside an integration-based 99% Expected Shortfall (ES) to accurately map tail-risk exposure during systemic regime shifts.

```bash
# To execute the full institutional production suite:
python run_pipeline.py
