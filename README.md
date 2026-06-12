

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
