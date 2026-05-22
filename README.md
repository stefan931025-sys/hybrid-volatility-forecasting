![GARCH-LSTM Dashboard](garch_lstm_backtest.png)

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
