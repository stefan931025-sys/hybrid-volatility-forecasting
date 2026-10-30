# Hybrid Volatility Forecasting

[![CI Pipeline](https://github.com/stefan931025-sys/hybrid-volatility-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/stefan931025-sys/hybrid-volatility-forecasting/actions/workflows/ci.yml)

![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📊 Backtest Analytics Dashboard

![Backtest Analytics Dashboard](https://raw.githubusercontent.com/stefan931025-sys/hybrid-volatility-forecasting/main/assets/backtest_chart.png)

## 🛠 Institutional Audit & Quantitative Refactoring

This repository was systematically updated to transition the core forecasting pipeline from a basic machine learning script into an institutional-grade, production-ready risk management framework.

### 1. Elimination of Look-Ahead Bias (`model_main.py` → `model_v2.py`)
* **Legacy Vulnerability:** The initial iteration calibrated the GARCH(1,1) parameters across the entire historical data sample before generating volatility inputs for the LSTM neural network, inadvertently leaking future variance structures into past data points.
* **Production Refactor:** Implemented a strict rolling-window calibration loop. The GARCH model is now re-calibrated dynamically at each time step using only historical information available up to that day, simulating an authentic live trading desk environment.

### 2. Resolution of Data & Scaler Leakage
* **Legacy Vulnerability:** Data preprocessing via `MinMaxScaler` was applied globally across the dataset before isolating the train and test sets, allowing out-of-sample statistical boundaries (minimum and maximum volatility peaks) to influence the training data.
* **Production Refactor:** Enforced absolute data isolation. The scaling parameters are now fitted strictly on the training partition and merely applied as a passive transform to the out-of-sample validation sequence, ensuring zero data leakage.

### 3. Integration of a Parametric Risk Backtesting Engine (`backtest_engine.py`)
To validate the predictive power of the refactored hybrid model, a formal risk management suite was integrated to translate volatility forecasts into actionable metrics:
* Generates daily 95% and 99% Parametric Value at Risk (VaR) thresholds using dynamic predicted conditional standard deviations ($\hat{\sigma}_{t+1}$).
* Establishes the mathematical baseline for regulatory backtesting (such as the Kupiec Proportion of Failures coverage test) to analyze the frequency and independence of VaR exceptions during market stress.

## 🧠 Model Architecture & Deep-Learning Pipeline

1. **GARCH(1,1) Variance Feature Extraction:** Models time-varying conditional heteroskedasticity to extract volatility persistence parameters ($\alpha, \beta$).
2. **LSTM Neural Network Layer:** Learns non-linear, temporal dependency structures from sequence inputs.
3. **Hybrid Signal Fusion:** Combines parametric volatility estimates with neural network predictions to forecast out-of-sample conditional volatility.

## 🚀 Quickstart

```bash
git clone [https://github.com/stefan931025-sys/hybrid-volatility-forecasting.git](https://github.com/stefan931025-sys/hybrid-volatility-forecasting.git)
cd hybrid-volatility-forecasting
python src/model_v2.py
