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
* **Legacy Vulnerability:** Data preprocessing via `MinMaxScaler` was applied globally across the dataset
