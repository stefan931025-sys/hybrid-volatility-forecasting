# Hybrid Volatility Forecasting Engine

[![Hybrid Volatility Forecasting CI](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/stefan931025-sys/hybrid-volatility-forecasting)
[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Systematic risk architecture combining GARCH(1,1) parametric volatility models and LSTM neural networks to map out-of-sample parametric Value-at-Risk (VaR) boundaries.

---

## Institutional Audit & Quantitative Refactoring

This repository was systematically updated to transition the core forecasting pipeline from a basic machine learning script into an institutional-grade, production-ready risk management framework.

### 1. Elimination of Look-Ahead Bias (`model_main.py` -> `model_v2.py`)
* **Legacy Vulnerability:** The initial iteration calibrated GARCH(1,1) parameters across the entire historical data sample before generating volatility inputs for the LSTM neural network, inadvertently leaking future variance structures into past data points.
* **Production Refactor:** Implemented a strict rolling-window calibration loop. The GARCH model is now re-calibrated dynamically at each time step using only historical information available up to that day, simulating an authentic live trading desk environment.

### 2. Resolution of Data & Scaler Leakage
* **Legacy Vulnerability:** Data preprocessing via `MinMaxScaler` was applied globally across the dataset before isolating train and test sets, allowing out-of-sample statistical boundaries to influence training data.
* **Production Refactor:** Enforced absolute data isolation. Scaling parameters are fitted strictly on the training partition and merely applied as a passive transform to out-of-sample validation sequences.

### 3. Integration of Parametric Risk Backtesting Engine (`backtest_engine.py`)
To validate the predictive power of the refactored hybrid model, a formal risk management suite was integrated to translate volatility forecasts into actionable metrics:
* **Parametric VaR:** Generates daily 95% and 99% Parametric Value at Risk thresholds using dynamic predicted conditional standard deviations ($\hat{\sigma}_{t+1}$).
* **Regulatory Backtesting:** Establishes the mathematical baseline for regulatory backtesting (Kupiec Proportion of Failures coverage test) to analyze the frequency and independence of VaR exceptions during market stress.

---

## Mathematical Specifications

### 1. Dynamic Rolling GARCH(1,1) Variance
$$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2 \quad \text{subject to } \alpha + \beta < 1$$

### 2. Hybrid LSTM Signal Fusion & Forecast
$$\hat{\sigma}_{t+1} = \Phi_{\text{LSTM}}\left(\sigma_t^2, r_t, r_{t-1}, \dots; \Theta\right)$$

### 3. Parametric Value at Risk (VaR) & Kupiec POF Test
$$\text{VaR}_{\alpha, t} = \mu_t + z_{\alpha} \hat{\sigma}_{t+1}$$
$$\text{LR}_{\text{POF}} = -2 \ln \left[ \frac{(1-p)^{T-N} p^N}{\left(1-\frac{N}{T}\right)^{T-N} \left(\frac{N}{T}\right)^N} \right] \sim \chi^2(1)$$
Where $T$ is the total backtest observations, $N$ is the number of VaR exceptions, and $p$ is the theoretical failure probability ($1 - \text{confidence level}$).

---

## System Architecture & Data Flow

```text
+-----------------------------------------------------------------------------------+
|                        ASSET RETURN TIME SERIES DATA                              |
|                   (Raw Historical Daily Log-Return Arrays)                        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     ROLLING GARCH(1,1) & SCALER ISOLATION LOOP                    |
|  * Rolling-Window Parameter Calibration Loop   * Fit-Transform Training Scalers    |
|  * Strict Zero-Lookahead Variance Features     * Out-of-Sample Scaler Transform   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        HYBRID LSTM NEURAL NETWORK LAYER                           |
|  * Sequence Windowing (Time-Steps)             * Non-Linear Variance Optimization |
|  * Forward Out-of-Sample Volatility Forecast   * Loss Minimization Engine         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     PARAMETRIC VAR & KUPIEC POF BACKTEST DESK                     |
|  * 95% / 99% Parametric VaR Thresholds         * Exception Violation Tracking     |
|  * Kupiec Proportion of Failures Test          * Monte Carlo Stress Simulations   |
+-----------------------------------------------------------------------------------+
