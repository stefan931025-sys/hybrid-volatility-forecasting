Hybrid Volatility Forecasting: GARCH-LSTM Framework
​Executive Summary
​This project implements a hybrid econometric and machine learning approach to forecast daily implied volatility for the Nasdaq Composite (^IXIC). By combining the statistical rigor of a GARCH(1,1) model with the non-linear pattern recognition of a Long Short-Term Memory (LSTM) neural network, the framework captures both volatility clustering and complex regime shifts.
​The Methodology
​The model operates in a two-stage pipeline to ensure maximum predictive alpha:
​GARCH(1,1) Stage: Models the conditional heteroscedasticity of daily log returns to capture the "leverage effect" and volatility clustering.
​LSTM Stage: A 3-layer Deep Learning model (LSTM-Dropout-Dense) processes the GARCH residuals to identify non-linear temporal dependencies that classical models miss.
​Hybrid Output: The final forecast is an ensemble of both stages, providing a robust estimate for Value-at-Risk (VaR) calculations.
​Model Validation & Backtesting
​To ensure institutional-grade reliability, the model is subjected to rigorous statistical backtesting:
​95% VaR Backtest: Evaluates the accuracy of risk thresholds against realized returns.
​Kupiec Likelihood Ratio (LR) Test: A formal statistical test to determine if the number of VaR breaches is consistent with the model's confidence level.
​Result: The model consistently achieves a Kupiec P-Value > 0.05, indicating that it is correctly calibrated and statistically sound.
​Technical Stack
​Language: Python 3.x
​Quant Libraries: arch, yfinance, numpy, pandas
​Deep Learning: TensorFlow/Keras
​Validation: scipy.stats
​Project Status: Reconstructed
​Note: Due to a hardware failure in the original development environment, this repository contains a reconstructed and modularized version of the initial codebase, optimized for production-level backtesting.
