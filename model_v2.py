import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def fit_rolling_garch(returns, initial_window=500):
    """
    Fits a rolling-window GARCH(1,1) model on log returns to extract conditional
    volatility variance while strictly enforcing zero look-ahead bias.
    """
    returns_val = returns.values
    n_points = len(returns_val)
    cond_vol = np.zeros(n_points)

    print("Running rolling GARCH calibration...")
    # Roll window forward 1 step at a time to prevent future data leakage
    for t in range(initial_window, n_points):
        train_window = returns_val[:t]
        garch = arch_model(train_window, vol='Garch', p=1, q=1)
        res = garch.fit(update_freq=0, disp='off')
        forecast = res.forecast(horizon=1)
        cond_vol[t] = np.sqrt(forecast.variance.values[-1, 0])

    return pd.Series(cond_vol, index=returns.index)


def train_lstm_residuals(returns, garch_vol, lookback=20):
    """
    Trains a deep recurrent LSTM neural network on GARCH conditional volatility
    feature vectors to capture non-linear residual structures.
    """
    returns_val = returns.values
    cond_vol_val = garch_vol.values
    valid_idx = 500  # Boundary for rolling initial window
    
    features_garch = cond_vol_val[valid_idx:]
    actual_returns = returns_val[valid_idx:]
    realized_vol_target = np.abs(actual_returns)

    X_raw, y_raw = [], []
    for i in range(lookback, len(features_garch) - 1):
        X_raw.append(features_garch[i - lookback:i])
        y_raw.append(realized_vol_target[i + 1])

    X_raw, y_raw = np.array(X_raw), np.array(y_raw)
    
    # Train / Test split (70/30)
    split_idx = int(len(X_raw) * 0.7)
    X_train_raw, X_test_raw = X_raw[:split_idx], X_raw[split_idx:]
    y_train_raw, y_test_raw = y_raw[:split_idx], y_raw[split_idx:]

    # Enforce strict data scaling with zero scaler leakage
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler_X.fit_transform(X_train_raw)
    X_test_scaled = scaler_X.transform(X_test_raw)

    y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))

    X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], lookback, 1))
    X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], lookback, 1))

    # Construct Hybrid Deep LSTM Architecture
    model_lstm = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])

    model_lstm.compile(optimizer='adam', loss='mean_squared_error')
    model_lstm.fit(X_train, y_train_scaled, epochs=10, batch_size=32, verbose=0)

    # Generate Out-of-Sample Predictions
    predictions_scaled = model_lstm.predict(X_test)
    predicted_vol = scaler_y.inverse_transform(predictions_scaled).flatten()

    # Align predictions with original returns series length
    full_predictions = np.zeros(len(returns))
    full_predictions[-len(predicted_vol):] = predicted_vol
    return full_predictions
