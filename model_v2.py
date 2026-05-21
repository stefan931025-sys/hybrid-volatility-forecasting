import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# 1. DATA ACQUISITION
data = yf.download('^IXIC', start='2015-01-01', end='2024-01-01')
returns = 100 * data['Close'].pct_change().dropna()

returns_val = returns.values
n_points = len(returns_val)
split_idx = int(n_points * 0.70)
lookback = 20

# 2. ROLLING GARCH CALIBRATION (No Look-Ahead Bias)
cond_vol = np.zeros(n_points)
initial_window = 500  

print("Running rolling GARCH calibration...")
for t in range(initial_window, n_points):
    train_window = returns_val[:t]
    garch = arch_model(train_window, vol='Garch', p=1, q=1)
    res = garch.fit(update_freq=0, disp='off')
    
    forecast = res.forecast(horizon=1)
    cond_vol[t] = np.sqrt(forecast.variance.values[-1, 0])

valid_idx = initial_window
features_garch = cond_vol[valid_idx:]
actual_returns = returns_val[valid_idx:]

# 3. TRAIN/TEST SPLIT FOR THE LSTM
realized_vol_target = np.abs(actual_returns)

X_raw, y_raw = [], []
for i in range(lookback, len(features_garch) - 1):
    X_raw.append(features_garch[i-lookback:i])
    y_raw.append(realized_vol_target[i+1]) 

X_raw, y_raw = np.array(X_raw), np.array(y_raw)

train_size = split_idx - initial_window
X_train_raw, X_test_raw = X_raw[:train_size], X_raw[train_size:]
y_train_raw, y_test_raw = y_raw[:train_size], y_raw[train_size:]

# 4. DATA SCALING (No Scaler Leakage)
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test_raw.reshape(-1, 1))

X_train = np.reshape(X_train_scaled, (X_train_scaled.shape[0], lookback, 1))
X_test = np.reshape(X_test_scaled, (X_test_scaled.shape[0], lookback, 1))

# 5. HYBRID LSTM ARCHITECTURE
model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X_train, y_train_scaled, epochs=10, batch_size=32, verbose=1)

# 6. PURE OUT-OF-SAMPLE PREDICTION
predictions_scaled = model_lstm.predict(X_test)
predicted_vol = scaler_y.inverse_transform(predictions_scaled).flatten()
actual_vol_test = y_test_raw

output = pd.DataFrame({
    'Actual_Realized_Vol': actual_vol_test,
    'Predicted_Vol': predicted_vol
})
output.to_csv('out_of_sample_results.csv', index=False)
print("Institutional audit corrections complete. True out-of-sample validation generated.")
