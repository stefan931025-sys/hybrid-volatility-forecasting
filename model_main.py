import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# 1. DATA ACQUISITION (Using Nasdaq as per your CV)
data = yf.download('^IXIC', start='2015-01-01', end='2024-01-01')
returns = 100 * data['Close'].pct_change().dropna()

# 2. GARCH(1,1) MODELING
model_garch = arch_model(returns, vol='Garch', p=1, q=1)
res_garch = model_garch.fit(disp='off')
garch_vol = res_garch.conditional_volatility

# 3. LSTM DATA PREPARATION
# We use the GARCH residuals as input for the LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_vol = scaler.fit_transform(garch_vol.values.reshape(-1, 1))

X, y = [], []
lookback = 20 # 20-day window
for i in range(lookback, len(scaled_vol)):
    X.append(scaled_vol[i-lookback:i, 0])
    y.append(scaled_vol[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# 4. HYBRID LSTM ARCHITECTURE
model_lstm = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(X, y, epochs=10, batch_size=32, verbose=0)

# 5. PREDICTION & EXPORT
predictions = model_lstm.predict(X)
predicted_vol = scaler.inverse_transform(predictions)

# Save results for the Kupiec Test we built earlier
output = pd.DataFrame({
    'Actual_Returns': returns.values[lookback:],
    'Predicted_Vol': predicted_vol.flatten()
})
output.to_csv('backtest_results.csv', index=False)
print("Project Reconstructed: backtest_results.csv generated.")
