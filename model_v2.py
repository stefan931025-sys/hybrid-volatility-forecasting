from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.preprocessing import RobustScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

logger = logging.getLogger("HybridVolatilityModel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class HybridVolatilityModel:
    def __init__(self, lookback_window: int = 60, p: int = 1, q: int = 1):
        self.lookback_window = lookback_window
        self.p = p
        self.q = q
        self.scaler = RobustScaler()
        self.model: Sequential | None = None

    def compute_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Fits GARCH(p,q) model and extracts conditional volatility securely."""
        logger.info("Fitting GARCH model to extract conditional volatility proxy...")
        # Scale returns to percentage for numerical stability in GARCH optimization
        scaled_returns = returns * 100.0
        am = arch_model(scaled_returns, vol='Garch', p=self.p, q=self.q, rescale=False)
        res = am.fit(disp='off')
        conditional_vol = res.conditional_volatility / 100.0  # Scale back
        return pd.Series(conditional_vol, index=returns.index)

    def prepare_sequences(self, features: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates rolling chronological sequence windows without look-ahead bias."""
        X, y = [], []
        for i in range(self.lookback_window, len(features)):
            X.append(features[i - self.lookback_window:i])
            y.append(target[i])
        return np.array(X), np.array(y)

    def build_lstm_architecture(self, input_shape: tuple[int, int]) -> Sequential:
        """Constructs a regularized LSTM architecture for volatility forecasting."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Linear activation for continuous volatility prediction
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='huber', metrics=['mae'])
        return model

    def train_and_forecast(self, df: pd.DataFrame, target_col: str, feature_cols: list[str], split_idx: int):
        """Executes walk-forward training and out-of-sample forecasting."""
        logger.info("Initiating walk-forward data pipeline...")

        # Split data chronologically to prevent look-ahead bias
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx - self.lookback_window:]  # Include lookback buffer for test set

        # Fit scaler strictly on training features
        train_features = train_df[feature_cols].values
        train_scaled = self.scaler.fit_transform(train_features)
        
        # Transform test features using training statistics
        test_features = test_df[feature_cols].values
        test_scaled = self.scaler.transform(test_features)

        train_target = train_df[target_col].values
        test_target = test_df[target_col].values

        # Build LSTM tensors
        X_train, y_train = self.prepare_sequences(train_scaled, train_target)
        X_test, y_test = self.prepare_sequences(test_scaled, test_target)

        logger.info(f"Training Tensor Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}")
        logger.info(f"Testing Tensor Shapes  -> X_test: {X_test.shape}, y_test: {y_test.shape}")

        # Compile and train model
        self.model = self.build_lstm_architecture((X_train.shape[1], X_train.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=25,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )

        predictions = self.model.predict(X_test)
        return history, predictions
