"""
LSTM Neural Network predictor for stock price sequences
Uses TensorFlow/Keras for deep learning time series prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional, Tuple
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from indicators.technical_indicators import TechnicalIndicators

class LSTMPredictor:
    """LSTM-based stock price predictor"""

    def __init__(self,
                 sequence_length: int = 60,
                 features_to_use: List[str] = None,
                 lstm_units: int = 100,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM predictor

        Args:
            sequence_length: Number of days to look back for prediction
            features_to_use: List of features to include in training
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Default features for LSTM
        self.features_to_use = features_to_use or [
            'Close', 'Volume', 'High', 'Low',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Returns', 'Volume_SMA', 'ATR'
        ]

        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.is_trained = False

        # Model save paths
        self.model_dir = Path('models/saved_models/lstm')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / 'lstm_model.h5'
        self.scaler_path = self.model_dir / 'scaler.pkl'
        self.config_path = self.model_dir / 'config.json'

    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])

        return np.array(X), np.array(y)

    def prepare_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for LSTM training"""
        df = data.copy()

        # Add technical indicators
        indicators = TechnicalIndicators()

        # Price-based indicators
        df['RSI'] = indicators.rsi(df['Close'])
        macd_data = indicators.macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']
        df['MACD_Hist'] = macd_data['histogram']

        # Bollinger Bands
        bb_data = indicators.bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Lower'] = bb_data['lower']
        df['BB_Middle'] = bb_data['middle']

        # Other indicators
        df['ATR'] = indicators.atr(df['High'], df['Low'], df['Close'])
        df['Volume_SMA'] = indicators.sma(df['Volume'], 20)

        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_to_High'] = df['Close'] / df['High']
        df['Close_to_Low'] = df['Close'] / df['Low']

        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Forward fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model architecture"""
        model = keras.Sequential([
            # First LSTM layer
            layers.LSTM(self.lstm_units,
                       return_sequences=True,
                       input_shape=input_shape),
            layers.Dropout(self.dropout_rate),

            # Second LSTM layer
            layers.LSTM(self.lstm_units // 2,
                       return_sequences=True),
            layers.Dropout(self.dropout_rate),

            # Third LSTM layer
            layers.LSTM(self.lstm_units // 4),
            layers.Dropout(self.dropout_rate),

            # Dense layers
            layers.Dense(50, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(25, activation='relu'),

            # Output layer (predicting next day's price)
            layers.Dense(1)
        ])

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )

        return model

    def train(self, symbols: List[str],
              lookback_days: int = 1000,
              validation_split: float = 0.2,
              epochs: int = 50,
              batch_size: int = 32) -> Dict:
        """Train LSTM model on multiple symbols"""

        print(f"Training LSTM model on {len(symbols)} symbols...")

        all_X, all_y = [], []

        for symbol in symbols:
            print(f"Processing {symbol}...")

            # Fetch data
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if len(data) < self.sequence_length + 50:
                    print(f"Insufficient data for {symbol}, skipping...")
                    continue

                # Prepare features
                df = self.prepare_features(symbol, data)

                # Select features
                feature_data = df[self.features_to_use].values
                target_data = df['Close'].values

                # Scale features
                if symbol == symbols[0]:  # Fit scaler on first symbol
                    scaled_features = self.scaler.fit_transform(feature_data)
                else:
                    scaled_features = self.scaler.transform(feature_data)

                # Create sequences
                X, y = self.create_sequences(scaled_features, target_data)

                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    print(f"  Added {len(X)} sequences from {symbol}")

            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue

        if not all_X:
            raise ValueError("No valid data found for training")

        # Combine all data
        X_combined = np.vstack(all_X)
        y_combined = np.hstack(all_y)

        print(f"\nTotal training sequences: {len(X_combined)}")
        print(f"Sequence shape: {X_combined.shape}")

        # Build model
        self.model = self.build_model((self.sequence_length, len(self.features_to_use)))

        # Train model
        print("Training LSTM model...")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]

        history = self.model.fit(
            X_combined, y_combined,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )

        self.is_trained = True

        # Save model and components
        self.save_model()

        # Calculate final metrics
        y_pred = self.model.predict(X_combined)
        mse = mean_squared_error(y_combined, y_pred)
        mae = mean_absolute_error(y_combined, y_pred)

        training_results = {
            'training_samples': len(X_combined),
            'features_used': self.features_to_use,
            'sequence_length': self.sequence_length,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'mse': mse,
            'mae': mae,
            'training_history': history.history
        }

        print(f"Training completed!")
        print(f"Final Loss: {training_results['final_loss']:.6f}")
        print(f"Final Val Loss: {training_results['final_val_loss']:.6f}")
        print(f"MSE: {mse:.2f}")
        print(f"MAE: {mae:.2f}")

        return training_results

    def predict_price(self, symbol: str, days_ahead: int = 1) -> Optional[Dict]:
        """Predict future price for a symbol"""

        if not self.is_trained and not self.load_model():
            print("No trained model available")
            return None

        try:
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.sequence_length + 100)

            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)

            if len(data) < self.sequence_length:
                print(f"Insufficient recent data for {symbol}")
                return None

            # Prepare features
            df = self.prepare_features(symbol, data)

            # Get recent sequence
            feature_data = df[self.features_to_use].values[-self.sequence_length:]
            scaled_features = self.scaler.transform(feature_data)

            # Reshape for prediction
            X = scaled_features.reshape(1, self.sequence_length, len(self.features_to_use))

            # Make prediction
            predicted_price = self.model.predict(X, verbose=0)[0][0]
            current_price = df['Close'].iloc[-1]

            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price

            # Generate action based on expected return
            if expected_return > 0.02:  # 2% threshold
                action = 'BUY'
                confidence = min(abs(expected_return) * 10, 1.0)
            elif expected_return < -0.02:
                action = 'SELL'
                confidence = min(abs(expected_return) * 10, 1.0)
            else:
                action = 'HOLD'
                confidence = 1.0 - abs(expected_return) * 5

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return': expected_return,
                'action': action,
                'confidence': confidence,
                'prediction_date': datetime.now().isoformat(),
                'model_type': 'LSTM'
            }

        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            return None

    def predict_multiple(self, symbols: List[str]) -> List[Dict]:
        """Predict for multiple symbols"""
        predictions = []

        for symbol in symbols:
            pred = self.predict_price(symbol)
            if pred:
                predictions.append(pred)

        return predictions

    def save_model(self):
        """Save trained model and components"""
        if self.model:
            self.model.save(self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            config = {
                'sequence_length': self.sequence_length,
                'features_to_use': self.features_to_use,
                'lstm_units': self.lstm_units,
                'dropout_rate': self.dropout_rate,
                'learning_rate': self.learning_rate,
                'trained_date': datetime.now().isoformat()
            }

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load saved model and components"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = keras.models.load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                        self.sequence_length = config['sequence_length']
                        self.features_to_use = config['features_to_use']
                        self.lstm_units = config['lstm_units']
                        self.dropout_rate = config['dropout_rate']
                        self.learning_rate = config['learning_rate']

                self.is_trained = True
                print(f"Model loaded from {self.model_path}")
                return True
            else:
                print("No saved model found")
                return False

        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    """Example usage"""
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

    # Initialize predictor
    predictor = LSTMPredictor(
        sequence_length=60,
        lstm_units=100,
        learning_rate=0.001
    )

    # Train model
    print("Training LSTM model...")
    results = predictor.train(symbols, lookback_days=1000, epochs=20)

    # Make predictions
    print("\nMaking predictions...")
    predictions = predictor.predict_multiple(symbols[:3])

    for pred in predictions:
        print(f"{pred['symbol']}: {pred['action']} (confidence: {pred['confidence']:.3f}, "
              f"expected_return: {pred['expected_return']:.3f})")

if __name__ == "__main__":
    main()