"""
Sequence-based predictor using scikit-learn
Alternative to LSTM when TensorFlow is not available
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
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

class SequencePredictor:
    """Sequence-based stock predictor using traditional ML"""

    def __init__(self,
                 sequence_length: int = 30,
                 model_type: str = 'gradient_boosting',
                 features_to_use: List[str] = None):
        """
        Initialize sequence predictor

        Args:
            sequence_length: Number of days to look back
            model_type: 'random_forest' or 'gradient_boosting'
            features_to_use: List of features to include
        """
        self.sequence_length = sequence_length
        self.model_type = model_type

        # Default features
        self.features_to_use = features_to_use or [
            'Close', 'Volume', 'High', 'Low',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
            'Returns', 'Volume_Ratio', 'ATR'
        ]

        # Initialize model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:  # gradient_boosting
            self.model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

        self.scaler = StandardScaler()
        self.is_trained = False

        # Model save paths
        self.model_dir = Path('models/saved_models/sequence')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / f'{model_type}_sequence_model.pkl'
        self.scaler_path = self.model_dir / f'{model_type}_scaler.pkl'
        self.config_path = self.model_dir / f'{model_type}_config.json'

    def create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences by flattening time windows"""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            # Flatten the sequence into a single feature vector
            sequence = data[i-self.sequence_length:i].flatten()
            X.append(sequence)
            y.append(target[i])

        return np.array(X), np.array(y)

    def prepare_features(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        df = data.copy()

        # Add technical indicators
        indicators = TechnicalIndicators()

        # Price-based indicators
        df['RSI'] = indicators.rsi(df['Close'])
        macd_data = indicators.macd(df['Close'])
        df['MACD'] = macd_data['macd']
        df['MACD_Signal'] = macd_data['signal']

        # Bollinger Bands
        bb_data = indicators.bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_data['upper']
        df['BB_Lower'] = bb_data['lower']

        # Other indicators
        df['ATR'] = indicators.atr(df['High'], df['Low'], df['Close'])

        # Price features
        df['Returns'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']

        # Volume features
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'Close_Lag{lag}'] = df['Close'].shift(lag)
            df[f'Volume_Lag{lag}'] = df['Volume'].shift(lag)
            df[f'Returns_Lag{lag}'] = df['Returns'].shift(lag)

        # Moving averages
        for window in [5, 10, 20]:
            df[f'SMA_{window}'] = df['Close'].rolling(window).mean()
            df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window).mean()

        # Forward fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')

        return df

    def train(self, symbols: List[str],
              lookback_days: int = 1000,
              test_size: float = 0.2) -> Dict:
        """Train sequence model on multiple symbols"""

        print(f"Training {self.model_type} sequence model on {len(symbols)} symbols...")

        all_X, all_y = [], []

        for symbol in symbols:
            print(f"Processing {symbol}...")

            try:
                # Fetch data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if len(data) < self.sequence_length + 50:
                    print(f"Insufficient data for {symbol}, skipping...")
                    continue

                # Prepare features
                df = self.prepare_features(symbol, data)

                # Select available features (handle missing columns)
                available_features = [f for f in self.features_to_use if f in df.columns]
                if len(available_features) < len(self.features_to_use):
                    missing = set(self.features_to_use) - set(available_features)
                    print(f"  Missing features for {symbol}: {missing}")

                if not available_features:
                    print(f"  No valid features for {symbol}, skipping...")
                    continue

                # Use available features
                feature_data = df[available_features].values
                target_data = df['Close'].values

                # Create sequences
                X, y = self.create_sequences(feature_data, target_data)

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
        print(f"Feature vector size: {X_combined.shape[1]}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.model, X_scaled, y_combined, cv=tscv, scoring='neg_mean_squared_error')

        print(f"Cross-validation MSE scores: {-cv_scores}")
        print(f"Mean CV MSE: {-cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Train on full dataset
        self.model.fit(X_scaled, y_combined)
        self.is_trained = True

        # Calculate final metrics
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y_combined, y_pred)
        mae = mean_absolute_error(y_combined, y_pred)

        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            # Create feature names for flattened sequences
            feature_names = []
            for t in range(self.sequence_length):
                for feat in available_features:
                    feature_names.append(f"{feat}_t-{self.sequence_length-t-1}")

            importance_data = list(zip(feature_names, self.model.feature_importances_))
            feature_importance = sorted(importance_data, key=lambda x: x[1], reverse=True)

        # Save model
        self.save_model()

        training_results = {
            'training_samples': len(X_combined),
            'features_used': available_features,
            'sequence_length': self.sequence_length,
            'cv_mse_mean': -cv_scores.mean(),
            'cv_mse_std': cv_scores.std(),
            'final_mse': mse,
            'final_mae': mae,
            'feature_importance': feature_importance[:20] if feature_importance else None,
            'model_type': self.model_type
        }

        print(f"\nTraining completed!")
        print(f"Final MSE: {mse:.4f}")
        print(f"Final MAE: {mae:.4f}")

        return training_results

    def predict_price(self, symbol: str) -> Optional[Dict]:
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

            # Select available features
            available_features = [f for f in self.features_to_use if f in df.columns]
            if not available_features:
                print(f"No valid features available for {symbol}")
                return None

            # Get recent sequence
            feature_data = df[available_features].values[-self.sequence_length:]
            sequence = feature_data.flatten().reshape(1, -1)
            scaled_sequence = self.scaler.transform(sequence)

            # Make prediction
            predicted_price = self.model.predict(scaled_sequence)[0]
            current_price = df['Close'].iloc[-1]

            # Calculate expected return
            expected_return = (predicted_price - current_price) / current_price

            # Generate action based on expected return
            if expected_return > 0.02:  # 2% threshold
                action = 'BUY'
                confidence = min(abs(expected_return) * 8, 1.0)
            elif expected_return < -0.02:
                action = 'SELL'
                confidence = min(abs(expected_return) * 8, 1.0)
            else:
                action = 'HOLD'
                confidence = 1.0 - abs(expected_return) * 10

            return {
                'symbol': symbol,
                'current_price': current_price,
                'predicted_price': predicted_price,
                'expected_return': expected_return,
                'action': action,
                'confidence': confidence,
                'prediction_date': datetime.now().isoformat(),
                'model_type': self.model_type
            }

        except Exception as e:
            print(f"Error predicting {symbol}: {e}")
            return None

    def save_model(self):
        """Save trained model"""
        if self.model and self.is_trained:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)

            config = {
                'sequence_length': self.sequence_length,
                'model_type': self.model_type,
                'features_to_use': self.features_to_use,
                'trained_date': datetime.now().isoformat()
            }

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            print(f"Model saved to {self.model_path}")

    def load_model(self) -> bool:
        """Load saved model"""
        try:
            if self.model_path.exists() and self.scaler_path.exists():
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)

                if self.config_path.exists():
                    with open(self.config_path, 'r') as f:
                        config = json.load(f)
                        self.sequence_length = config['sequence_length']
                        self.model_type = config['model_type']
                        self.features_to_use = config['features_to_use']

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
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']

    # Test both model types
    for model_type in ['random_forest', 'gradient_boosting']:
        print(f"\n=== Testing {model_type.upper()} Sequence Predictor ===")

        predictor = SequencePredictor(
            sequence_length=20,
            model_type=model_type
        )

        # Train model
        results = predictor.train(symbols, lookback_days=500)

        # Make predictions
        print(f"\nPredictions using {model_type}:")
        for symbol in symbols[:3]:
            pred = predictor.predict_price(symbol)
            if pred:
                print(f"{pred['symbol']}: {pred['action']} "
                      f"(confidence: {pred['confidence']:.3f}, "
                      f"expected_return: {pred['expected_return']:.3f})")

if __name__ == "__main__":
    main()