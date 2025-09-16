"""
Random Forest Machine Learning Predictor
Uses technical indicators to predict stock price direction
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from indicators.technical_indicators import TechnicalIndicators, create_features_dataframe

class RandomForestPredictor:
    """Random Forest-based stock prediction model"""
    
    def __init__(self, model_type: str = 'classification'):
        """Initialize the predictor
        
        Args:
            model_type: 'classification' for direction prediction, 'regression' for return prediction
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = {}
        self.model_path = Path('models/saved_models')
        self.model_path.mkdir(exist_ok=True)
        
        # Model parameters
        if model_type == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from OHLCV data"""
        
        # Create features using technical indicators
        features_df = create_features_dataframe(df)
        
        # Select features for ML
        feature_columns = [
            # Price-based features
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26', 'EMA_50',
            'Price_to_SMA20', 'Price_to_SMA50',
            'SMA5_to_SMA20', 'SMA20_to_SMA50',
            
            # Momentum features
            'RSI', 'RSI_30',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'Stoch_K', 'Stoch_D',
            'Williams_R', 'CCI',
            
            # Volatility features
            'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_PercentB',
            'ATR',
            
            # Trend features
            'ADX', 'Plus_DI', 'Minus_DI',
            'SuperTrend', 'ST_Direction',
            
            # Volume features
            'Volume_Ratio', 'MFI', 'OBV',
            
            # Price relationships
            'High_Low_Ratio', 'Close_to_High', 'Close_to_Low',
            
            # Returns and volatility
            'Returns', 'Returns_5d', 'Volatility',
            
            # Lag features
            'Close_Lag1', 'Close_Lag2',
            'Volume_Lag1', 'RSI_Lag1', 'MACD_Lag1'
        ]
        
        # Filter features that exist in the dataframe
        available_features = [col for col in feature_columns if col in features_df.columns]
        
        # Extract features and target
        X = features_df[available_features].copy()
        
        if self.model_type == 'classification':
            y = features_df['Target_Direction'].copy()
        else:
            y = features_df['Target_Return'].copy()
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train(self, symbols: List[str], lookback_days: int = 1000) -> Dict[str, float]:
        """Train the model on multiple symbols"""
        
        print(f"Training Random Forest {self.model_type} model...")
        print(f"Symbols: {symbols}")
        
        all_X = []
        all_y = []
        
        # Collect data from multiple symbols
        for symbol in symbols:
            print(f"Processing {symbol}...")
            
            try:
                # Download data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)
                
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if len(data) < 200:  # Need minimum data for indicators
                    print(f"  Insufficient data for {symbol}")
                    continue
                
                # Prepare features
                X, y = self.prepare_features(data)
                
                if len(X) > 50:  # Minimum samples
                    all_X.append(X)
                    all_y.append(y)
                    print(f"  Added {len(X)} samples from {symbol}")
                
            except Exception as e:
                print(f"  Error processing {symbol}: {e}")
                continue
        
        if not all_X:
            raise ValueError("No valid data found for training")
        
        # Combine all data
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        print(f"\nTotal training samples: {len(X_combined)}")
        print(f"Features: {len(X_combined.columns)}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_combined, cv=tscv, scoring='accuracy' if self.model_type == 'classification' else 'neg_mean_squared_error')
        
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Train final model
        self.model.fit(X_scaled, y_combined)
        
        # Feature importance
        self.feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 most important features:")
        for feature, importance in sorted_importance[:10]:
            print(f"  {feature}: {importance:.4f}")
        
        # Test set performance
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_combined, test_size=0.2, random_state=42, shuffle=False
        )
        
        test_predictions = self.model.predict(X_test)
        
        if self.model_type == 'classification':
            test_accuracy = accuracy_score(y_test, test_predictions)
            print(f"\nTest accuracy: {test_accuracy:.4f}")
            
            # Direction accuracy for returns > 1%
            significant_moves = np.abs(y_test) > 0.01
            if significant_moves.sum() > 0:
                sig_accuracy = accuracy_score(y_test[significant_moves], test_predictions[significant_moves])
                print(f"Accuracy on moves >1%: {sig_accuracy:.4f}")
            
            return {
                'cv_score': cv_scores.mean(),
                'test_accuracy': test_accuracy,
                'samples': len(X_combined),
                'features': len(self.feature_names)
            }
        else:
            test_mse = mean_squared_error(y_test, test_predictions)
            test_rmse = np.sqrt(test_mse)
            print(f"\nTest RMSE: {test_rmse:.6f}")
            
            return {
                'cv_score': -cv_scores.mean(),
                'test_rmse': test_rmse,
                'samples': len(X_combined),
                'features': len(self.feature_names)
            }
    
    def predict(self, data: pd.DataFrame) -> Dict[str, float]:
        """Make prediction on new data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X, _ = self.prepare_features(data)
        
        if len(X) == 0:
            return {'prediction': 0.5, 'confidence': 0.0}
        
        # Use the last row for prediction
        latest_features = X.iloc[-1:]
        
        # Scale features
        X_scaled = self.scaler.transform(latest_features)
        
        if self.model_type == 'classification':
            # Get probability predictions
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Assuming binary classification (0=down, 1=up)
            prediction = probabilities[1] if len(probabilities) > 1 else 0.5
            confidence = max(probabilities) - 0.5  # Distance from random
            
            return {
                'prediction': prediction,
                'confidence': confidence * 2,  # Scale to 0-1
                'direction': 'UP' if prediction > 0.5 else 'DOWN',
                'probability_up': prediction,
                'probability_down': 1 - prediction
            }
        else:
            # Regression prediction
            predicted_return = self.model.predict(X_scaled)[0]
            
            # Confidence based on prediction magnitude
            confidence = min(abs(predicted_return) * 10, 1.0)
            
            return {
                'predicted_return': predicted_return,
                'confidence': confidence,
                'direction': 'UP' if predicted_return > 0 else 'DOWN',
                'target_price': data['Close'].iloc[-1] * (1 + predicted_return)
            }
    
    def predict_symbol(self, symbol: str, days: int = 100) -> Dict[str, float]:
        """Predict for a specific symbol"""
        
        try:
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if len(data) < 50:
                return {'error': f'Insufficient data for {symbol}'}
            
            return self.predict(data)
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_model(self, filename: str = None):
        """Save trained model to disk"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'random_forest_{self.model_type}_{timestamp}.pkl'
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'trained_date': datetime.now().isoformat()
        }
        
        filepath = self.model_path / filename
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
        return str(filepath)
    
    def load_model(self, filepath: str):
        """Load trained model from disk"""
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        
        print(f"Model loaded from {filepath}")
        print(f"Trained: {model_data.get('trained_date', 'Unknown')}")
        print(f"Features: {len(self.feature_names)}")
    
    def backtest(self, symbol: str, start_date: str, end_date: str) -> Dict[str, float]:
        """Backtest the model on historical data"""
        
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        print(f"Backtesting {symbol} from {start_date} to {end_date}")
        
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if len(data) < 200:
            return {'error': 'Insufficient data for backtesting'}
        
        # Prepare features
        X, y_true = self.prepare_features(data)
        
        if len(X) < 50:
            return {'error': 'Insufficient samples for backtesting'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.model_type == 'classification':
            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, predictions)
            
            # Strategy returns (assuming we trade based on predictions)
            returns = X['Returns'].values[1:]  # Align with predictions
            strategy_returns = np.where(predictions[:-1] == 1, returns, -returns)
            
            total_return = np.sum(strategy_returns)
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252) if np.std(strategy_returns) > 0 else 0
            
            return {
                'accuracy': accuracy,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'trades': len(predictions),
                'avg_confidence': np.mean(probabilities)
            }
        
        else:
            predictions = self.model.predict(X_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, predictions)
            rmse = np.sqrt(mse)
            
            # Correlation between predicted and actual returns
            correlation = np.corrcoef(predictions, y_true)[0, 1]
            
            return {
                'rmse': rmse,
                'correlation': correlation,
                'samples': len(predictions)
            }


def train_ensemble_model():
    """Train a Random Forest model on multiple stock symbols"""
    
    # Top volume stocks for training
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'AMD', 'NFLX', 'CRM']
    
    # Train classification model
    print("Training classification model...")
    classifier = RandomForestPredictor('classification')
    clf_results = classifier.train(symbols, lookback_days=2000)
    
    # Save model
    clf_path = classifier.save_model('rf_classifier_production.pkl')
    
    # Train regression model
    print("\nTraining regression model...")
    regressor = RandomForestPredictor('regression')
    reg_results = regressor.train(symbols, lookback_days=2000)
    
    # Save model
    reg_path = regressor.save_model('rf_regressor_production.pkl')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Classification model: {clf_path}")
    print(f"  Accuracy: {clf_results.get('test_accuracy', 0):.4f}")
    print(f"  Samples: {clf_results.get('samples', 0)}")
    
    print(f"\nRegression model: {reg_path}")
    print(f"  RMSE: {reg_results.get('test_rmse', 0):.6f}")
    print(f"  Samples: {reg_results.get('samples', 0)}")
    
    return classifier, regressor


def test_predictions():
    """Test predictions on recent data"""
    
    # Load models
    classifier = RandomForestPredictor('classification')
    regressor = RandomForestPredictor('regression')
    
    try:
        classifier.load_model('models/saved_models/rf_classifier_production.pkl')
        regressor.load_model('models/saved_models/rf_regressor_production.pkl')
    except FileNotFoundError:
        print("Models not found. Run train_ensemble_model() first.")
        return
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'NVDA']
    
    print("\n" + "="*60)
    print("ML PREDICTIONS")
    print("="*60)
    
    for symbol in test_symbols:
        print(f"\n{symbol}:")
        
        # Classification prediction
        clf_pred = classifier.predict_symbol(symbol)
        print(f"  Direction: {clf_pred.get('direction', 'UNKNOWN')}")
        print(f"  Probability: {clf_pred.get('probability_up', 0):.3f}")
        print(f"  Confidence: {clf_pred.get('confidence', 0):.3f}")
        
        # Regression prediction
        reg_pred = regressor.predict_symbol(symbol)
        print(f"  Expected Return: {reg_pred.get('predicted_return', 0)*100:.2f}%")
        print(f"  Target Price: ${reg_pred.get('target_price', 0):.2f}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Forest Stock Predictor')
    parser.add_argument('--train', action='store_true', help='Train new models')
    parser.add_argument('--test', action='store_true', help='Test predictions')
    parser.add_argument('--backtest', type=str, help='Backtest on symbol')
    
    args = parser.parse_args()
    
    if args.train:
        train_ensemble_model()
    elif args.test:
        test_predictions()
    elif args.backtest:
        # Load classifier
        classifier = RandomForestPredictor('classification')
        try:
            classifier.load_model('models/saved_models/rf_classifier_production.pkl')
            
            # Backtest
            results = classifier.backtest(
                args.backtest, 
                (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
            
            print(f"\nBacktest results for {args.backtest}:")
            for key, value in results.items():
                print(f"  {key}: {value}")
        
        except FileNotFoundError:
            print("Model not found. Run with --train first.")
    
    else:
        print("Use --train to train models, --test to test predictions, or --backtest SYMBOL")


if __name__ == "__main__":
    main()