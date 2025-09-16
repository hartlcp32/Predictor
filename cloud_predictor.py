#!/usr/bin/env python
"""
Lightweight predictor for GitHub Actions
Generates predictions without database dependency
"""

import json
import random
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# List of stocks to analyze (top volume stocks)
STOCKS = ['AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMD', 'SPY', 'QQQ', 'AMZN', 'META', 'GOOGL']

# Strategies
STRATEGIES = [
    'momentum',
    'mean_reversion', 
    'volume_breakout',
    'technical_indicators',
    'pattern_recognition',
    'volatility_arbitrage',
    'moving_average_crossover',
    'support_resistance',
    'market_sentiment',
    'ensemble'
]

def fetch_stock_data(symbol):
    """Fetch recent stock data"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period='1mo')
        
        if hist.empty:
            return None
            
        # Calculate basic indicators
        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['SMA_5'] = hist['Close'].rolling(5).mean()
        hist['Volume_Avg'] = hist['Volume'].rolling(20).mean()
        
        latest = hist.iloc[-1]
        prev = hist.iloc[-2] if len(hist) > 1 else latest
        
        # Calculate features
        features = {
            'price': latest['Close'],
            'volume': latest['Volume'],
            'volume_ratio': latest['Volume'] / hist['Volume_Avg'].iloc[-1] if hist['Volume_Avg'].iloc[-1] > 0 else 1,
            'price_change': (latest['Close'] - prev['Close']) / prev['Close'] if prev['Close'] > 0 else 0,
            'sma_ratio': latest['Close'] / latest['SMA_20'] if latest['SMA_20'] > 0 else 1,
            'volatility': hist['Close'].pct_change().std(),
            'rsi': calculate_rsi(hist['Close'])
        }
        
        return features
        
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if not rsi.empty else 50

def generate_prediction(features, strategy):
    """Generate prediction based on features and strategy"""
    
    score = 0
    
    if strategy == 'momentum':
        if features['price_change'] > 0.02:
            score += 0.6
        if features['volume_ratio'] > 1.5:
            score += 0.3
            
    elif strategy == 'mean_reversion':
        if features['rsi'] < 30:
            score += 0.7
        elif features['rsi'] > 70:
            score -= 0.7
            
    elif strategy == 'volume_breakout':
        if features['volume_ratio'] > 2.0:
            score += 0.8
            
    elif strategy == 'technical_indicators':
        if features['sma_ratio'] > 1.02:
            score += 0.5
        elif features['sma_ratio'] < 0.98:
            score -= 0.5
            
    else:
        # Random for other strategies (simplified)
        score = random.uniform(-0.5, 0.5)
    
    # Determine position
    if score > 0.3:
        position = 'LONG'
    elif score < -0.3:
        position = 'SHORT'
    else:
        position = 'HOLD'
        
    # Calculate confidence
    confidence = min(abs(score), 1.0) * 0.7 + 0.3
    
    # Calculate projected return
    projected = score * 10  # Simplified projection
    
    return {
        'position': position,
        'confidence': confidence,
        'projected': f"{projected:+.1f}%",
        'score': score
    }

def main():
    """Generate predictions and save to JSON"""
    
    print(f"Generating predictions for {datetime.now().strftime('%Y-%m-%d')}")
    
    # Fetch data for all stocks
    stock_features = {}
    for symbol in STOCKS:
        print(f"Fetching {symbol}...")
        features = fetch_stock_data(symbol)
        if features:
            stock_features[symbol] = features
    
    # Generate predictions for each strategy
    predictions = {}
    used_stocks = set()
    
    for strategy in STRATEGIES:
        # Find best stock for this strategy
        best_stock = None
        best_score = -999
        
        for symbol, features in stock_features.items():
            if symbol not in used_stocks:
                pred = generate_prediction(features, strategy)
                
                if abs(pred['score']) > abs(best_score):
                    best_score = pred['score']
                    best_stock = symbol
                    best_prediction = pred
        
        if best_stock:
            predictions[strategy] = {
                'stock': best_stock,
                'position': best_prediction['position'],
                'confidence': best_prediction['confidence'],
                'projected': best_prediction['projected'],
                'score': best_prediction['score']
            }
            used_stocks.add(best_stock)
        else:
            # Fallback if no stock available
            predictions[strategy] = {
                'stock': 'SPY',
                'position': 'HOLD',
                'confidence': 0.5,
                'projected': '+0.0%',
                'score': 0
            }
    
    # Save predictions
    output = {
        'predictions': [
            {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'predictions': predictions
            }
        ],
        'last_updated': datetime.now().isoformat()
    }
    
    # Keep last 30 days of predictions
    try:
        with open('predictions_data.json', 'r') as f:
            existing = json.load(f)
            if 'predictions' in existing:
                output['predictions'].extend(existing['predictions'][:29])
    except:
        pass
    
    with open('predictions_data.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Predictions saved to predictions_data.json")
    
    # Also update trades file (empty for now)
    trades = {
        'active_trades': [],
        'closed_trades': [],
        'last_updated': datetime.now().isoformat()
    }
    
    with open('trades_data.json', 'w') as f:
        json.dump(trades, f, indent=2)
    
    print("Cloud prediction complete!")

if __name__ == "__main__":
    main()