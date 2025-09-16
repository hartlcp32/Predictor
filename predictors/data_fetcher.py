import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class StockDataFetcher:
    def __init__(self, lookback_days=100):
        self.lookback_days = lookback_days
        # Top 10 most liquid stocks
        self.stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                      'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
        
    def fetch_data(self, symbol, period='3mo'):
        """Fetch historical data for a single stock"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            if len(data) > 0:
                return data
            else:
                print(f"No data available for {symbol}")
                return None
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            return None
    
    def fetch_all_stocks(self):
        """Fetch data for all stocks in the list"""
        all_data = {}
        for symbol in self.stocks:
            print(f"Fetching data for {symbol}...")
            data = self.fetch_data(symbol)
            if data is not None:
                all_data[symbol] = data
        return all_data
    
    def prepare_features(self, data):
        """Prepare technical features from price data"""
        df = data.copy()
        
        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['dollar_volume'] = df['Close'] * df['Volume']
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Price ratios
        df['price_to_sma20'] = df['Close'] / df['sma_20']
        df['price_to_sma50'] = df['Close'] / df['sma_50']
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_to_high'] = df['Close'] / df['High']
        df['close_to_low'] = df['Close'] / df['Low']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_diff'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df.dropna()
    
    def get_latest_features(self, stock_symbols=None):
        """Get the latest features for all stocks"""
        features_dict = {}

        # Use custom symbols if provided, otherwise use default
        if stock_symbols is not None:
            # Temporarily override the stock list
            original_stocks = self.stocks
            self.stocks = stock_symbols
            stock_data = self.fetch_all_stocks()
            self.stocks = original_stocks  # Restore original
        else:
            stock_data = self.fetch_all_stocks()
        
        for symbol, data in stock_data.items():
            features = self.prepare_features(data)
            if len(features) > 0:
                # Get the last row of features
                latest = features.iloc[-1].to_dict()
                latest['symbol'] = symbol
                latest['date'] = features.index[-1].strftime('%Y-%m-%d')
                features_dict[symbol] = latest
        
        return features_dict
    
    def save_historical_data(self, output_dir='data'):
        """Save historical data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        stock_data = self.fetch_all_stocks()
        timestamp = datetime.now().strftime('%Y%m%d')
        
        for symbol, data in stock_data.items():
            # Save raw data
            data.to_csv(f"{output_dir}/{symbol}_{timestamp}.csv")
            
            # Save features
            features = self.prepare_features(data)
            features.to_csv(f"{output_dir}/{symbol}_features_{timestamp}.csv")
        
        print(f"Data saved to {output_dir}/")
        
        # Save metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'stocks': self.stocks,
            'lookback_days': self.lookback_days,
            'files_created': [f"{symbol}_{timestamp}.csv" for symbol in stock_data.keys()]
        }
        
        with open(f"{output_dir}/metadata_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    fetcher = StockDataFetcher()
    print("Fetching latest stock data...")
    features = fetcher.get_latest_features()
    print(f"Fetched data for {len(features)} stocks")
    
    # Save for testing
    with open('latest_features.json', 'w') as f:
        json.dump(features, f, indent=2, default=str)