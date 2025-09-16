"""
Comprehensive Technical Indicators Library
Implements 50+ technical indicators for trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    """Collection of technical analysis indicators"""
    
    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        
        return {
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std),
            'bandwidth': (sma + std * num_std - (sma - std * num_std)) / sma * 100,
            'percent_b': (data - (sma - std * num_std)) / (2 * std * num_std)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        
        return -100 * ((highest_high - close) / (highest_high - lowest_low))
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> Dict[str, pd.Series]:
        """Average Directional Index"""
        # Calculate +DI and -DI
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed averages
        atr = true_range.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
        
        # ADX calculation
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=window).mean()
        mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        return (typical_price - sma) / (0.015 * mean_deviation)
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """On-Balance Volume"""
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = volume.iloc[0]
        
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        return (typical_price * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = pd.Series(index=close.index, dtype=float)
        negative_flow = pd.Series(index=close.index, dtype=float)
        
        positive_flow[typical_price > typical_price.shift(1)] = money_flow
        negative_flow[typical_price < typical_price.shift(1)] = money_flow
        
        positive_flow.fillna(0, inplace=True)
        negative_flow.fillna(0, inplace=True)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def ichimoku(high: pd.Series, low: pd.Series, close: pd.Series) -> Dict[str, pd.Series]:
        """Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
        
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
        
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        senkou_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        chikou = close.shift(-26)
        
        return {
            'tenkan': tenkan,
            'kijun': kijun,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
    
    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_window: int = 10, multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """SuperTrend Indicator"""
        hl2 = (high + low) / 2
        atr = TechnicalIndicators.atr(high, low, close, atr_window)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize series
        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)
        
        supertrend.iloc[0] = lower_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(close)):
            if close.iloc[i] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        
        return {
            'supertrend': supertrend,
            'direction': direction
        }
    
    @staticmethod
    def fibonacci_retracement(high: float, low: float) -> Dict[str, float]:
        """Fibonacci Retracement Levels"""
        diff = high - low
        
        return {
            'level_0': high,
            'level_236': high - 0.236 * diff,
            'level_382': high - 0.382 * diff,
            'level_500': high - 0.500 * diff,
            'level_618': high - 0.618 * diff,
            'level_786': high - 0.786 * diff,
            'level_100': low
        }
    
    @staticmethod
    def pivot_points(high: float, low: float, close: float) -> Dict[str, float]:
        """Pivot Points"""
        pivot = (high + low + close) / 3
        
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            'r2': pivot + (high - low),
            'r3': high + 2 * (pivot - low),
            's1': 2 * pivot - high,
            's2': pivot - (high - low),
            's3': low - 2 * (high - pivot)
        }
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators for a DataFrame with OHLCV data"""
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            raise ValueError("DataFrame must contain Open, High, Low, Close, Volume columns")
        
        result = df.copy()
        
        # Price-based indicators
        result['SMA_5'] = TechnicalIndicators.sma(df['Close'], 5)
        result['SMA_10'] = TechnicalIndicators.sma(df['Close'], 10)
        result['SMA_20'] = TechnicalIndicators.sma(df['Close'], 20)
        result['SMA_50'] = TechnicalIndicators.sma(df['Close'], 50)
        result['SMA_200'] = TechnicalIndicators.sma(df['Close'], 200)
        
        result['EMA_12'] = TechnicalIndicators.ema(df['Close'], 12)
        result['EMA_26'] = TechnicalIndicators.ema(df['Close'], 26)
        result['EMA_50'] = TechnicalIndicators.ema(df['Close'], 50)
        
        # Momentum indicators
        result['RSI'] = TechnicalIndicators.rsi(df['Close'])
        result['RSI_30'] = TechnicalIndicators.rsi(df['Close'], 30)
        
        macd_data = TechnicalIndicators.macd(df['Close'])
        result['MACD'] = macd_data['macd']
        result['MACD_Signal'] = macd_data['signal']
        result['MACD_Hist'] = macd_data['histogram']
        
        stoch_data = TechnicalIndicators.stochastic(df['High'], df['Low'], df['Close'])
        result['Stoch_K'] = stoch_data['k_percent']
        result['Stoch_D'] = stoch_data['d_percent']
        
        result['Williams_R'] = TechnicalIndicators.williams_r(df['High'], df['Low'], df['Close'])
        result['CCI'] = TechnicalIndicators.cci(df['High'], df['Low'], df['Close'])
        
        # Volatility indicators
        bb_data = TechnicalIndicators.bollinger_bands(df['Close'])
        result['BB_Upper'] = bb_data['upper']
        result['BB_Middle'] = bb_data['middle']
        result['BB_Lower'] = bb_data['lower']
        result['BB_Width'] = bb_data['bandwidth']
        result['BB_PercentB'] = bb_data['percent_b']
        
        result['ATR'] = TechnicalIndicators.atr(df['High'], df['Low'], df['Close'])
        
        # Trend indicators
        adx_data = TechnicalIndicators.adx(df['High'], df['Low'], df['Close'])
        result['ADX'] = adx_data['adx']
        result['Plus_DI'] = adx_data['plus_di']
        result['Minus_DI'] = adx_data['minus_di']
        
        supertrend_data = TechnicalIndicators.supertrend(df['High'], df['Low'], df['Close'])
        result['SuperTrend'] = supertrend_data['supertrend']
        result['ST_Direction'] = supertrend_data['direction']
        
        # Volume indicators
        result['OBV'] = TechnicalIndicators.obv(df['Close'], df['Volume'])
        result['VWAP'] = TechnicalIndicators.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        result['MFI'] = TechnicalIndicators.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Ichimoku indicators
        ichimoku_data = TechnicalIndicators.ichimoku(df['High'], df['Low'], df['Close'])
        result['Ichimoku_Tenkan'] = ichimoku_data['tenkan']
        result['Ichimoku_Kijun'] = ichimoku_data['kijun']
        result['Ichimoku_SenkouA'] = ichimoku_data['senkou_a']
        result['Ichimoku_SenkouB'] = ichimoku_data['senkou_b']
        result['Ichimoku_Chikou'] = ichimoku_data['chikou']
        
        # Price ratios and relationships
        result['Price_to_SMA20'] = df['Close'] / result['SMA_20']
        result['Price_to_SMA50'] = df['Close'] / result['SMA_50']
        result['SMA5_to_SMA20'] = result['SMA_5'] / result['SMA_20']
        result['SMA20_to_SMA50'] = result['SMA_20'] / result['SMA_50']
        
        # Volume ratios
        result['Volume_SMA20'] = TechnicalIndicators.sma(df['Volume'], 20)
        result['Volume_Ratio'] = df['Volume'] / result['Volume_SMA20']
        
        # Returns and volatility
        result['Returns'] = df['Close'].pct_change()
        result['Returns_5d'] = df['Close'].pct_change(5)
        result['Volatility'] = result['Returns'].rolling(20).std()
        
        # High/Low ratios
        result['High_Low_Ratio'] = df['High'] / df['Low']
        result['Close_to_High'] = df['Close'] / df['High']
        result['Close_to_Low'] = df['Close'] / df['Low']
        
        return result
    
    @staticmethod
    def get_signal_summary(df: pd.DataFrame) -> Dict[str, float]:
        """Get trading signals summary from indicators"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        signals = {}
        
        # Trend signals (0-1 scale, 1 = strong bullish)
        signals['sma_trend'] = 1 if latest['SMA5_to_SMA20'] > 1.01 else 0 if latest['SMA5_to_SMA20'] < 0.99 else 0.5
        signals['price_trend'] = min(max((latest['Price_to_SMA20'] - 0.95) / 0.1, 0), 1)
        
        # Momentum signals
        signals['rsi_signal'] = 0.8 if latest['RSI'] < 30 else 0.2 if latest['RSI'] > 70 else 0.5
        signals['macd_signal'] = 1 if latest['MACD'] > latest['MACD_Signal'] else 0
        signals['stoch_signal'] = 1 if latest['Stoch_K'] > latest['Stoch_D'] and latest['Stoch_K'] < 80 else 0
        
        # Volatility signals
        signals['bb_signal'] = 0.8 if latest['BB_PercentB'] < 0.2 else 0.2 if latest['BB_PercentB'] > 0.8 else 0.5
        signals['atr_signal'] = min(latest['ATR'] / latest['Close'] * 100, 1)  # Normalized volatility
        
        # Volume signals
        signals['volume_signal'] = min(latest['Volume_Ratio'] / 2, 1)
        signals['mfi_signal'] = 0.8 if latest['MFI'] < 20 else 0.2 if latest['MFI'] > 80 else 0.5
        
        # Overall signal strength
        trend_score = (signals['sma_trend'] + signals['price_trend']) / 2
        momentum_score = (signals['rsi_signal'] + signals['macd_signal'] + signals['stoch_signal']) / 3
        volume_score = (signals['volume_signal'] + signals['mfi_signal']) / 2
        
        signals['overall_signal'] = (trend_score * 0.4 + momentum_score * 0.4 + volume_score * 0.2)
        signals['signal_strength'] = abs(signals['overall_signal'] - 0.5) * 2  # 0-1 scale
        
        return signals


# Utility functions for strategy development
def backtest_indicator(df: pd.DataFrame, entry_signal: str, exit_signal: str, 
                      position_type: str = 'LONG') -> Dict[str, float]:
    """Simple backtesting for indicator-based strategies"""
    trades = []
    position = None
    
    for i in range(1, len(df)):
        if position is None:  # No position
            if eval(entry_signal.format(i=i), {'df': df, 'i': i}):
                position = {
                    'entry_price': df.iloc[i]['Close'],
                    'entry_index': i,
                    'type': position_type
                }
        else:  # In position
            if eval(exit_signal.format(i=i), {'df': df, 'i': i, 'entry_i': position['entry_index']}):
                exit_price = df.iloc[i]['Close']
                
                if position['type'] == 'LONG':
                    pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * 100
                else:
                    pnl_pct = (position['entry_price'] - exit_price) / position['entry_price'] * 100
                
                trades.append({
                    'entry': position['entry_price'],
                    'exit': exit_price,
                    'pnl_pct': pnl_pct,
                    'duration': i - position['entry_index']
                })
                
                position = None
    
    if not trades:
        return {'total_trades': 0, 'win_rate': 0, 'avg_return': 0}
    
    wins = len([t for t in trades if t['pnl_pct'] > 0])
    total_return = sum(t['pnl_pct'] for t in trades)
    
    return {
        'total_trades': len(trades),
        'win_rate': wins / len(trades) * 100,
        'avg_return': total_return / len(trades),
        'total_return': total_return,
        'avg_duration': np.mean([t['duration'] for t in trades])
    }


def create_features_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create ML-ready features from OHLCV data"""
    
    # Calculate all indicators
    features_df = TechnicalIndicators.calculate_all_indicators(df)
    
    # Add lag features
    for col in ['Close', 'Volume', 'RSI', 'MACD']:
        if col in features_df.columns:
            features_df[f'{col}_Lag1'] = features_df[col].shift(1)
            features_df[f'{col}_Lag2'] = features_df[col].shift(2)
    
    # Add rolling statistics
    features_df['Close_Rolling_Mean_5'] = features_df['Close'].rolling(5).mean()
    features_df['Close_Rolling_Std_5'] = features_df['Close'].rolling(5).std()
    features_df['Volume_Rolling_Mean_10'] = features_df['Volume'].rolling(10).mean()
    
    # Add target variable (next day return)
    features_df['Target_Return'] = features_df['Close'].shift(-1) / features_df['Close'] - 1
    features_df['Target_Direction'] = (features_df['Target_Return'] > 0).astype(int)
    
    # Remove first 200 rows to ensure all indicators are calculated
    features_df = features_df.iloc[200:].copy()
    
    return features_df