"""
Enhanced prediction strategies with individual timeframes and exit rules
Each strategy has its own optimal holding period and exit conditions
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for all prediction strategies with flexible timeframes"""
    
    def __init__(self, name, min_hold_days=1, max_hold_days=30, target_profit=0.08, stop_loss=0.05):
        self.name = name
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.target_profit = target_profit
        self.stop_loss = stop_loss
        self.predictions = []
        
    @abstractmethod
    def predict(self, features):
        """Make a prediction based on features"""
        pass
    
    @abstractmethod
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        """Determine if position should be closed"""
        pass
    
    def get_position(self, score):
        """Convert score to LONG/SHORT position"""
        if score > 0.1:
            return "LONG"
        elif score < -0.1:
            return "SHORT"
        else:
            return "HOLD"
    
    def get_confidence(self, score):
        """Convert score to confidence level"""
        return min(abs(score), 1.0)
    
    def get_timeframe_text(self):
        """Get timeframe description"""
        if self.min_hold_days == self.max_hold_days:
            return f"{self.min_hold_days}D"
        else:
            return f"{self.min_hold_days}-{self.max_hold_days}D"


class MomentumStrategy(BaseStrategy):
    """Momentum - Rides trends for 3-10 days"""
    
    def __init__(self):
        super().__init__("Momentum", min_hold_days=3, max_hold_days=10, target_profit=0.08, stop_loss=0.05)
    
    def predict(self, features):
        score = 0
        
        # Strong recent performance continues
        if features.get('sma_5', 0) > features.get('sma_20', 0):
            score += 0.3
        
        if features.get('price_to_sma20', 1) > 1.02:
            score += 0.2
        if features.get('price_to_sma50', 1) > 1.05:
            score += 0.2
            
        if features.get('volume_ratio', 1) > 1.2:
            score += 0.1
        
        # Inverse for negative momentum
        if features.get('returns', 0) < -0.02:
            score = -score
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score,
            'timeframe': self.get_timeframe_text()
        }
    
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        pnl = (current_price - entry_price) / entry_price
        
        # Minimum hold period
        if days_held < self.min_hold_days:
            return False, ""
            
        # Maximum hold period
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit if momentum reverses
        if features.get('returns', 0) < -0.02:
            return True, "Momentum reversed"
            
        # Profit target
        if pnl > self.target_profit:
            return True, f"Profit target {self.target_profit:.1%}"
            
        # Stop loss
        if pnl < -self.stop_loss:
            return True, f"Stop loss {self.stop_loss:.1%}"
            
        return False, ""


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion - Plays oversold/overbought for 1-5 days"""
    
    def __init__(self):
        super().__init__("Mean Reversion", min_hold_days=1, max_hold_days=5, target_profit=0.06, stop_loss=0.04)
    
    def predict(self, features):
        score = 0
        
        # RSI oversold/overbought
        rsi = features.get('rsi', 50)
        if rsi < 30:
            score += 0.4
        elif rsi > 70:
            score -= 0.4
        
        # Price deviation from moving average
        price_to_sma = features.get('price_to_sma20', 1)
        if price_to_sma < 0.95:
            score += 0.3
        elif price_to_sma > 1.05:
            score -= 0.3
        
        # Bollinger band position
        bb_pos = features.get('bb_position', 0.5)
        if bb_pos < 0.2:
            score += 0.3
        elif bb_pos > 0.8:
            score -= 0.3
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score,
            'timeframe': self.get_timeframe_text()
        }
    
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        pnl = (current_price - entry_price) / entry_price
        
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit when RSI returns to neutral
        rsi = features.get('rsi', 50)
        if 45 < rsi < 55:
            return True, "RSI normalized"
            
        if pnl > self.target_profit:
            return True, f"Profit target {self.target_profit:.1%}"
            
        if pnl < -self.stop_loss:
            return True, f"Stop loss {self.stop_loss:.1%}"
            
        return False, ""


class VolumeBreakoutStrategy(BaseStrategy):
    """Volume Breakout - Catches volume spikes for 1-3 days"""
    
    def __init__(self):
        super().__init__("Volume Breakout", min_hold_days=1, max_hold_days=3, target_profit=0.10, stop_loss=0.06)
    
    def predict(self, features):
        score = 0
        volume_ratio = features.get('volume_ratio', 1)
        
        # Extreme volume spike
        if volume_ratio > 2.0:
            if features.get('returns', 0) > 0.01:
                score += 0.6  # Bullish breakout
            else:
                score -= 0.4  # Selling pressure
        
        # High volume with price action
        if volume_ratio > 1.5:
            if features.get('close_to_high', 0) > 0.98:
                score += 0.3
            elif features.get('close_to_low', 0) < 1.02:
                score -= 0.3
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score,
            'timeframe': self.get_timeframe_text()
        }
    
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        pnl = (current_price - entry_price) / entry_price
        
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit when volume normalizes
        if features.get('volume_ratio', 1) < 0.8:
            return True, "Volume normalized"
            
        if pnl > self.target_profit:
            return True, f"Profit target {self.target_profit:.1%}"
            
        if pnl < -self.stop_loss:
            return True, f"Stop loss {self.stop_loss:.1%}"
            
        return False, ""


class TechnicalIndicatorStrategy(BaseStrategy):
    """Technical Indicators - MACD/RSI signals for 5-15 days"""
    
    def __init__(self):
        super().__init__("Technical Indicators", min_hold_days=5, max_hold_days=15, target_profit=0.12, stop_loss=0.07)
    
    def predict(self, features):
        score = 0
        
        # MACD signal
        if features.get('macd_diff', 0) > 0:
            score += 0.3
        else:
            score -= 0.3
        
        # RSI levels
        rsi = features.get('rsi', 50)
        if 40 < rsi < 60:
            score += 0.1
        elif rsi < 40:
            score += 0.2
        elif rsi > 60:
            score -= 0.2
        
        # Bollinger squeeze
        bb_width = (features.get('bb_upper', 0) - features.get('bb_lower', 0)) / features.get('bb_middle', 1)
        if bb_width < 0.04:
            if features.get('returns', 0) > 0:
                score += 0.3
            else:
                score -= 0.3
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score,
            'timeframe': self.get_timeframe_text()
        }
    
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        pnl = (current_price - entry_price) / entry_price
        
        if days_held < self.min_hold_days:
            return False, ""
            
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit on MACD reversal
        if features.get('macd_diff', 0) < -0.1:
            return True, "MACD reversal"
            
        if pnl > self.target_profit:
            return True, f"Profit target {self.target_profit:.1%}"
            
        if pnl < -self.stop_loss:
            return True, f"Stop loss {self.stop_loss:.1%}"
            
        return False, ""


class SwingTradingStrategy(BaseStrategy):
    """Swing Trading - Longer holds for 10-30 days"""
    
    def __init__(self):
        super().__init__("Swing Trading", min_hold_days=10, max_hold_days=30, target_profit=0.20, stop_loss=0.10)
    
    def predict(self, features):
        score = 0
        
        # Longer-term trend following
        if features.get('sma_20', 0) > features.get('sma_50', 0):
            score += 0.4
        
        # Weekly momentum
        if features.get('price_to_sma50', 1) > 1.1:
            score += 0.3
        elif features.get('price_to_sma50', 1) < 0.9:
            score -= 0.3
        
        # Volume trend
        if features.get('volume_ratio', 1) > 1.1:
            score += 0.2
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score,
            'timeframe': self.get_timeframe_text()
        }
    
    def get_exit_condition(self, entry_price, current_price, days_held, features):
        pnl = (current_price - entry_price) / entry_price
        
        if days_held < self.min_hold_days:
            return False, ""
            
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit on trend reversal
        if features.get('sma_20', 0) < features.get('sma_50', 0):
            return True, "Trend reversal"
            
        if pnl > self.target_profit:
            return True, f"Profit target {self.target_profit:.1%}"
            
        if pnl < -self.stop_loss:
            return True, f"Stop loss {self.stop_loss:.1%}"
            
        return False, ""


def get_all_flexible_strategies():
    """Return instances of all flexible strategies"""
    return [
        MomentumStrategy(),
        MeanReversionStrategy(),
        VolumeBreakoutStrategy(),
        TechnicalIndicatorStrategy(),
        SwingTradingStrategy()
    ]