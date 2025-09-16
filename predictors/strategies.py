import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Base class for all prediction strategies"""
    
    def __init__(self, name, min_hold_days=1, max_hold_days=30):
        self.name = name
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
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
    """Strategy 1: Classic price momentum - Hold 3-10 days"""
    
    def __init__(self):
        super().__init__("Momentum", min_hold_days=3, max_hold_days=10)
    
    def predict(self, features):
        # Strong recent performance continues
        score = 0
        
        if 'returns' in features:
            # 5-day momentum
            if features.get('sma_5', 0) > features.get('sma_20', 0):
                score += 0.3
            
            # Price above moving averages
            if features.get('price_to_sma20', 1) > 1.02:
                score += 0.2
            if features.get('price_to_sma50', 1) > 1.05:
                score += 0.2
                
            # Volume confirmation
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
        """Exit momentum trade when trend weakens or time limit hit"""
        if days_held >= self.max_hold_days:
            return True, "Max time reached"
        
        # Exit if momentum reverses
        if features.get('returns', 0) < -0.02:  # 2% drop
            return True, "Momentum reversed"
            
        # Take profit at 8%
        pnl = (current_price - entry_price) / entry_price
        if pnl > 0.08:
            return True, "Profit target hit"
            
        # Stop loss at -5%
        if pnl < -0.05:
            return True, "Stop loss"
            
        return False, ""


class MeanReversionStrategy(BaseStrategy):
    """Strategy 2: Betting on return to average"""

    def __init__(self):
        super().__init__("Mean Reversion")

    def get_exit_condition(self, position, features):
        """Exit when price returns to mean"""
        if position == 'LONG':
            # Exit long when RSI normalizes or price above SMA
            if features.get('rsi', 50) > 50 or features.get('price_to_sma20', 1) > 1.0:
                return True
        elif position == 'SHORT':
            # Exit short when RSI normalizes or price below SMA
            if features.get('rsi', 50) < 50 or features.get('price_to_sma20', 1) < 1.0:
                return True
        return False

    def predict(self, features):
        score = 0

        # Oversold conditions
        if features.get('rsi', 50) < 30:
            score += 0.4
        elif features.get('rsi', 50) > 70:
            score -= 0.4

        # Price deviation from moving average
        price_to_sma = features.get('price_to_sma20', 1)
        if price_to_sma < 0.95:  # 5% below MA
            score += 0.3
        elif price_to_sma > 1.05:  # 5% above MA
            score -= 0.3

        # Bollinger band position
        bb_pos = features.get('bb_position', 0.5)
        if bb_pos < 0.2:  # Near lower band
            score += 0.3
        elif bb_pos > 0.8:  # Near upper band
            score -= 0.3

        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class VolumeBreakoutStrategy(BaseStrategy):
    """Strategy 3: High volume anomaly detection"""
    
    def __init__(self):
        super().__init__("Volume Breakout")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        volume_ratio = features.get('volume_ratio', 1)
        
        # Extreme volume spike
        if volume_ratio > 2.0:
            # Check price action
            if features.get('returns', 0) > 0.01:
                score += 0.6  # Bullish breakout
            else:
                score -= 0.4  # Potential selling pressure
        
        # High volume with price at resistance
        if volume_ratio > 1.5:
            if features.get('close_to_high', 0) > 0.98:
                score += 0.3
            elif features.get('close_to_low', 0) < 1.02:
                score -= 0.3
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class TechnicalIndicatorStrategy(BaseStrategy):
    """Strategy 4: RSI, MACD, Bollinger Bands"""
    
    def __init__(self):
        super().__init__("Technical Indicators")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

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
            score += 0.1  # Neutral zone, slight bullish
        elif rsi < 40:
            score += 0.2  # Oversold
        elif rsi > 60:
            score -= 0.2  # Overbought
        
        # Bollinger squeeze
        bb_width = (features.get('bb_upper', 0) - features.get('bb_lower', 0)) / features.get('bb_middle', 1)
        if bb_width < 0.04:  # Tight bands, expect breakout
            if features.get('returns', 0) > 0:
                score += 0.3
            else:
                score -= 0.3
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class PatternRecognitionStrategy(BaseStrategy):
    """Strategy 5: Head & shoulders, triangles, flags"""
    
    def __init__(self):
        super().__init__("Pattern Recognition")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        # Simplified pattern detection based on price levels
        high_low_ratio = features.get('high_low_ratio', 1)
        
        # Consolidation pattern (triangle)
        if high_low_ratio < 1.01:  # Very tight range
            if features.get('volume_ratio', 1) < 0.8:  # Decreasing volume
                score += 0.4  # Breakout imminent
        
        # Flag pattern (trend continuation)
        if features.get('volatility', 0) < 0.01:  # Low volatility
            if features.get('price_to_sma20', 1) > 1.02:
                score += 0.3  # Bullish flag
            elif features.get('price_to_sma20', 1) < 0.98:
                score -= 0.3  # Bearish flag
        
        # Support/resistance break
        if features.get('close_to_high', 0) > 0.995:
            score += 0.3  # Breaking resistance
        elif features.get('close_to_low', 0) < 1.005:
            score -= 0.3  # Breaking support
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class VolatilityArbitrageStrategy(BaseStrategy):
    """Strategy 6: VIX-based predictions"""
    
    def __init__(self):
        super().__init__("Volatility Arbitrage")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        volatility = features.get('volatility', 0.02)
        
        # High volatility mean reversion
        if volatility > 0.03:
            # Expect reversal
            if features.get('returns', 0) < -0.02:
                score += 0.5  # Oversold in high vol
            else:
                score -= 0.3  # Overbought in high vol
        
        # Low volatility trend following
        elif volatility < 0.01:
            if features.get('returns', 0) > 0:
                score += 0.3  # Trend continuation
            else:
                score -= 0.3
        
        # Volatility expansion setup
        if volatility < 0.008:  # Extremely low vol
            score += 0.2  # Expect expansion (usually upward)
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class MovingAverageCrossoverStrategy(BaseStrategy):
    """Strategy 7: SMA/EMA crosses"""
    
    def __init__(self):
        super().__init__("Moving Average Crossover")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        # Golden cross / Death cross
        sma_5 = features.get('sma_5', 0)
        sma_20 = features.get('sma_20', 0)
        sma_50 = features.get('sma_50', 0)
        
        if sma_5 > sma_20:
            score += 0.3
            if sma_20 > sma_50:
                score += 0.3  # Strong uptrend
        elif sma_5 < sma_20:
            score -= 0.3
            if sma_20 < sma_50:
                score -= 0.3  # Strong downtrend
        
        # EMA crossover
        ema_12 = features.get('ema_12', 0)
        ema_26 = features.get('ema_26', 0)
        
        if ema_12 > ema_26:
            score += 0.2
        else:
            score -= 0.2
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class SupportResistanceStrategy(BaseStrategy):
    """Strategy 8: Price level analysis"""
    
    def __init__(self):
        super().__init__("Support/Resistance")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        # Price near high/low
        close_to_high = features.get('close_to_high', 0.5)
        close_to_low = features.get('close_to_low', 1.5)
        
        # Resistance test
        if close_to_high > 0.98:
            if features.get('volume_ratio', 1) > 1.3:
                score += 0.4  # Breaking resistance with volume
            else:
                score -= 0.3  # Likely rejection
        
        # Support test
        if close_to_low < 1.02:
            if features.get('volume_ratio', 1) > 1.3:
                score -= 0.4  # Breaking support with volume
            else:
                score += 0.3  # Likely bounce
        
        # Mid-range positioning
        bb_position = features.get('bb_position', 0.5)
        if 0.4 < bb_position < 0.6:
            # Neutral zone, look at momentum
            if features.get('macd_diff', 0) > 0:
                score += 0.2
            else:
                score -= 0.2
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class MarketSentimentStrategy(BaseStrategy):
    """Strategy 9: Volume-weighted momentum"""
    
    def __init__(self):
        super().__init__("Market Sentiment")
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        score = 0
        
        # Volume-weighted price action
        volume_ratio = features.get('volume_ratio', 1)
        returns = features.get('returns', 0)
        
        # Strong volume with price movement
        sentiment_score = returns * volume_ratio
        
        if sentiment_score > 0.02:
            score += 0.5  # Strong bullish sentiment
        elif sentiment_score < -0.02:
            score -= 0.5  # Strong bearish sentiment
        
        # Dollar volume trend
        dollar_volume = features.get('dollar_volume', 0)
        if dollar_volume > 0:  # Simplified - would need historical comparison
            if returns > 0:
                score += 0.2
            else:
                score -= 0.1  # Selling pressure
        
        # Accumulation/Distribution proxy
        close_position = (features.get('Close', 0) - features.get('Low', 0)) / (features.get('High', 1) - features.get('Low', 0))
        if close_position > 0.7:
            score += 0.2  # Accumulation
        elif close_position < 0.3:
            score -= 0.2  # Distribution
        
        return {
            'position': self.get_position(score),
            'confidence': self.get_confidence(score),
            'score': score
        }


class EnsembleStrategy(BaseStrategy):
    """Strategy 10: Weighted combination of all strategies"""
    
    def __init__(self):
        super().__init__("Ensemble")
        self.strategies = [
            MomentumStrategy(),
            MeanReversionStrategy(),
            VolumeBreakoutStrategy(),
            TechnicalIndicatorStrategy(),
            PatternRecognitionStrategy(),
            VolatilityArbitrageStrategy(),
            MovingAverageCrossoverStrategy(),
            SupportResistanceStrategy(),
            MarketSentimentStrategy()
        ]
        # Weights based on hypothetical backtesting performance
        self.weights = [0.15, 0.10, 0.08, 0.13, 0.07, 0.09, 0.12, 0.11, 0.15]
    
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits

    def predict(self, features):
        weighted_score = 0
        predictions = []
        
        for strategy, weight in zip(self.strategies, self.weights):
            pred = strategy.predict(features)
            weighted_score += pred['score'] * weight
            predictions.append(pred)
        
        # Consensus check
        long_count = sum(1 for p in predictions if p['position'] == 'LONG')
        short_count = sum(1 for p in predictions if p['position'] == 'SHORT')
        
        # Adjust confidence based on consensus
        consensus_factor = max(long_count, short_count) / len(predictions)
        
        return {
            'position': self.get_position(weighted_score),
            'confidence': min(self.get_confidence(weighted_score) * consensus_factor, 1.0),
            'score': weighted_score,
            'consensus': f"{long_count}L/{short_count}S"
        }


def get_all_strategies():
    """Return instances of all strategies"""
    return [
        MomentumStrategy(),
        MeanReversionStrategy(),
        VolumeBreakoutStrategy(),
        TechnicalIndicatorStrategy(),
        PatternRecognitionStrategy(),
        VolatilityArbitrageStrategy(),
        MovingAverageCrossoverStrategy(),
        SupportResistanceStrategy(),
        MarketSentimentStrategy(),
        EnsembleStrategy()
    ]