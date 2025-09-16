"""
Comprehensive risk management system for stock trading
Includes position sizing, portfolio risk, drawdown control, and risk metrics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import yfinance as yf
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from indicators.technical_indicators import TechnicalIndicators

@dataclass
class RiskMetrics:
    """Risk metrics for a position or portfolio"""
    value_at_risk_95: float
    value_at_risk_99: float
    expected_shortfall_95: float
    expected_shortfall_99: float
    sharpe_ratio: float
    sortino_ratio: float
    maximum_drawdown: float
    volatility: float
    beta: float
    correlation_spy: float

@dataclass
class PositionRisk:
    """Risk assessment for a single position"""
    symbol: str
    current_price: float
    position_size: int
    position_value: float
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_percent: float
    volatility: float
    beta: float
    correlation_spy: float
    recommended_size: int
    risk_score: float  # 0-100, higher = riskier

class RiskManager:
    """Advanced risk management system"""

    def __init__(self,
                 max_portfolio_risk: float = 0.02,  # 2% max portfolio risk
                 max_position_risk: float = 0.01,   # 1% max per position
                 max_correlation: float = 0.7,      # Max correlation between positions
                 max_sector_allocation: float = 0.3, # Max 30% in one sector
                 volatility_lookback: int = 252):    # Days for volatility calculation
        """
        Initialize risk manager

        Args:
            max_portfolio_risk: Maximum portfolio risk as % of capital
            max_position_risk: Maximum position risk as % of capital
            max_correlation: Maximum correlation between positions
            max_sector_allocation: Maximum allocation to one sector
            volatility_lookback: Days to look back for volatility calculation
        """
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_correlation = max_correlation
        self.max_sector_allocation = max_sector_allocation
        self.volatility_lookback = volatility_lookback

        self.indicators = TechnicalIndicators()

        # Market data cache
        self.price_cache = {}
        self.spy_data = None
        self._load_market_data()

    def _load_market_data(self):
        """Load SPY data for beta and correlation calculations"""
        try:
            spy = yf.Ticker('SPY')
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.volatility_lookback + 50)
            self.spy_data = spy.history(start=start_date, end=end_date)
            print("Market data loaded successfully")
        except Exception as e:
            print(f"Error loading market data: {e}")
            self.spy_data = None

    def calculate_position_size(self,
                              symbol: str,
                              entry_price: float,
                              stop_loss: float,
                              portfolio_value: float,
                              confidence: float = 1.0) -> Dict:
        """
        Calculate optimal position size based on risk parameters

        Args:
            symbol: Stock symbol
            entry_price: Intended entry price
            stop_loss: Stop loss price
            portfolio_value: Current portfolio value
            confidence: Model confidence (0-1)

        Returns:
            Dictionary with position sizing recommendations
        """

        # Get historical data for volatility calculation
        risk_per_share = abs(entry_price - stop_loss)
        risk_percent_per_share = risk_per_share / entry_price

        # Basic position size based on fixed risk
        max_risk_amount = portfolio_value * self.max_position_risk
        basic_size = int(max_risk_amount / risk_per_share)

        # Adjust for volatility
        volatility = self._calculate_volatility(symbol)
        volatility_multiplier = min(1.0, 0.2 / volatility) if volatility > 0 else 1.0

        # Adjust for confidence
        confidence_multiplier = confidence ** 2  # Square to be more conservative

        # Adjust for correlation with existing positions
        correlation_multiplier = 1.0  # Placeholder for now

        # Calculate final position size
        recommended_size = int(basic_size * volatility_multiplier * confidence_multiplier * correlation_multiplier)

        # Ensure minimum and maximum bounds
        min_size = 1
        max_size = int(portfolio_value * 0.1 / entry_price)  # Max 10% of portfolio
        recommended_size = max(min_size, min(recommended_size, max_size))

        return {
            'symbol': symbol,
            'recommended_size': recommended_size,
            'basic_size': basic_size,
            'volatility_multiplier': volatility_multiplier,
            'confidence_multiplier': confidence_multiplier,
            'correlation_multiplier': correlation_multiplier,
            'max_risk_amount': max_risk_amount,
            'actual_risk_amount': recommended_size * risk_per_share,
            'risk_percent': (recommended_size * risk_per_share) / portfolio_value,
            'volatility': volatility
        }

    def assess_position_risk(self,
                           symbol: str,
                           quantity: int,
                           entry_price: float,
                           current_price: float,
                           stop_loss: float,
                           take_profit: Optional[float] = None) -> PositionRisk:
        """Assess risk for an existing position"""

        position_value = quantity * current_price
        risk_amount = quantity * abs(current_price - stop_loss)
        risk_percent = risk_amount / position_value if position_value > 0 else 0

        # Calculate volatility and market metrics
        volatility = self._calculate_volatility(symbol)
        beta = self._calculate_beta(symbol)
        correlation = self._calculate_correlation_spy(symbol)

        # Calculate risk score (0-100)
        risk_score = self._calculate_risk_score(volatility, beta, risk_percent)

        # Recommend position size based on current risk
        recommended_size = self._recommend_position_adjustment(
            symbol, current_price, stop_loss, position_value, volatility
        )

        return PositionRisk(
            symbol=symbol,
            current_price=current_price,
            position_size=quantity,
            position_value=position_value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=risk_amount,
            risk_percent=risk_percent,
            volatility=volatility,
            beta=beta,
            correlation_spy=correlation,
            recommended_size=recommended_size,
            risk_score=risk_score
        )

    def calculate_portfolio_risk(self, positions: List[Dict]) -> Dict:
        """Calculate portfolio-level risk metrics"""

        if not positions:
            return {
                'total_value': 0,
                'total_risk': 0,
                'risk_percent': 0,
                'diversification_ratio': 1.0,
                'correlation_matrix': None,
                'sector_allocation': {},
                'risk_metrics': None
            }

        # Calculate total portfolio value and risk
        total_value = sum([pos['value'] for pos in positions])
        total_risk = sum([pos.get('risk_amount', 0) for pos in positions])
        risk_percent = total_risk / total_value if total_value > 0 else 0

        # Get returns for correlation analysis
        symbols = [pos['symbol'] for pos in positions]
        returns_data = self._get_returns_matrix(symbols)

        correlation_matrix = None
        diversification_ratio = 1.0

        if returns_data is not None and len(returns_data.columns) > 1:
            correlation_matrix = returns_data.corr()

            # Calculate diversification ratio
            weights = np.array([pos['value'] / total_value for pos in positions])
            portfolio_variance = np.dot(weights, np.dot(returns_data.cov() * 252, weights))
            weighted_variance = sum([w**2 * returns_data[sym].var() * 252
                                   for w, sym in zip(weights, symbols)])
            diversification_ratio = weighted_variance / portfolio_variance if portfolio_variance > 0 else 1.0

        # Calculate portfolio risk metrics
        portfolio_returns = None
        if returns_data is not None:
            weights = [pos['value'] / total_value for pos in positions]
            portfolio_returns = (returns_data * weights).sum(axis=1)

        risk_metrics = self._calculate_risk_metrics(portfolio_returns) if portfolio_returns is not None else None

        # Sector allocation (simplified)
        sector_allocation = self._estimate_sector_allocation(symbols)

        return {
            'total_value': total_value,
            'total_risk': total_risk,
            'risk_percent': risk_percent,
            'diversification_ratio': diversification_ratio,
            'correlation_matrix': correlation_matrix,
            'sector_allocation': sector_allocation,
            'risk_metrics': risk_metrics,
            'position_count': len(positions),
            'average_correlation': correlation_matrix.mean().mean() if correlation_matrix is not None else 0
        }

    def check_risk_limits(self, positions: List[Dict], new_position: Optional[Dict] = None) -> Dict:
        """Check if portfolio exceeds risk limits"""

        all_positions = positions.copy()
        if new_position:
            all_positions.append(new_position)

        portfolio_risk = self.calculate_portfolio_risk(all_positions)

        violations = []
        warnings = []

        # Check portfolio risk limit
        if portfolio_risk['risk_percent'] > self.max_portfolio_risk:
            violations.append(f"Portfolio risk {portfolio_risk['risk_percent']:.2%} exceeds limit {self.max_portfolio_risk:.2%}")

        # Check individual position limits
        for pos in all_positions:
            pos_risk_percent = pos.get('risk_amount', 0) / portfolio_risk['total_value']
            if pos_risk_percent > self.max_position_risk:
                violations.append(f"{pos['symbol']} risk {pos_risk_percent:.2%} exceeds limit {self.max_position_risk:.2%}")

        # Check correlation limits
        if portfolio_risk['correlation_matrix'] is not None:
            high_correlations = []
            corr_matrix = portfolio_risk['correlation_matrix']
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > self.max_correlation:
                        sym1, sym2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_correlations.append(f"{sym1}-{sym2}: {corr:.2f}")

            if high_correlations:
                warnings.append(f"High correlations: {', '.join(high_correlations)}")

        # Check diversification
        if portfolio_risk['diversification_ratio'] < 1.5:
            warnings.append(f"Low diversification ratio: {portfolio_risk['diversification_ratio']:.2f}")

        return {
            'violations': violations,
            'warnings': warnings,
            'portfolio_risk': portfolio_risk,
            'risk_approved': len(violations) == 0
        }

    def suggest_risk_adjustments(self, positions: List[Dict]) -> List[str]:
        """Suggest adjustments to reduce portfolio risk"""

        suggestions = []
        portfolio_risk = self.calculate_portfolio_risk(positions)

        # If portfolio risk is too high
        if portfolio_risk['risk_percent'] > self.max_portfolio_risk:
            reduction_needed = portfolio_risk['risk_percent'] - self.max_portfolio_risk
            suggestions.append(f"Reduce overall position sizes by ~{reduction_needed:.1%} to meet risk limits")

        # If correlation is too high
        if portfolio_risk['average_correlation'] > self.max_correlation:
            suggestions.append("Consider adding positions in uncorrelated sectors or asset classes")

        # If diversification is poor
        if portfolio_risk['diversification_ratio'] < 1.5:
            suggestions.append("Increase diversification by adding positions in different sectors")

        # Individual position suggestions
        for pos in positions:
            risk_assessment = self.assess_position_risk(
                pos['symbol'], pos['quantity'], pos['entry_price'],
                pos['current_price'], pos['stop_loss']
            )

            if risk_assessment.risk_score > 70:
                suggestions.append(f"Consider reducing {pos['symbol']} position (risk score: {risk_assessment.risk_score:.0f})")

            if risk_assessment.volatility > 0.4:
                suggestions.append(f"{pos['symbol']} has high volatility ({risk_assessment.volatility:.1%}), consider tighter stops")

        return suggestions

    def _calculate_volatility(self, symbol: str, window: int = 30) -> float:
        """Calculate annualized volatility for a symbol"""
        try:
            if symbol not in self.price_cache:
                ticker = yf.Ticker(symbol)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.volatility_lookback)
                data = ticker.history(start=start_date, end=end_date)
                self.price_cache[symbol] = data

            data = self.price_cache[symbol]
            if len(data) < window:
                return 0.3  # Default volatility

            returns = data['Close'].pct_change().dropna()
            volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
            return volatility if not np.isnan(volatility) else 0.3

        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.3  # Default volatility

    def _calculate_beta(self, symbol: str) -> float:
        """Calculate beta relative to SPY"""
        try:
            if symbol == 'SPY' or self.spy_data is None:
                return 1.0

            if symbol not in self.price_cache:
                return 1.0  # Default beta

            stock_data = self.price_cache[symbol]

            # Align dates
            common_dates = stock_data.index.intersection(self.spy_data.index)
            if len(common_dates) < 60:
                return 1.0

            stock_returns = stock_data.loc[common_dates, 'Close'].pct_change().dropna()
            spy_returns = self.spy_data.loc[common_dates, 'Close'].pct_change().dropna()

            # Align the returns
            common_dates = stock_returns.index.intersection(spy_returns.index)
            if len(common_dates) < 60:
                return 1.0

            stock_returns = stock_returns.loc[common_dates]
            spy_returns = spy_returns.loc[common_dates]

            # Calculate beta
            covariance = np.cov(stock_returns, spy_returns)[0, 1]
            spy_variance = np.var(spy_returns)
            beta = covariance / spy_variance if spy_variance > 0 else 1.0

            return beta if not np.isnan(beta) else 1.0

        except Exception as e:
            print(f"Error calculating beta for {symbol}: {e}")
            return 1.0

    def _calculate_correlation_spy(self, symbol: str) -> float:
        """Calculate correlation with SPY"""
        try:
            if symbol == 'SPY' or self.spy_data is None:
                return 1.0

            if symbol not in self.price_cache:
                return 0.0

            stock_data = self.price_cache[symbol]

            # Align dates and calculate correlation
            common_dates = stock_data.index.intersection(self.spy_data.index)
            if len(common_dates) < 60:
                return 0.0

            stock_returns = stock_data.loc[common_dates, 'Close'].pct_change().dropna()
            spy_returns = self.spy_data.loc[common_dates, 'Close'].pct_change().dropna()

            # Align the returns
            common_dates = stock_returns.index.intersection(spy_returns.index)
            if len(common_dates) < 60:
                return 0.0

            correlation = np.corrcoef(stock_returns.loc[common_dates],
                                   spy_returns.loc[common_dates])[0, 1]

            return correlation if not np.isnan(correlation) else 0.0

        except Exception as e:
            print(f"Error calculating correlation for {symbol}: {e}")
            return 0.0

    def _calculate_risk_score(self, volatility: float, beta: float, risk_percent: float) -> float:
        """Calculate overall risk score (0-100)"""

        # Volatility component (0-40 points)
        vol_score = min(40, volatility * 100)

        # Beta component (0-30 points)
        beta_score = min(30, abs(beta - 1) * 30)

        # Position risk component (0-30 points)
        risk_score = min(30, risk_percent * 1000)

        total_score = vol_score + beta_score + risk_score
        return min(100, total_score)

    def _recommend_position_adjustment(self, symbol: str, current_price: float,
                                     stop_loss: float, position_value: float,
                                     volatility: float) -> int:
        """Recommend position size adjustment"""

        # This is a simplified recommendation
        # In practice, you'd want more sophisticated logic

        risk_per_share = abs(current_price - stop_loss)
        max_risk = position_value * self.max_position_risk

        recommended_size = int(max_risk / risk_per_share)

        # Adjust for volatility
        if volatility > 0.4:  # High volatility
            recommended_size = int(recommended_size * 0.7)
        elif volatility < 0.2:  # Low volatility
            recommended_size = int(recommended_size * 1.2)

        return max(1, recommended_size)

    def _get_returns_matrix(self, symbols: List[str], days: int = 252) -> Optional[pd.DataFrame]:
        """Get returns matrix for correlation analysis"""
        try:
            returns_data = {}

            for symbol in symbols:
                if symbol in self.price_cache:
                    data = self.price_cache[symbol]
                    returns = data['Close'].pct_change().dropna()
                    if len(returns) >= days // 2:  # At least half the requested data
                        returns_data[symbol] = returns.tail(days)

            if len(returns_data) < 2:
                return None

            return pd.DataFrame(returns_data).dropna()

        except Exception as e:
            print(f"Error creating returns matrix: {e}")
            return None

    def _calculate_risk_metrics(self, returns: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Basic statistics
            volatility = returns.std() * np.sqrt(252)
            mean_return = returns.mean() * 252

            # Sharpe ratio (assuming 2% risk-free rate)
            sharpe_ratio = (mean_return - 0.02) / volatility if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = (mean_return - 0.02) / downside_volatility if downside_volatility > 0 else 0

            # Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)

            # Expected Shortfall (Conditional VaR)
            es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
            es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99

            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Beta and correlation with SPY
            beta = self._calculate_beta('SPY')  # This would need symbol context
            correlation_spy = self._calculate_correlation_spy('SPY')

            return RiskMetrics(
                value_at_risk_95=var_95,
                value_at_risk_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                maximum_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                correlation_spy=correlation_spy
            )

        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0.3, 1.0, 0.0)

    def _estimate_sector_allocation(self, symbols: List[str]) -> Dict[str, float]:
        """Estimate sector allocation (simplified)"""

        # This is a simplified sector mapping
        # In practice, you'd want to use a proper sector classification API
        sector_mapping = {
            'AAPL': 'Technology',
            'MSFT': 'Technology',
            'GOOGL': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'JPM': 'Financial',
            'JNJ': 'Healthcare',
            'SPY': 'Diversified'
        }

        sector_count = {}
        for symbol in symbols:
            sector = sector_mapping.get(symbol, 'Other')
            sector_count[sector] = sector_count.get(sector, 0) + 1

        # Convert to percentages
        total = len(symbols)
        return {sector: count/total for sector, count in sector_count.items()}

def main():
    """Example usage of risk management system"""

    risk_manager = RiskManager()

    # Example portfolio
    positions = [
        {
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.0,
            'current_price': 155.0,
            'stop_loss': 145.0,
            'value': 15500
        },
        {
            'symbol': 'MSFT',
            'quantity': 50,
            'entry_price': 300.0,
            'current_price': 310.0,
            'stop_loss': 290.0,
            'value': 15500
        }
    ]

    # Calculate portfolio risk
    portfolio_risk = risk_manager.calculate_portfolio_risk(positions)
    print(f"Portfolio Risk Analysis:")
    print(f"Total Value: ${portfolio_risk['total_value']:,.0f}")
    print(f"Total Risk: ${portfolio_risk['total_risk']:,.0f}")
    print(f"Risk Percentage: {portfolio_risk['risk_percent']:.2%}")
    print(f"Diversification Ratio: {portfolio_risk['diversification_ratio']:.2f}")

    # Check risk limits
    risk_check = risk_manager.check_risk_limits(positions)
    print(f"\nRisk Check:")
    print(f"Approved: {risk_check['risk_approved']}")
    if risk_check['violations']:
        print(f"Violations: {risk_check['violations']}")
    if risk_check['warnings']:
        print(f"Warnings: {risk_check['warnings']}")

    # Position sizing example
    sizing = risk_manager.calculate_position_size(
        symbol='GOOGL',
        entry_price=100.0,
        stop_loss=95.0,
        portfolio_value=50000,
        confidence=0.8
    )
    print(f"\nPosition Sizing for GOOGL:")
    print(f"Recommended Size: {sizing['recommended_size']} shares")
    print(f"Risk Amount: ${sizing['actual_risk_amount']:.0f}")
    print(f"Risk Percentage: {sizing['risk_percent']:.2%}")

if __name__ == "__main__":
    main()