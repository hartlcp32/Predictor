#!/usr/bin/env python3
"""
Risk analysis system for evaluating prediction strategy performance
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy/pandas not installed. Using basic calculations for risk analysis.")


@dataclass
class RiskMetrics:
    """Risk metrics for a strategy or portfolio"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    value_at_risk: float
    expected_shortfall: float
    calmar_ratio: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    recovery_factor: float


class RiskAnalyzer:
    def __init__(self, data_dir='docs'):
        self.data_dir = data_dir
        self.predictions_file = os.path.join(data_dir, 'predictions_data.json')
        self.performance_file = os.path.join(data_dir, 'performance_data.json')
        self.risk_file = os.path.join(data_dir, 'risk_analysis.json')

    def load_predictions_data(self) -> Dict:
        """Load predictions data"""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return {'predictions': []}

    def load_performance_data(self) -> Dict:
        """Load performance data"""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return {'strategy_performance': {}}

    def extract_returns_series(self, strategy_name: str, predictions_data: Dict) -> List[float]:
        """Extract return series for a specific strategy"""
        returns = []

        for day_data in predictions_data['predictions']:
            if strategy_name in day_data['predictions']:
                prediction = day_data['predictions'][strategy_name]

                # Only include completed predictions with actual results
                if 'actual' in prediction and 'accuracy_metrics' in prediction:
                    actual_return = prediction['accuracy_metrics']['actual_return']
                    returns.append(actual_return)

        return returns

    def calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0

        if NUMPY_AVAILABLE:
            returns_array = np.array(returns)
            avg_return = np.mean(returns_array)
            volatility = np.std(returns_array)
        else:
            # Fallback to basic Python calculations
            avg_return = sum(returns) / len(returns)
            variance = sum((x - avg_return) ** 2 for x in returns) / len(returns)
            volatility = variance ** 0.5

        if volatility == 0:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_return = avg_return / 100 - daily_rf  # Convert % to decimal

        return excess_return / (volatility / 100) * np.sqrt(252)  # Annualized

    def calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) == 0:
            return 0.0

        returns_array = np.array(returns) / 100  # Convert to decimal
        avg_return = np.mean(returns_array)

        # Calculate downside deviation
        negative_returns = returns_array[returns_array < 0]
        if len(negative_returns) == 0:
            return float('inf')

        downside_deviation = np.sqrt(np.mean(negative_returns ** 2))

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate) ** (1/252) - 1
        excess_return = avg_return - daily_rf

        return excess_return / downside_deviation * np.sqrt(252)  # Annualized

    def calculate_max_drawdown(self, returns: List[float]) -> Tuple[float, int, int]:
        """Calculate maximum drawdown and its duration"""
        if len(returns) == 0:
            return 0.0, 0, 0

        cumulative_returns = np.cumprod(1 + np.array(returns) / 100)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max

        max_drawdown = np.min(drawdowns)
        max_dd_idx = np.argmin(drawdowns)

        # Find drawdown start and end
        start_idx = max_dd_idx
        while start_idx > 0 and drawdowns[start_idx - 1] >= drawdowns[start_idx]:
            start_idx -= 1

        end_idx = max_dd_idx
        while end_idx < len(drawdowns) - 1 and drawdowns[end_idx + 1] <= 0:
            end_idx += 1

        duration = end_idx - start_idx + 1

        return abs(max_drawdown), start_idx, duration

    def calculate_var_and_es(self, returns: List[float], confidence: float = 0.05) -> Tuple[float, float]:
        """Calculate Value at Risk and Expected Shortfall"""
        if len(returns) == 0:
            return 0.0, 0.0

        returns_array = np.array(returns)
        sorted_returns = np.sort(returns_array)

        # Value at Risk (percentile)
        var_index = int(confidence * len(sorted_returns))
        var = abs(sorted_returns[var_index]) if var_index < len(sorted_returns) else 0.0

        # Expected Shortfall (conditional VaR)
        tail_returns = sorted_returns[:var_index] if var_index > 0 else sorted_returns[:1]
        expected_shortfall = abs(np.mean(tail_returns)) if len(tail_returns) > 0 else 0.0

        return var, expected_shortfall

    def calculate_profit_metrics(self, returns: List[float]) -> Tuple[float, float, float, float]:
        """Calculate profit-related metrics"""
        if len(returns) == 0:
            return 0.0, 0.0, 0.0, 0.0

        returns_array = np.array(returns)
        wins = returns_array[returns_array > 0]
        losses = returns_array[returns_array < 0]

        win_rate = len(wins) / len(returns_array) if len(returns_array) > 0 else 0.0
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0

        # Profit factor
        total_profit = np.sum(wins) if len(wins) > 0 else 0.0
        total_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.01  # Avoid division by zero

        profit_factor = total_profit / total_loss

        return win_rate, avg_win, avg_loss, profit_factor

    def calculate_strategy_risk_metrics(self, strategy_name: str, predictions_data: Dict) -> RiskMetrics:
        """Calculate comprehensive risk metrics for a strategy"""
        returns = self.extract_returns_series(strategy_name, predictions_data)

        if len(returns) < 5:  # Need minimum data points
            return RiskMetrics(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                value_at_risk=0.0,
                expected_shortfall=0.0,
                calmar_ratio=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                recovery_factor=0.0
            )

        # Calculate all metrics
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        max_drawdown, _, _ = self.calculate_max_drawdown(returns)
        volatility = np.std(returns)
        var, expected_shortfall = self.calculate_var_and_es(returns)
        win_rate, avg_win, avg_loss, profit_factor = self.calculate_profit_metrics(returns)

        # Calmar ratio (annual return / max drawdown)
        annual_return = np.mean(returns) * 252 if len(returns) > 0 else 0.0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

        # Recovery factor (total return / max drawdown)
        total_return = np.sum(returns) if len(returns) > 0 else 0.0
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0.0

        return RiskMetrics(
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown * 100,  # Convert to percentage
            volatility=volatility,
            value_at_risk=var,
            expected_shortfall=expected_shortfall,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            recovery_factor=recovery_factor
        )

    def calculate_portfolio_risk_metrics(self, predictions_data: Dict) -> RiskMetrics:
        """Calculate risk metrics for the entire portfolio (all strategies combined)"""
        # Aggregate returns across all strategies by date
        daily_returns = {}

        for day_data in predictions_data['predictions']:
            date = day_data['date']
            day_return = 0.0
            strategy_count = 0

            for strategy_name, prediction in day_data['predictions'].items():
                if 'actual' in prediction and 'accuracy_metrics' in prediction:
                    actual_return = prediction['accuracy_metrics']['actual_return']
                    day_return += actual_return
                    strategy_count += 1

            if strategy_count > 0:
                daily_returns[date] = day_return / strategy_count  # Average return

        portfolio_returns = list(daily_returns.values())
        return self.calculate_strategy_risk_metrics('portfolio', {
            'predictions': [
                {'date': date, 'predictions': {
                    'portfolio': {
                        'actual': f"{ret:+.1f}%",
                        'accuracy_metrics': {'actual_return': ret}
                    }
                }} for date, ret in daily_returns.items()
            ]
        })

    def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk analysis report"""
        predictions_data = self.load_predictions_data()

        if not predictions_data['predictions']:
            return self.generate_sample_risk_report()

        # Calculate risk metrics for each strategy
        strategy_risk_metrics = {}
        strategy_names = set()

        # Collect all strategy names
        for day_data in predictions_data['predictions']:
            strategy_names.update(day_data['predictions'].keys())

        # Calculate metrics for each strategy
        for strategy_name in strategy_names:
            risk_metrics = self.calculate_strategy_risk_metrics(strategy_name, predictions_data)
            strategy_risk_metrics[strategy_name] = {
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'max_drawdown': risk_metrics.max_drawdown,
                'volatility': risk_metrics.volatility,
                'value_at_risk_95': risk_metrics.value_at_risk,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'calmar_ratio': risk_metrics.calmar_ratio,
                'win_rate': risk_metrics.win_rate,
                'avg_win': risk_metrics.avg_win,
                'avg_loss': risk_metrics.avg_loss,
                'profit_factor': risk_metrics.profit_factor,
                'recovery_factor': risk_metrics.recovery_factor,
                'risk_adjusted_return': risk_metrics.sharpe_ratio,  # Proxy
                'risk_score': self.calculate_risk_score(risk_metrics)
            }

        # Calculate portfolio-level metrics
        portfolio_metrics = self.calculate_portfolio_risk_metrics(predictions_data)
        portfolio_risk_data = {
            'sharpe_ratio': portfolio_metrics.sharpe_ratio,
            'sortino_ratio': portfolio_metrics.sortino_ratio,
            'max_drawdown': portfolio_metrics.max_drawdown,
            'volatility': portfolio_metrics.volatility,
            'value_at_risk_95': portfolio_metrics.value_at_risk,
            'expected_shortfall': portfolio_metrics.expected_shortfall,
            'calmar_ratio': portfolio_metrics.calmar_ratio,
            'win_rate': portfolio_metrics.win_rate,
            'profit_factor': portfolio_metrics.profit_factor,
            'risk_score': self.calculate_risk_score(portfolio_metrics)
        }

        # Risk rankings
        risk_rankings = self.calculate_risk_rankings(strategy_risk_metrics)

        report = {
            'strategy_risk_metrics': strategy_risk_metrics,
            'portfolio_risk_metrics': portfolio_risk_data,
            'risk_rankings': risk_rankings,
            'market_conditions': self.analyze_market_conditions(predictions_data),
            'risk_alerts': self.generate_risk_alerts(strategy_risk_metrics, portfolio_risk_data),
            'generated_at': datetime.now().isoformat()
        }

        # Save report
        with open(self.risk_file, 'w') as f:
            json.dump(report, f, indent=2)

        return report

    def calculate_risk_score(self, risk_metrics: RiskMetrics) -> float:
        """Calculate overall risk score (0-100, lower is better)"""
        score = 0.0

        # Sharpe ratio component (25% weight)
        if risk_metrics.sharpe_ratio > 1.5:
            score += 0
        elif risk_metrics.sharpe_ratio > 1.0:
            score += 10
        elif risk_metrics.sharpe_ratio > 0.5:
            score += 20
        else:
            score += 30

        # Max drawdown component (30% weight)
        if risk_metrics.max_drawdown < 5:
            score += 0
        elif risk_metrics.max_drawdown < 10:
            score += 15
        elif risk_metrics.max_drawdown < 20:
            score += 25
        else:
            score += 35

        # Volatility component (25% weight)
        if risk_metrics.volatility < 10:
            score += 0
        elif risk_metrics.volatility < 20:
            score += 10
        elif risk_metrics.volatility < 30:
            score += 20
        else:
            score += 25

        # Win rate component (20% weight)
        if risk_metrics.win_rate > 0.7:
            score += 0
        elif risk_metrics.win_rate > 0.6:
            score += 5
        elif risk_metrics.win_rate > 0.5:
            score += 10
        else:
            score += 20

        return min(score, 100)  # Cap at 100

    def calculate_risk_rankings(self, strategy_risk_metrics: Dict) -> Dict:
        """Calculate risk-based rankings for strategies"""
        strategies = list(strategy_risk_metrics.keys())

        rankings = {
            'best_sharpe': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['sharpe_ratio'], reverse=True),
            'lowest_drawdown': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['max_drawdown']),
            'best_sortino': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['sortino_ratio'], reverse=True),
            'lowest_volatility': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['volatility']),
            'best_profit_factor': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['profit_factor'], reverse=True),
            'lowest_risk_score': sorted(strategies, key=lambda x: strategy_risk_metrics[x]['risk_score'])
        }

        return rankings

    def analyze_market_conditions(self, predictions_data: Dict) -> Dict:
        """Analyze current market conditions based on recent predictions"""
        # Simple market condition analysis
        recent_predictions = predictions_data['predictions'][-5:] if len(predictions_data['predictions']) >= 5 else predictions_data['predictions']

        long_count = 0
        short_count = 0
        total_predictions = 0

        for day_data in recent_predictions:
            for strategy_name, prediction in day_data['predictions'].items():
                if prediction.get('position') == 'LONG':
                    long_count += 1
                elif prediction.get('position') == 'SHORT':
                    short_count += 1
                total_predictions += 1

        if total_predictions == 0:
            return {'condition': 'UNKNOWN', 'sentiment': 'NEUTRAL', 'confidence': 0}

        long_ratio = long_count / total_predictions
        short_ratio = short_count / total_predictions

        if long_ratio > 0.7:
            condition = 'BULLISH'
            sentiment = 'POSITIVE'
        elif short_ratio > 0.7:
            condition = 'BEARISH'
            sentiment = 'NEGATIVE'
        else:
            condition = 'NEUTRAL'
            sentiment = 'MIXED'

        return {
            'condition': condition,
            'sentiment': sentiment,
            'long_ratio': long_ratio,
            'short_ratio': short_ratio,
            'confidence': max(long_ratio, short_ratio)
        }

    def generate_risk_alerts(self, strategy_risk_metrics: Dict, portfolio_risk_metrics: Dict) -> List[Dict]:
        """Generate risk alerts based on metrics"""
        alerts = []

        # Portfolio-level alerts
        if portfolio_risk_metrics['max_drawdown'] > 20:
            alerts.append({
                'level': 'HIGH',
                'type': 'DRAWDOWN',
                'message': f"Portfolio max drawdown of {portfolio_risk_metrics['max_drawdown']:.1f}% exceeds 20% threshold"
            })

        if portfolio_risk_metrics['sharpe_ratio'] < 0.5:
            alerts.append({
                'level': 'MEDIUM',
                'type': 'PERFORMANCE',
                'message': f"Portfolio Sharpe ratio of {portfolio_risk_metrics['sharpe_ratio']:.2f} is below acceptable threshold"
            })

        # Strategy-level alerts
        for strategy_name, metrics in strategy_risk_metrics.items():
            if metrics['risk_score'] > 80:
                alerts.append({
                    'level': 'HIGH',
                    'type': 'RISK_SCORE',
                    'message': f"Strategy '{strategy_name}' has high risk score of {metrics['risk_score']:.0f}/100"
                })

            if metrics['max_drawdown'] > 30:
                alerts.append({
                    'level': 'HIGH',
                    'type': 'STRATEGY_DRAWDOWN',
                    'message': f"Strategy '{strategy_name}' max drawdown of {metrics['max_drawdown']:.1f}% is excessive"
                })

        return alerts

    def generate_sample_risk_report(self) -> Dict:
        """Generate sample risk report for demonstration"""
        strategies = ['momentum', 'mean_reversion', 'volume_breakout', 'technical_indicators',
                     'pattern_recognition', 'volatility_arbitrage', 'moving_average_crossover',
                     'support_resistance', 'market_sentiment', 'ensemble']

        strategy_risk_metrics = {}
        for strategy in strategies:
            strategy_risk_metrics[strategy] = {
                'sharpe_ratio': np.random.normal(0.8, 0.3),
                'sortino_ratio': np.random.normal(1.1, 0.4),
                'max_drawdown': np.random.uniform(5, 25),
                'volatility': np.random.uniform(10, 30),
                'value_at_risk_95': np.random.uniform(8, 15),
                'expected_shortfall': np.random.uniform(12, 20),
                'calmar_ratio': np.random.uniform(0.5, 2.0),
                'win_rate': np.random.uniform(0.45, 0.75),
                'avg_win': np.random.uniform(5, 12),
                'avg_loss': np.random.uniform(4, 10),
                'profit_factor': np.random.uniform(0.8, 2.5),
                'recovery_factor': np.random.uniform(1.0, 3.0),
                'risk_adjusted_return': np.random.uniform(0.5, 1.5),
                'risk_score': np.random.uniform(20, 80)
            }

        portfolio_risk_metrics = {
            'sharpe_ratio': 0.95,
            'sortino_ratio': 1.25,
            'max_drawdown': 12.3,
            'volatility': 18.5,
            'value_at_risk_95': 9.2,
            'expected_shortfall': 14.1,
            'calmar_ratio': 1.8,
            'win_rate': 0.62,
            'profit_factor': 1.45,
            'risk_score': 35.2
        }

        return {
            'strategy_risk_metrics': strategy_risk_metrics,
            'portfolio_risk_metrics': portfolio_risk_metrics,
            'risk_rankings': self.calculate_risk_rankings(strategy_risk_metrics),
            'market_conditions': {'condition': 'BULLISH', 'sentiment': 'POSITIVE', 'confidence': 0.68},
            'risk_alerts': [],
            'generated_at': datetime.now().isoformat()
        }

    def print_risk_summary(self):
        """Print risk analysis summary to console"""
        report = self.generate_risk_report()

        print("\n" + "="*70)
        print("📊 STOCK PREDICTOR RISK ANALYSIS REPORT")
        print("="*70)

        # Portfolio metrics
        portfolio = report['portfolio_risk_metrics']
        print(f"\n🎯 PORTFOLIO RISK METRICS:")
        print(f"   Sharpe Ratio: {portfolio['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {portfolio['max_drawdown']:.1f}%")
        print(f"   Volatility: {portfolio['volatility']:.1f}%")
        print(f"   Win Rate: {portfolio['win_rate']:.1%}")
        print(f"   Risk Score: {portfolio['risk_score']:.0f}/100")

        # Best performers
        rankings = report['risk_rankings']
        print(f"\n🏆 BEST RISK-ADJUSTED PERFORMERS:")
        print(f"   Best Sharpe: {rankings['best_sharpe'][0] if rankings['best_sharpe'] else 'N/A'}")
        print(f"   Lowest Drawdown: {rankings['lowest_drawdown'][0] if rankings['lowest_drawdown'] else 'N/A'}")
        print(f"   Lowest Risk Score: {rankings['lowest_risk_score'][0] if rankings['lowest_risk_score'] else 'N/A'}")

        # Market conditions
        market = report['market_conditions']
        print(f"\n🌍 MARKET CONDITIONS:")
        print(f"   Condition: {market['condition']}")
        print(f"   Sentiment: {market['sentiment']}")
        print(f"   Confidence: {market.get('confidence', 0):.1%}")

        # Alerts
        alerts = report['risk_alerts']
        if alerts:
            print(f"\n⚠️  RISK ALERTS:")
            for alert in alerts[:3]:  # Show top 3 alerts
                print(f"   {alert['level']}: {alert['message']}")

        print(f"\n✅ Risk analysis saved to: {self.risk_file}")


if __name__ == "__main__":
    analyzer = RiskAnalyzer()
    analyzer.print_risk_summary()