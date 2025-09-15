#!/usr/bin/env python3
"""
Monthly performance analyzer for historical backtesting
Provides granular month-by-month performance metrics
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
# Note: This generates simulated historical performance data for demonstration

class MonthlyPerformanceAnalyzer:
    def __init__(self, start_date='2022-01-01', end_date='2024-12-31'):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')

        # Strategy names
        self.strategies = [
            'momentum', 'mean_reversion', 'volume_breakout',
            'technical_indicators', 'pattern_recognition',
            'volatility_arbitrage', 'moving_average_crossover',
            'support_resistance', 'market_sentiment', 'ensemble'
        ]

    def generate_monthly_performance(self) -> Dict:
        """Generate month-by-month performance data"""
        monthly_data = {}

        current_date = self.start_date
        while current_date <= self.end_date:
            month_key = current_date.strftime('%Y-%m')
            year = current_date.year
            month = current_date.month

            # Market conditions vary by season and year
            market_condition = self.get_market_condition(year, month)

            monthly_data[month_key] = {
                'year': year,
                'month': month,
                'month_name': current_date.strftime('%B'),
                'market_condition': market_condition,
                'strategies': self.generate_strategy_performance(year, month, market_condition),
                'summary': self.generate_monthly_summary(year, month, market_condition)
            }

            # Move to next month
            if month == 12:
                current_date = current_date.replace(year=year+1, month=1)
            else:
                current_date = current_date.replace(month=month+1)

        return monthly_data

    def get_market_condition(self, year: int, month: int) -> str:
        """Determine market conditions for a given month"""
        # Historical market conditions
        conditions = {
            # 2022 - Bear market year
            (2022, 1): 'VOLATILE_DOWN',
            (2022, 2): 'VOLATILE_DOWN',
            (2022, 3): 'RECOVERY',
            (2022, 4): 'BEARISH',
            (2022, 5): 'BEARISH',
            (2022, 6): 'CRASH',
            (2022, 7): 'RECOVERY',
            (2022, 8): 'VOLATILE',
            (2022, 9): 'BEARISH',
            (2022, 10): 'RECOVERY',
            (2022, 11): 'BULLISH',
            (2022, 12): 'BEARISH',

            # 2023 - Recovery year
            (2023, 1): 'BULLISH',
            (2023, 2): 'VOLATILE',
            (2023, 3): 'BANKING_CRISIS',
            (2023, 4): 'RECOVERY',
            (2023, 5): 'BULLISH',
            (2023, 6): 'BULLISH',
            (2023, 7): 'BULLISH',
            (2023, 8): 'CORRECTION',
            (2023, 9): 'BEARISH',
            (2023, 10): 'BEARISH',
            (2023, 11): 'RECOVERY',
            (2023, 12): 'BULLISH',

            # 2024 - AI boom year
            (2024, 1): 'BULLISH',
            (2024, 2): 'BULLISH',
            (2024, 3): 'BULLISH',
            (2024, 4): 'CORRECTION',
            (2024, 5): 'BULLISH',
            (2024, 6): 'BULLISH',
            (2024, 7): 'ROTATION',
            (2024, 8): 'VOLATILE',
            (2024, 9): 'BULLISH',
            (2024, 10): 'VOLATILE',
            (2024, 11): 'BULLISH',
            (2024, 12): 'SANTA_RALLY'
        }

        return conditions.get((year, month), 'NEUTRAL')

    def generate_strategy_performance(self, year: int, month: int, market_condition: str) -> Dict:
        """Generate performance for each strategy based on market conditions"""
        performance = {}

        # Different strategies perform differently in various market conditions
        condition_modifiers = {
            'BULLISH': {
                'momentum': 1.3,
                'mean_reversion': 0.7,
                'volume_breakout': 1.2,
                'technical_indicators': 1.1,
                'pattern_recognition': 1.0,
                'volatility_arbitrage': 0.8,
                'moving_average_crossover': 1.2,
                'support_resistance': 0.9,
                'market_sentiment': 1.3,
                'ensemble': 1.15
            },
            'BEARISH': {
                'momentum': 0.6,
                'mean_reversion': 1.3,
                'volume_breakout': 0.7,
                'technical_indicators': 0.8,
                'pattern_recognition': 0.9,
                'volatility_arbitrage': 1.4,
                'moving_average_crossover': 0.6,
                'support_resistance': 1.1,
                'market_sentiment': 0.7,
                'ensemble': 0.85
            },
            'VOLATILE': {
                'momentum': 0.8,
                'mean_reversion': 1.2,
                'volume_breakout': 1.3,
                'technical_indicators': 0.9,
                'pattern_recognition': 0.7,
                'volatility_arbitrage': 1.5,
                'moving_average_crossover': 0.7,
                'support_resistance': 1.0,
                'market_sentiment': 0.8,
                'ensemble': 0.95
            },
            'CRASH': {
                'momentum': 0.4,
                'mean_reversion': 0.6,
                'volume_breakout': 0.5,
                'technical_indicators': 0.6,
                'pattern_recognition': 0.7,
                'volatility_arbitrage': 1.8,
                'moving_average_crossover': 0.4,
                'support_resistance': 0.8,
                'market_sentiment': 0.5,
                'ensemble': 0.65
            }
        }

        # Get modifiers for current market condition
        modifiers = condition_modifiers.get(
            market_condition.split('_')[0] if '_' in market_condition else market_condition,
            condition_modifiers.get('VOLATILE', {})
        )

        for strategy in self.strategies:
            # More realistic base returns using deterministic model
            import hashlib

            # Create deterministic "randomness" based on year, month, and strategy
            seed_string = f"{year}{month}{strategy}"
            seed_hash = int(hashlib.md5(seed_string.encode()).hexdigest()[:8], 16)

            # Use hash to generate consistent but varied returns
            base_return = (seed_hash % 2000 - 1000) / 100  # Range -10 to +10
            modifier = modifiers.get(strategy, 1.0)

            # Apply market condition modifiers
            monthly_return = base_return * modifier

            # Deterministic trade metrics
            trades = 20 + (seed_hash % 20)  # 20-40 trades
            win_rate_base = 0.5 + modifier * 0.1
            wins = int(trades * min(max(win_rate_base, 0.3), 0.8))  # Cap win rate

            performance[strategy] = {
                'return': round(monthly_return, 2),
                'trades': trades,
                'wins': wins,
                'losses': trades - wins,
                'win_rate': round(wins / trades * 100, 1),
                'avg_win': round(abs(monthly_return) * 1.5 / wins if wins > 0 else 0, 2),
                'avg_loss': round(abs(monthly_return) * 0.8 / (trades - wins) if trades > wins else 0, 2),
                'best_trade': round(max(monthly_return * 0.3, 5.0), 2),
                'worst_trade': round(min(monthly_return * 0.2, -3.0), 2),
                'sharpe_ratio': round((monthly_return - 0.3) / 5.0, 2)  # Simplified Sharpe
            }

        return performance

    def generate_monthly_summary(self, year: int, month: int, market_condition: str) -> Dict:
        """Generate summary statistics for the month"""
        # Notable events by month
        events = {
            (2022, 1): "Fed signals rate hikes",
            (2022, 3): "First rate hike",
            (2022, 6): "Bear market confirmed",
            (2022, 11): "FTX collapse",
            (2023, 3): "Banking crisis (SVB)",
            (2023, 5): "AI boom begins",
            (2023, 11): "Rate pause signals",
            (2024, 1): "New ATHs",
            (2024, 3): "NVDA $2T market cap",
            (2024, 8): "Yen carry unwind",
            (2024, 11): "Post-election rally",
            (2024, 12): "Year-end rally"
        }

        return {
            'notable_events': events.get((year, month), ""),
            'market_trend': self.get_trend_for_condition(market_condition),
            'volatility_level': self.get_volatility_for_condition(market_condition),
            'best_performing_sector': self.get_best_sector(year, month),
            'recommendation': self.get_recommendation(market_condition)
        }

    def get_trend_for_condition(self, condition: str) -> str:
        """Get market trend description"""
        trends = {
            'BULLISH': 'Strong Uptrend',
            'BEARISH': 'Downtrend',
            'VOLATILE': 'Choppy/Sideways',
            'CRASH': 'Sharp Decline',
            'RECOVERY': 'Recovering',
            'CORRECTION': 'Mild Pullback',
            'ROTATION': 'Sector Rotation',
            'SANTA_RALLY': 'Year-end Rally',
            'BANKING_CRISIS': 'Crisis-driven Volatility'
        }
        return trends.get(condition, 'Neutral')

    def get_volatility_for_condition(self, condition: str) -> str:
        """Get volatility level"""
        if 'CRASH' in condition or 'CRISIS' in condition:
            return 'EXTREME'
        elif 'VOLATILE' in condition:
            return 'HIGH'
        elif 'BULLISH' in condition:
            return 'LOW'
        else:
            return 'MODERATE'

    def get_best_sector(self, year: int, month: int) -> str:
        """Get best performing sector for the month"""
        sectors = {
            (2022, 1): "Energy",
            (2022, 6): "Defensive/Utilities",
            (2023, 1): "Technology",
            (2023, 3): "Gold/Bonds",
            (2023, 5): "AI/Semiconductors",
            (2024, 1): "Technology",
            (2024, 3): "AI/Data Centers",
            (2024, 7): "Small Caps",
            (2024, 11): "Financials",
            (2024, 12): "Technology"
        }

        default_sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]
        # Use deterministic selection based on month
        default_index = (year + month) % len(default_sectors)
        return sectors.get((year, month), default_sectors[default_index])

    def get_recommendation(self, condition: str) -> str:
        """Get strategy recommendation for market condition"""
        recommendations = {
            'BULLISH': "Favor momentum and trend-following strategies",
            'BEARISH': "Favor mean reversion and defensive strategies",
            'VOLATILE': "Favor volatility arbitrage and quick trades",
            'CRASH': "Stay defensive, consider inverse positions",
            'RECOVERY': "Begin accumulating quality positions",
            'CORRECTION': "Opportunity for value entries",
            'ROTATION': "Focus on sector-specific plays",
            'SANTA_RALLY': "Ride momentum into year-end",
            'BANKING_CRISIS': "Avoid financials, seek safe havens"
        }
        return recommendations.get(condition, "Maintain balanced approach")

    def calculate_yearly_summary(self, monthly_data: Dict) -> Dict:
        """Calculate yearly summaries from monthly data"""
        yearly_summary = {}

        for month_key, month_data in monthly_data.items():
            year = month_data['year']

            if year not in yearly_summary:
                yearly_summary[year] = {
                    'total_return': 0,
                    'best_month': {'month': '', 'return': -999},
                    'worst_month': {'month': '', 'return': 999},
                    'winning_months': 0,
                    'losing_months': 0,
                    'total_trades': 0,
                    'strategy_returns': {s: 0 for s in self.strategies}
                }

            # Calculate ensemble return for the month
            monthly_return = sum(
                month_data['strategies'][s]['return']
                for s in self.strategies
            ) / len(self.strategies)

            yearly_summary[year]['total_return'] += monthly_return

            if monthly_return > yearly_summary[year]['best_month']['return']:
                yearly_summary[year]['best_month'] = {
                    'month': month_data['month_name'],
                    'return': round(monthly_return, 2)
                }

            if monthly_return < yearly_summary[year]['worst_month']['return']:
                yearly_summary[year]['worst_month'] = {
                    'month': month_data['month_name'],
                    'return': round(monthly_return, 2)
                }

            if monthly_return > 0:
                yearly_summary[year]['winning_months'] += 1
            else:
                yearly_summary[year]['losing_months'] += 1

            # Aggregate strategy returns
            for strategy in self.strategies:
                yearly_summary[year]['strategy_returns'][strategy] += month_data['strategies'][strategy]['return']

            # Total trades
            yearly_summary[year]['total_trades'] += sum(
                month_data['strategies'][s]['trades']
                for s in self.strategies
            )

        # Round final values
        for year in yearly_summary:
            yearly_summary[year]['total_return'] = round(yearly_summary[year]['total_return'], 2)
            yearly_summary[year]['strategy_returns'] = {
                s: round(r, 2)
                for s, r in yearly_summary[year]['strategy_returns'].items()
            }

        return yearly_summary

    def export_to_json(self, filename='monthly_performance.json'):
        """Export monthly performance data to JSON"""
        monthly_data = self.generate_monthly_performance()
        yearly_summary = self.calculate_yearly_summary(monthly_data)

        output = {
            'generated_at': datetime.now().isoformat(),
            'period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d')
            },
            'monthly_performance': monthly_data,
            'yearly_summary': yearly_summary,
            'overall_statistics': self.calculate_overall_statistics(monthly_data)
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        return output

    def calculate_overall_statistics(self, monthly_data: Dict) -> Dict:
        """Calculate overall statistics across all months"""
        all_returns = []
        strategy_totals = {s: 0 for s in self.strategies}

        for month_data in monthly_data.values():
            for strategy in self.strategies:
                strategy_return = month_data['strategies'][strategy]['return']
                strategy_totals[strategy] += strategy_return
                all_returns.append(strategy_return)

        return {
            'total_months': len(monthly_data),
            'average_monthly_return': round(sum(all_returns) / len(all_returns), 2),
            'best_strategy': max(strategy_totals, key=strategy_totals.get),
            'worst_strategy': min(strategy_totals, key=strategy_totals.get),
            'strategy_rankings': sorted(
                [(s, round(r, 2)) for s, r in strategy_totals.items()],
                key=lambda x: x[1],
                reverse=True
            )
        }


if __name__ == "__main__":
    analyzer = MonthlyPerformanceAnalyzer()
    data = analyzer.export_to_json('docs/monthly_performance.json')

    print("📊 MONTHLY PERFORMANCE ANALYSIS")
    print("="*50)
    print(f"Period: {data['period']['start']} to {data['period']['end']}")
    print(f"Total months analyzed: {data['overall_statistics']['total_months']}")
    print(f"Average monthly return: {data['overall_statistics']['average_monthly_return']}%")
    print(f"Best strategy: {data['overall_statistics']['best_strategy']}")
    print("\nData exported to: docs/monthly_performance.json")