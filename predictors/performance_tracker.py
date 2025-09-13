#!/usr/bin/env python3
"""
Performance tracking system for validating prediction accuracy
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class PerformanceTracker:
    def __init__(self, data_dir='docs'):
        self.data_dir = data_dir
        self.predictions_file = os.path.join(data_dir, 'predictions_data.json')
        self.performance_file = os.path.join(data_dir, 'performance_data.json')

    def load_predictions_data(self) -> Dict:
        """Load existing predictions data"""
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                return json.load(f)
        return {'predictions': [], 'statistics': {}}

    def load_performance_data(self) -> Dict:
        """Load existing performance tracking data"""
        if os.path.exists(self.performance_file):
            with open(self.performance_file, 'r') as f:
                return json.load(f)
        return {
            'strategy_performance': {},
            'daily_results': [],
            'overall_stats': {}
        }

    def fetch_actual_returns(self, symbol: str, start_date: str, days: int = 7) -> Optional[float]:
        """Fetch actual stock returns for a given period"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = start + timedelta(days=days + 2)  # Buffer for weekends

            data = ticker.history(start=start, end=end)
            if len(data) >= 2:
                start_price = data['Close'].iloc[0]

                # Try to get exactly 'days' trading days later
                if len(data) >= days + 1:
                    end_price = data['Close'].iloc[days]
                else:
                    end_price = data['Close'].iloc[-1]  # Use last available

                return (end_price - start_price) / start_price * 100
            return None
        except Exception as e:
            print(f"Error fetching returns for {symbol}: {e}")
            return None

    def calculate_prediction_accuracy(self, predicted: str, actual: float) -> Dict:
        """Calculate accuracy metrics for a prediction"""
        try:
            # Parse predicted return (e.g., "+7.5%" -> 7.5)
            predicted_val = float(predicted.replace('%', '').replace('+', ''))

            # Direction accuracy
            predicted_direction = 1 if predicted_val > 0 else -1 if predicted_val < 0 else 0
            actual_direction = 1 if actual > 0 else -1 if actual < 0 else 0
            direction_correct = predicted_direction == actual_direction

            # Magnitude accuracy (how close the prediction was)
            magnitude_error = abs(predicted_val - actual)
            magnitude_accuracy = max(0, 100 - magnitude_error * 10)  # Scale error

            return {
                'predicted_return': predicted_val,
                'actual_return': actual,
                'direction_correct': direction_correct,
                'magnitude_error': magnitude_error,
                'magnitude_accuracy': magnitude_accuracy,
                'absolute_error': abs(predicted_val - actual)
            }
        except Exception as e:
            print(f"Error calculating accuracy: {e}")
            return None

    def update_prediction_results(self) -> Dict:
        """Update predictions with actual results for completed time periods"""
        predictions_data = self.load_predictions_data()
        updated = False

        current_date = datetime.now()

        for day_data in predictions_data['predictions']:
            pred_date = datetime.strptime(day_data['date'], '%Y-%m-%d')
            days_elapsed = (current_date - pred_date).days

            # Only process predictions older than 7 days
            if days_elapsed >= 7:
                for strategy_name, prediction in day_data['predictions'].items():
                    # Skip if already has actual result or no stock pick
                    if 'actual' in prediction or prediction.get('stock') == 'NONE':
                        continue

                    # Fetch actual returns
                    actual_return = self.fetch_actual_returns(
                        prediction['stock'],
                        day_data['date'],
                        7
                    )

                    if actual_return is not None:
                        prediction['actual'] = f"{actual_return:+.1f}%"
                        prediction['accuracy_metrics'] = self.calculate_prediction_accuracy(
                            prediction['projected'],
                            actual_return
                        )
                        updated = True
                        print(f"✅ Updated {strategy_name} {prediction['stock']}: {prediction['projected']} vs {prediction['actual']}")

        if updated:
            # Save updated predictions
            with open(self.predictions_file, 'w') as f:
                json.dump(predictions_data, f, indent=2)

        return predictions_data

    def calculate_strategy_performance(self, predictions_data: Dict) -> Dict:
        """Calculate real performance statistics for each strategy"""
        strategy_stats = {}

        for day_data in predictions_data['predictions']:
            for strategy_name, prediction in day_data['predictions'].items():
                if strategy_name not in strategy_stats:
                    strategy_stats[strategy_name] = {
                        'total_predictions': 0,
                        'completed_predictions': 0,
                        'correct_direction': 0,
                        'total_return': 0.0,
                        'win_rate': 0.0,
                        'avg_return': 0.0,
                        'best_pick': None,
                        'worst_pick': None
                    }

                stats = strategy_stats[strategy_name]
                stats['total_predictions'] += 1

                # Only process completed predictions with actual results
                if 'actual' in prediction and 'accuracy_metrics' in prediction:
                    stats['completed_predictions'] += 1
                    metrics = prediction['accuracy_metrics']

                    if metrics['direction_correct']:
                        stats['correct_direction'] += 1

                    actual_return = metrics['actual_return']
                    stats['total_return'] += actual_return

                    # Track best and worst picks
                    pick_data = {
                        'date': day_data['date'],
                        'stock': prediction['stock'],
                        'predicted': prediction['projected'],
                        'actual': prediction['actual'],
                        'return': actual_return
                    }

                    if stats['best_pick'] is None or actual_return > stats['best_pick']['return']:
                        stats['best_pick'] = pick_data

                    if stats['worst_pick'] is None or actual_return < stats['worst_pick']['return']:
                        stats['worst_pick'] = pick_data

        # Calculate final metrics
        for strategy_name, stats in strategy_stats.items():
            if stats['completed_predictions'] > 0:
                stats['win_rate'] = stats['correct_direction'] / stats['completed_predictions']
                stats['avg_return'] = stats['total_return'] / stats['completed_predictions']
            else:
                stats['win_rate'] = 0.0
                stats['avg_return'] = 0.0

        return strategy_stats

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        # Update predictions with actual results
        predictions_data = self.update_prediction_results()

        # Calculate strategy performance
        strategy_performance = self.calculate_strategy_performance(predictions_data)

        # Calculate overall statistics
        total_predictions = sum(s['completed_predictions'] for s in strategy_performance.values())
        total_correct = sum(s['correct_direction'] for s in strategy_performance.values())
        total_return = sum(s['total_return'] for s in strategy_performance.values())

        overall_stats = {
            'total_predictions_made': sum(s['total_predictions'] for s in strategy_performance.values()),
            'total_predictions_completed': total_predictions,
            'overall_win_rate': total_correct / total_predictions if total_predictions > 0 else 0,
            'overall_avg_return': total_return / total_predictions if total_predictions > 0 else 0,
            'best_strategy': max(strategy_performance.keys(),
                               key=lambda x: strategy_performance[x]['win_rate']) if strategy_performance else None,
            'worst_strategy': min(strategy_performance.keys(),
                                key=lambda x: strategy_performance[x]['win_rate']) if strategy_performance else None,
            'last_updated': datetime.now().isoformat()
        }

        # Save performance data
        performance_data = {
            'strategy_performance': strategy_performance,
            'overall_stats': overall_stats,
            'generated_at': datetime.now().isoformat()
        }

        with open(self.performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)

        # Update the statistics in predictions_data.json
        predictions_data['statistics'] = {
            'total_predictions': overall_stats['total_predictions_completed'],
            'overall_accuracy': f"{overall_stats['overall_win_rate']:.1%}",
            'best_strategy': overall_stats['best_strategy'],
            'overall_return': f"{overall_stats['overall_avg_return']:+.1%}",
            'strategies': {
                name: {
                    'win_rate': stats['win_rate'],
                    'total_return': stats['total_return'],
                    'avg_return': stats['avg_return'],
                    'completed_predictions': stats['completed_predictions']
                }
                for name, stats in strategy_performance.items()
            }
        }

        with open(self.predictions_file, 'w') as f:
            json.dump(predictions_data, f, indent=2)

        return performance_data

    def print_performance_summary(self):
        """Print a summary of performance to console"""
        performance_data = self.generate_performance_report()

        print("\n" + "="*60)
        print("📊 STOCK PREDICTOR PERFORMANCE REPORT")
        print("="*60)

        overall = performance_data['overall_stats']
        print(f"Total Predictions Made: {overall['total_predictions_made']}")
        print(f"Completed Predictions: {overall['total_predictions_completed']}")
        print(f"Overall Win Rate: {overall['overall_win_rate']:.1%}")
        print(f"Average Return: {overall['overall_avg_return']:+.2f}%")
        print(f"Best Strategy: {overall['best_strategy']}")
        print(f"Worst Strategy: {overall['worst_strategy']}")

        print("\n📈 STRATEGY BREAKDOWN:")
        print("-" * 60)

        for strategy_name, stats in performance_data['strategy_performance'].items():
            print(f"{strategy_name:<25} | "
                  f"Win Rate: {stats['win_rate']:>6.1%} | "
                  f"Avg Return: {stats['avg_return']:>+7.2f}% | "
                  f"Completed: {stats['completed_predictions']:>3}")

        print(f"\n✅ Performance data saved to: {self.performance_file}")


if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.print_performance_summary()