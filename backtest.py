"""
Backtesting system to validate strategy performance
Run this to see how strategies would have performed historically
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from predictors.strategies import get_all_strategies
from predictors.data_fetcher import StockDataFetcher
import json

class Backtester:
    def __init__(self, start_date='2022-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.fetcher = StockDataFetcher()
        self.strategies = get_all_strategies()
        self.results = {}
        
    def fetch_historical_data(self):
        """Fetch 2 years of historical data for backtesting"""
        print(f"Fetching historical data from {self.start_date} to {self.end_date}...")
        all_data = {}
        
        for symbol in self.fetcher.stocks:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date)
            if len(data) > 0:
                all_data[symbol] = data
                print(f"  {symbol}: {len(data)} days of data")
        
        return all_data
    
    def calculate_weekly_returns(self, data):
        """Calculate actual 1-week forward returns"""
        weekly_returns = []
        
        for i in range(len(data) - 5):  # Need 5 days ahead
            week_return = (data['Close'].iloc[i+5] - data['Close'].iloc[i]) / data['Close'].iloc[i]
            weekly_returns.append({
                'date': data.index[i],
                'return': week_return * 100  # Convert to percentage
            })
        
        return pd.DataFrame(weekly_returns)
    
    def backtest_strategy(self, strategy, stock_data):
        """Test a strategy on historical data"""
        predictions = []
        
        for symbol, data in stock_data.items():
            # Prepare features for each day
            features = self.fetcher.prepare_features(data)
            
            # Skip if not enough data
            if len(features) < 60:  # Need 60 days of history minimum
                continue
            
            # Calculate weekly returns
            weekly_returns = self.calculate_weekly_returns(features)
            
            # Make predictions for each week
            for i in range(60, len(features) - 5):  # Start after warmup, stop 5 days before end
                # Get features up to this point
                current_features = features.iloc[i].to_dict()
                current_features['symbol'] = symbol
                
                # Make prediction
                prediction = strategy.predict(current_features)
                
                if prediction['position'] != 'HOLD':
                    # Get actual return 1 week later
                    actual_return = weekly_returns.iloc[i-60]['return'] if i-60 < len(weekly_returns) else 0
                    
                    # Calculate if prediction was correct
                    if prediction['position'] == 'LONG':
                        correct = actual_return > 0
                        profit = actual_return
                    else:  # SHORT
                        correct = actual_return < 0
                        profit = -actual_return  # Profit from short position
                    
                    predictions.append({
                        'date': features.index[i],
                        'symbol': symbol,
                        'position': prediction['position'],
                        'confidence': prediction['confidence'],
                        'predicted_direction': 'UP' if prediction['position'] == 'LONG' else 'DOWN',
                        'actual_return': actual_return,
                        'correct': correct,
                        'profit': profit
                    })
        
        return pd.DataFrame(predictions)
    
    def run_backtest(self):
        """Run backtest for all strategies"""
        print("\n" + "="*60)
        print("BACKTESTING ALL STRATEGIES (1-WEEK PREDICTIONS)")
        print("="*60)
        
        # Fetch historical data
        stock_data = self.fetch_historical_data()
        
        results_summary = []
        
        for strategy in self.strategies:
            print(f"\nTesting {strategy.name}...")
            
            # Backtest the strategy
            predictions = self.backtest_strategy(strategy, stock_data)
            
            if len(predictions) > 0:
                # Calculate metrics
                total_predictions = len(predictions)
                correct_predictions = predictions['correct'].sum()
                accuracy = (correct_predictions / total_predictions) * 100
                
                # Calculate returns
                avg_profit = predictions['profit'].mean()
                total_return = predictions['profit'].sum()
                
                # Win/loss ratio
                winning_trades = predictions[predictions['profit'] > 0]
                losing_trades = predictions[predictions['profit'] <= 0]
                
                if len(losing_trades) > 0:
                    avg_win = winning_trades['profit'].mean() if len(winning_trades) > 0 else 0
                    avg_loss = abs(losing_trades['profit'].mean())
                    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
                else:
                    profit_factor = float('inf')
                
                # Sharpe ratio (simplified)
                if predictions['profit'].std() > 0:
                    sharpe = (avg_profit / predictions['profit'].std()) * np.sqrt(52)  # Annualized
                else:
                    sharpe = 0
                
                result = {
                    'Strategy': strategy.name,
                    'Total Predictions': total_predictions,
                    'Accuracy (%)': round(accuracy, 2),
                    'Avg Weekly Return (%)': round(avg_profit, 2),
                    'Total Return (%)': round(total_return / 100, 2),  # Divide by 100 for cumulative
                    'Profit Factor': round(profit_factor, 2),
                    'Sharpe Ratio': round(sharpe, 2),
                    'Win Rate (%)': round(len(winning_trades) / total_predictions * 100, 2)
                }
                
                results_summary.append(result)
                
                # Store detailed results
                self.results[strategy.name] = predictions
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('Sharpe Ratio', ascending=False)
        
        # Display results
        print("\n" + "="*60)
        print("BACKTEST RESULTS SUMMARY (2023-2024)")
        print("="*60)
        print(summary_df.to_string(index=False))
        
        # Save results
        summary_df.to_csv('backtest_results.csv', index=False)
        print("\n✅ Results saved to backtest_results.csv")
        
        # Save detailed results
        with open('backtest_detailed.json', 'w') as f:
            json.dump({
                'summary': summary_df.to_dict('records'),
                'test_period': f"{self.start_date} to {self.end_date}",
                'total_stocks': len(stock_data),
                'prediction_horizon': '1 week'
            }, f, indent=2, default=str)
        
        return summary_df
    
    def validate_best_strategies(self):
        """Identify and validate the best performing strategies"""
        summary = self.run_backtest()
        
        print("\n" + "="*60)
        print("RECOMMENDED STRATEGIES FOR LIVE TRADING")
        print("="*60)
        
        # Get top 3 strategies by Sharpe ratio
        top_strategies = summary.nlargest(3, 'Sharpe Ratio')
        
        for i, row in top_strategies.iterrows():
            print(f"\n{row['Strategy']}:")
            print(f"  • Win Rate: {row['Win Rate (%)']}%")
            print(f"  • Avg Weekly Return: {row['Avg Weekly Return (%)']}%")
            print(f"  • Sharpe Ratio: {row['Sharpe Ratio']}")
            print(f"  • Profit Factor: {row['Profit Factor']}")
        
        print("\n" + "="*60)
        print("IMPORTANT NOTES:")
        print("="*60)
        print("• These are BACKTEST results - past performance ≠ future results")
        print("• Strategies use technical indicators only, no fundamental analysis")
        print("• 1-week predictions have high uncertainty")
        print("• Transaction costs and slippage not included")
        print("• This is NOT financial advice - for educational purposes only")
        
        return top_strategies

if __name__ == "__main__":
    print("Starting backtest of all 10 strategies...")
    print("This will take a few minutes to download and process data...\n")
    
    backtester = Backtester(
        start_date='2023-01-01',
        end_date='2024-12-31'
    )
    
    # Run validation
    best_strategies = backtester.validate_best_strategies()
    
    print("\n🎯 Backtest complete! Check backtest_results.csv for details.")