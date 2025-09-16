"""
Performance Tracking System
Tracks real-time performance of all predictions and trades
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

class PerformanceTracker:
    """Real-time performance tracking system"""
    
    def __init__(self, db_path: str = "predictor.db"):
        """Initialize performance tracker"""
        self.db = DatabaseManager(db_path)
        self.cache = {}
        self.last_update = None
        
    def update_all_trades(self):
        """Update all active trades with current prices"""
        print("\nUpdating active trades...")
        
        active_trades = self.db.get_active_trades()
        
        if not active_trades:
            print("No active trades to update")
            return
            
        # Get unique tickers
        tickers = list(set([t['ticker'] for t in active_trades]))
        
        # Fetch current prices
        current_prices = self.fetch_current_prices(tickers)
        
        # Update each trade
        updated_count = 0
        for trade in active_trades:
            ticker = trade['ticker']
            
            if ticker not in current_prices:
                print(f"Could not get price for {ticker}")
                continue
                
            current_price = current_prices[ticker]
            entry_price = trade.get('entry_price')
            
            # If no entry price, set it now
            if not entry_price:
                self.db.update_trade(
                    trade['id'],
                    entry_price=current_price,
                    entry_date=datetime.now().strftime('%Y-%m-%d')
                )
                print(f"Set entry price for {ticker}: ${current_price:.2f}")
                continue
                
            # Calculate current P&L
            position = trade['position']
            if position == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * 100
            else:  # SHORT
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
                
            # Check stop loss (5% loss)
            if pnl_pct <= -5:
                self.close_trade(
                    trade['id'],
                    current_price,
                    'STOPPED',
                    'Stop loss triggered'
                )
                print(f"Stop loss triggered for {ticker}: {pnl_pct:.2f}%")
                
            # Check take profit (10% gain)
            elif pnl_pct >= 10:
                self.close_trade(
                    trade['id'],
                    current_price,
                    'CLOSED',
                    'Take profit triggered'
                )
                print(f"Take profit triggered for {ticker}: {pnl_pct:.2f}%")
                
            # Check holding period
            elif self.check_holding_period(trade):
                self.close_trade(
                    trade['id'],
                    current_price,
                    'CLOSED',
                    'Max holding period reached'
                )
                print(f"Max holding period reached for {ticker}: {pnl_pct:.2f}%")
                
            updated_count += 1
            
        print(f"Updated {updated_count} trades")
        self.last_update = datetime.now()
        
    def close_trade(self, trade_id: int, exit_price: float, status: str, reason: str):
        """Close a trade and record final P&L"""
        # Get trade details
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                "SELECT entry_price, shares FROM trades WHERE id = ?",
                (trade_id,)
            )
            trade = cursor.fetchone()
            
        if not trade:
            return
            
        entry_price = trade[0]
        shares = trade[1] or 100
        
        # Calculate P&L
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        
        # Update trade
        self.db.update_trade(
            trade_id,
            exit_date=datetime.now().strftime('%Y-%m-%d'),
            exit_price=exit_price,
            status=status,
            pnl=pnl,
            pnl_percent=pnl_pct,
            exit_reason=reason
        )
        
    def check_holding_period(self, trade: Dict) -> bool:
        """Check if trade has exceeded max holding period"""
        if not trade.get('entry_date'):
            return False
            
        entry_date = datetime.strptime(trade['entry_date'], '%Y-%m-%d')
        days_held = (datetime.now() - entry_date).days
        
        # Get strategy max hold days (default 7)
        return days_held >= 7
        
    def fetch_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Fetch current prices for multiple tickers"""
        prices = {}
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                price = info.get('currentPrice') or info.get('previousClose')
                
                if price:
                    prices[ticker] = price
                    
            except Exception as e:
                print(f"Error fetching price for {ticker}: {e}")
                
        return prices
        
    def calculate_strategy_performance(self, strategy: str = None):
        """Calculate performance metrics for strategies"""
        print("\nCalculating strategy performance...")
        
        # Get closed trades
        with self.db.get_connection() as conn:
            if strategy:
                query = """
                    SELECT t.*, s.name as strategy
                    FROM trades t
                    JOIN predictions p ON t.prediction_id = p.id
                    JOIN strategies s ON p.strategy_id = s.id
                    WHERE t.status IN ('CLOSED', 'STOPPED')
                    AND s.name = ?
                """
                cursor = conn.execute(query, (strategy,))
            else:
                query = """
                    SELECT t.*, s.name as strategy
                    FROM trades t
                    JOIN predictions p ON t.prediction_id = p.id
                    JOIN strategies s ON p.strategy_id = s.id
                    WHERE t.status IN ('CLOSED', 'STOPPED')
                """
                cursor = conn.execute(query)
                
            trades = [dict(row) for row in cursor.fetchall()]
            
        if not trades:
            print("No closed trades to analyze")
            return {}
            
        # Group by strategy
        strategy_metrics = {}
        df = pd.DataFrame(trades)
        
        for strat in df['strategy'].unique():
            strat_trades = df[df['strategy'] == strat]
            
            metrics = {
                'total_trades': len(strat_trades),
                'winning_trades': len(strat_trades[strat_trades['pnl_percent'] > 0]),
                'losing_trades': len(strat_trades[strat_trades['pnl_percent'] <= 0]),
                'total_pnl': strat_trades['pnl'].sum(),
                'avg_return': strat_trades['pnl_percent'].mean(),
                'best_trade': strat_trades['pnl_percent'].max(),
                'worst_trade': strat_trades['pnl_percent'].min(),
                'sharpe_ratio': self.calculate_sharpe(strat_trades['pnl_percent']),
                'win_rate': len(strat_trades[strat_trades['pnl_percent'] > 0]) / len(strat_trades) * 100
            }
            
            # Calculate profit factor
            wins = strat_trades[strat_trades['pnl'] > 0]['pnl'].sum()
            losses = abs(strat_trades[strat_trades['pnl'] < 0]['pnl'].sum())
            metrics['profit_factor'] = wins / losses if losses > 0 else float('inf')
            
            # Calculate max drawdown
            metrics['max_drawdown'] = self.calculate_max_drawdown(strat_trades)
            
            strategy_metrics[strat] = metrics
            
            # Save to database
            self.db.save_performance_metrics(
                strat,
                datetime.now().strftime('%Y-%m-%d'),
                metrics
            )
            
        return strategy_metrics
        
    def calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - (risk_free_rate * 100 / 252)  # Daily risk-free rate
        
        if returns.std() == 0:
            return 0
            
        return (excess_returns.mean() / returns.std()) * np.sqrt(252)
        
    def calculate_max_drawdown(self, trades: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if trades.empty:
            return 0
            
        # Sort by date
        trades = trades.sort_values('entry_date')
        
        # Calculate cumulative returns
        cumulative = (1 + trades['pnl_percent'] / 100).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        return drawdown.min()
        
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'strategies': {},
            'overall': {}
        }
        
        # Get strategy performance
        strategy_metrics = self.calculate_strategy_performance()
        
        # Overall metrics
        total_trades = sum(m['total_trades'] for m in strategy_metrics.values())
        total_wins = sum(m['winning_trades'] for m in strategy_metrics.values())
        total_pnl = sum(m['total_pnl'] for m in strategy_metrics.values())
        
        report['overall'] = {
            'total_trades': total_trades,
            'total_wins': total_wins,
            'total_losses': total_trades - total_wins,
            'overall_win_rate': (total_wins / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'active_trades': len(self.db.get_active_trades())
        }
        
        # Print summary
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Win Rate: {report['overall']['overall_win_rate']:.1f}%")
        print(f"  Total P&L: ${total_pnl:,.2f}")
        print(f"  Active Trades: {report['overall']['active_trades']}")
        
        print(f"\nSTRATEGY BREAKDOWN:")
        for strategy, metrics in strategy_metrics.items():
            report['strategies'][strategy] = metrics
            
            print(f"\n  {strategy.upper()}:")
            print(f"    Trades: {metrics['total_trades']}")
            print(f"    Win Rate: {metrics['win_rate']:.1f}%")
            print(f"    Avg Return: {metrics['avg_return']:.2f}%")
            print(f"    Total P&L: ${metrics['total_pnl']:,.2f}")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.2f}")
            print(f"    Max DD: {metrics['max_drawdown']:.2f}%")
            
        # Save report to file
        report_path = Path('reports') / f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"\nReport saved to {report_path}")
        print("="*60)
        
        return report
        
    def monitor_predictions(self):
        """Monitor recent predictions and create trades"""
        print("\nMonitoring predictions for trade creation...")
        
        # Get today's predictions
        today = datetime.now().strftime('%Y-%m-%d')
        predictions = self.db.get_predictions(date=today)
        
        if not predictions:
            print("No predictions for today")
            return
            
        # Check each prediction
        created_count = 0
        for pred in predictions:
            # Check if trade already exists
            with self.db.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT id FROM trades WHERE prediction_id = ?",
                    (pred['id'],)
                )
                if cursor.fetchone():
                    continue
                    
            # Only create trades for LONG/SHORT positions
            if pred['position'] in ['LONG', 'SHORT']:
                # High confidence threshold
                if pred['confidence'] >= 0.7:
                    trade_id = self.db.create_trade(
                        prediction_id=pred['id'],
                        status='ACTIVE'
                    )
                    created_count += 1
                    print(f"Created trade for {pred['ticker']} ({pred['strategy']}): {pred['position']}")
                    
        print(f"Created {created_count} new trades")
        
    def run_daily_update(self):
        """Run complete daily update cycle"""
        print("\n" + "#"*60)
        print(f"DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("#"*60)
        
        # Monitor predictions and create trades
        self.monitor_predictions()
        
        # Update all active trades
        self.update_all_trades()
        
        # Generate performance report
        self.generate_performance_report()
        
        # Get database stats
        stats = self.db.get_database_stats()
        print("\nDATABASE STATS:")
        for table, count in stats.items():
            if table != 'date_range':
                print(f"  {table}: {count} records")
                

def main():
    """Run performance tracking"""
    tracker = PerformanceTracker()
    
    # Run daily update
    tracker.run_daily_update()
    
if __name__ == "__main__":
    main()