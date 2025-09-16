"""
Backtesting Engine for Stock Predictor
Historical performance analysis and strategy optimization
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import yfinance as yf
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

@dataclass
class Trade:
    """Trade data structure"""
    ticker: str
    entry_date: str
    entry_price: float
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    position: str = 'LONG'
    shares: int = 100
    pnl: float = 0
    pnl_percent: float = 0
    status: str = 'OPEN'
    strategy: str = ''
    
@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000
    position_size: float = 0.1  # 10% per position
    max_positions: int = 10
    stop_loss: float = -0.05  # -5%
    take_profit: float = 0.10  # +10%
    max_hold_days: int = 7
    commission: float = 0.001  # 0.1% per trade
    slippage: float = 0.001  # 0.1% slippage
    
class BacktestEngine:
    """Main backtesting engine"""
    
    def __init__(self, config: BacktestConfig = None):
        """Initialize backtesting engine"""
        self.config = config or BacktestConfig()
        self.db = DatabaseManager()
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.current_capital = self.config.initial_capital
        self.open_positions: Dict[str, Trade] = {}
        self.price_cache = {}
        
    def load_historical_prices(self, tickers: List[str], start_date: str, end_date: str):
        """Load historical price data"""
        print(f"Loading historical prices for {len(tickers)} tickers...")
        
        for ticker in tickers:
            try:
                # Try database first
                df = self.db.get_price_history(ticker, start_date, end_date)
                
                if df.empty:
                    # Fetch from Yahoo Finance
                    print(f"  Downloading {ticker} from Yahoo Finance...")
                    stock = yf.Ticker(ticker)
                    df = stock.history(start=start_date, end=end_date)
                    
                    # Save to database
                    for date, row in df.iterrows():
                        self.db.save_price_history(
                            ticker=ticker,
                            date=date.strftime('%Y-%m-%d'),
                            open=row['Open'],
                            high=row['High'],
                            low=row['Low'],
                            close=row['Close'],
                            volume=row['Volume']
                        )
                        
                self.price_cache[ticker] = df
                
            except Exception as e:
                print(f"  Error loading {ticker}: {e}")
                
        print(f"Loaded prices for {len(self.price_cache)} tickers")
        
    def get_price(self, ticker: str, date: str, price_type: str = 'close') -> Optional[float]:
        """Get historical price for a ticker on a specific date"""
        if ticker not in self.price_cache:
            return None
            
        df = self.price_cache[ticker]
        
        try:
            # Convert date string to datetime
            target_date = pd.to_datetime(date)
            
            # Find closest available date
            if target_date in df.index:
                return df.loc[target_date, price_type.capitalize()]
            else:
                # Get nearest date
                idx = df.index.get_indexer([target_date], method='nearest')[0]
                if idx >= 0 and idx < len(df):
                    return df.iloc[idx][price_type.capitalize()]
                    
        except Exception:
            pass
            
        return None
        
    def execute_trade(self, ticker: str, date: str, position: str, strategy: str, confidence: float):
        """Execute a trade based on signal"""
        # Check if we can open new position
        if len(self.open_positions) >= self.config.max_positions:
            return
            
        # Check if already have position in this ticker
        if ticker in self.open_positions:
            return
            
        # Get entry price
        entry_price = self.get_price(ticker, date, 'open')
        if not entry_price:
            return
            
        # Apply slippage
        if position == 'LONG':
            entry_price *= (1 + self.config.slippage)
        else:
            entry_price *= (1 - self.config.slippage)
            
        # Calculate position size
        position_value = self.current_capital * self.config.position_size
        shares = int(position_value / entry_price)
        
        if shares <= 0:
            return
            
        # Apply commission
        commission = position_value * self.config.commission
        self.current_capital -= commission
        
        # Create trade
        trade = Trade(
            ticker=ticker,
            entry_date=date,
            entry_price=entry_price,
            position=position,
            shares=shares,
            strategy=strategy
        )
        
        self.open_positions[ticker] = trade
        
        # Deduct from capital
        self.current_capital -= (shares * entry_price)
        
    def check_exit_conditions(self, trade: Trade, current_date: str, current_price: float) -> Tuple[bool, str]:
        """Check if trade should be exited"""
        # Calculate current P&L
        if trade.position == 'LONG':
            pnl_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            pnl_pct = (trade.entry_price - current_price) / trade.entry_price
            
        # Check stop loss
        if pnl_pct <= self.config.stop_loss:
            return True, 'stop_loss'
            
        # Check take profit
        if pnl_pct >= self.config.take_profit:
            return True, 'take_profit'
            
        # Check max holding period
        entry_date = pd.to_datetime(trade.entry_date)
        current_date_dt = pd.to_datetime(current_date)
        days_held = (current_date_dt - entry_date).days
        
        if days_held >= self.config.max_hold_days:
            return True, 'max_hold_days'
            
        return False, ''
        
    def close_trade(self, trade: Trade, exit_date: str, exit_price: float, reason: str):
        """Close an open trade"""
        # Apply slippage
        if trade.position == 'LONG':
            exit_price *= (1 - self.config.slippage)
        else:
            exit_price *= (1 + self.config.slippage)
            
        # Calculate P&L
        if trade.position == 'LONG':
            pnl = (exit_price - trade.entry_price) * trade.shares
            pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
        else:
            pnl = (trade.entry_price - exit_price) * trade.shares
            pnl_pct = (trade.entry_price - exit_price) / trade.entry_price
            
        # Apply commission
        commission = trade.shares * exit_price * self.config.commission
        pnl -= commission
        
        # Update trade
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.pnl_percent = pnl_pct * 100
        trade.status = 'CLOSED'
        
        # Add to capital
        self.current_capital += (trade.shares * exit_price - commission)
        
        # Record trade
        self.trades.append(trade)
        
        # Remove from open positions
        del self.open_positions[trade.ticker]
        
    def update_positions(self, date: str):
        """Update all open positions"""
        positions_to_close = []
        
        for ticker, trade in self.open_positions.items():
            current_price = self.get_price(ticker, date, 'close')
            
            if not current_price:
                continue
                
            should_exit, reason = self.check_exit_conditions(trade, date, current_price)
            
            if should_exit:
                positions_to_close.append((trade, current_price, reason))
                
        # Close positions
        for trade, exit_price, reason in positions_to_close:
            self.close_trade(trade, date, exit_price, reason)
            
    def run_backtest(self, start_date: str, end_date: str, strategy: str = None):
        """Run backtest for a period"""
        print(f"\nRunning backtest from {start_date} to {end_date}")
        
        # Get all predictions in date range
        predictions = self.db.get_predictions()
        
        # Filter by date range
        df_pred = pd.DataFrame(predictions)
        if df_pred.empty:
            print("No predictions found")
            return {}
            
        df_pred['date'] = pd.to_datetime(df_pred['date'])
        mask = (df_pred['date'] >= start_date) & (df_pred['date'] <= end_date)
        df_pred = df_pred[mask]
        
        # Filter by strategy if specified
        if strategy:
            df_pred = df_pred[df_pred['strategy'] == strategy]
            
        # Get unique tickers
        tickers = df_pred['ticker'].unique().tolist()
        
        # Load historical prices
        self.load_historical_prices(tickers, start_date, end_date)
        
        # Sort predictions by date
        df_pred = df_pred.sort_values('date')
        
        # Track daily equity
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            
            # Update open positions
            self.update_positions(date_str)
            
            # Get predictions for this date
            day_predictions = df_pred[df_pred['date'] == date]
            
            # Execute new trades
            for _, pred in day_predictions.iterrows():
                if pred['position'] in ['LONG', 'SHORT'] and pred['confidence'] >= 0.6:
                    self.execute_trade(
                        ticker=pred['ticker'],
                        date=date_str,
                        position=pred['position'],
                        strategy=pred['strategy'],
                        confidence=pred['confidence']
                    )
                    
            # Calculate portfolio value
            portfolio_value = self.current_capital
            
            for ticker, trade in self.open_positions.items():
                current_price = self.get_price(ticker, date_str, 'close')
                if current_price:
                    portfolio_value += trade.shares * current_price
                    
            self.equity_curve.append({
                'date': date_str,
                'value': portfolio_value,
                'capital': self.current_capital,
                'positions': len(self.open_positions)
            })
            
        # Close any remaining positions
        for ticker, trade in list(self.open_positions.items()):
            last_price = self.get_price(ticker, end_date, 'close')
            if last_price:
                self.close_trade(trade, end_date, last_price, 'end_of_backtest')
                
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Save results to database
        if strategy:
            self.db.save_backtest_result(strategy, start_date, end_date, metrics)
            
        return metrics
        
    def calculate_metrics(self) -> Dict[str, Any]:
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
            
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl <= 0])
        
        # Calculate returns
        final_value = self.equity_curve[-1]['value'] if self.equity_curve else self.config.initial_capital
        total_return = ((final_value - self.config.initial_capital) / self.config.initial_capital) * 100
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Average trade
        avg_win = np.mean([t.pnl_percent for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl_percent for t in self.trades if t.pnl <= 0]) if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in self.trades if t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in self.trades if t.pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio
        if len(self.equity_curve) > 1:
            returns = pd.Series([self.equity_curve[i]['value'] / self.equity_curve[i-1]['value'] - 1
                               for i in range(1, len(self.equity_curve))])
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0
            
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        # Best and worst trades
        best_trade = max([t.pnl_percent for t in self.trades]) if self.trades else 0
        worst_trade = min([t.pnl_percent for t in self.trades]) if self.trades else 0
        
        metrics = {
            'initial_capital': self.config.initial_capital,
            'final_capital': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'total_pnl': sum([t.pnl for t in self.trades]),
            'avg_trade_return': np.mean([t.pnl_percent for t in self.trades]) if self.trades else 0
        }
        
        return metrics
        
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve"""
        if len(self.equity_curve) < 2:
            return 0
            
        values = [e['value'] for e in self.equity_curve]
        peak = values[0]
        max_dd = 0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                max_dd = max(max_dd, dd)
                
        return max_dd
        
    def print_results(self, metrics: Dict[str, Any]):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"\nCAPITAL:")
        print(f"  Initial: ${metrics['initial_capital']:,.2f}")
        print(f"  Final: ${metrics['final_capital']:,.2f}")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:,.2f}")
        
        print(f"\nTRADES:")
        print(f"  Total: {metrics['total_trades']}")
        print(f"  Winners: {metrics['winning_trades']}")
        print(f"  Losers: {metrics['losing_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}%")
        
        print(f"\nPERFORMANCE:")
        print(f"  Avg Win: {metrics['avg_win']:.2f}%")
        print(f"  Avg Loss: {metrics['avg_loss']:.2f}%")
        print(f"  Best Trade: {metrics['best_trade']:.2f}%")
        print(f"  Worst Trade: {metrics['worst_trade']:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2f}%")
        
        print("="*60)
        
    def save_trades_to_csv(self, filepath: str):
        """Save trade history to CSV"""
        if not self.trades:
            print("No trades to save")
            return
            
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'ticker': trade.ticker,
                'strategy': trade.strategy,
                'position': trade.position,
                'entry_date': trade.entry_date,
                'entry_price': trade.entry_price,
                'exit_date': trade.exit_date,
                'exit_price': trade.exit_price,
                'shares': trade.shares,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'status': trade.status
            })
            
        df = pd.DataFrame(trades_data)
        df.to_csv(filepath, index=False)
        print(f"Trades saved to {filepath}")
        

def run_strategy_comparison():
    """Run backtest for all strategies and compare"""
    print("\n" + "#"*60)
    print("STRATEGY COMPARISON BACKTEST")
    print("#"*60)
    
    # Get all strategies
    db = DatabaseManager()
    with db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM strategies")
        strategies = [row[0] for row in cursor.fetchall()]
        
    # Backtest period
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    results = {}
    
    for strategy in strategies:
        print(f"\nBacktesting {strategy}...")
        
        # Create new engine for each strategy
        engine = BacktestEngine()
        metrics = engine.run_backtest(start_date, end_date, strategy)
        
        if metrics['total_trades'] > 0:
            engine.print_results(metrics)
            results[strategy] = metrics
            
    # Compare strategies
    if results:
        print("\n" + "="*60)
        print("STRATEGY COMPARISON")
        print("="*60)
        
        # Sort by total return
        sorted_strategies = sorted(results.items(), key=lambda x: x[1]['total_return'], reverse=True)
        
        print(f"\n{'Strategy':<25} {'Return':<10} {'Sharpe':<10} {'Win Rate':<10} {'Trades':<10}")
        print("-"*65)
        
        for strategy, metrics in sorted_strategies:
            print(f"{strategy:<25} {metrics['total_return']:>9.2f}% {metrics['sharpe_ratio']:>9.2f} "
                  f"{metrics['win_rate']:>9.1f}% {metrics['total_trades']:>9}")
                  
    return results
    

def main():
    """Run backtesting"""
    # Run strategy comparison
    results = run_strategy_comparison()
    
    # Save results
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(output_dir / f'backtest_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"\nResults saved to backtest_results/backtest_{timestamp}.json")
    
if __name__ == "__main__":
    main()