"""
Comprehensive backtesting system for stock prediction strategies
Supports walk-forward analysis, multiple timeframes, and detailed metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from typing import List, Dict, Optional, Tuple, Any
import json
import sys
from pathlib import Path
from dataclasses import dataclass
import sqlite3

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.random_forest_predictor import RandomForestPredictor
from models.sequence_predictor import SequencePredictor
from indicators.technical_indicators import TechnicalIndicators

@dataclass
class Trade:
    """Represents a single trade"""
    symbol: str
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    quantity: int
    side: str  # 'long' or 'short'
    strategy: str
    pnl: float
    pnl_percent: float
    exit_reason: str
    hold_days: int

@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    trades: List[Trade]
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    avg_hold_days: float
    equity_curve: pd.DataFrame
    monthly_returns: pd.DataFrame
    drawdown_curve: pd.Series

class Backtester:
    """Comprehensive backtesting engine"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,   # 0.05% slippage
                 risk_per_trade: float = 0.02):  # 2% risk per trade
        """
        Initialize backtester

        Args:
            initial_capital: Starting capital
            commission: Commission rate (as decimal)
            slippage: Slippage rate (as decimal)
            risk_per_trade: Maximum risk per trade (as decimal)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_per_trade = risk_per_trade

        self.indicators = TechnicalIndicators()

        # Results storage
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.positions = {}  # symbol -> position info

        # Database for storing results
        self.db_path = Path('backtesting/backtest_results.db')
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for storing backtest results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT UNIQUE,
                strategy TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                annual_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                total_trades INTEGER,
                created_at TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_name TEXT,
                symbol TEXT,
                entry_date TEXT,
                exit_date TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                side TEXT,
                strategy TEXT,
                pnl REAL,
                pnl_percent REAL,
                exit_reason TEXT,
                hold_days INTEGER,
                FOREIGN KEY (run_name) REFERENCES backtest_runs (run_name)
            )
        """)

        conn.commit()
        conn.close()

    def run_strategy_backtest(self,
                            strategy_name: str,
                            symbols: List[str],
                            start_date: datetime,
                            end_date: datetime,
                            predictor_type: str = 'random_forest',
                            walk_forward: bool = True,
                            rebalance_frequency: str = 'daily') -> BacktestResults:
        """
        Run backtest for a strategy

        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols to trade
            start_date: Backtest start date
            end_date: Backtest end date
            predictor_type: 'random_forest' or 'sequence'
            walk_forward: Use walk-forward analysis
            rebalance_frequency: 'daily', 'weekly', 'monthly'
        """

        print(f"Starting backtest: {strategy_name}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Symbols: {symbols}")
        print(f"Predictor: {predictor_type}")

        # Initialize predictor
        if predictor_type == 'random_forest':
            predictor = RandomForestPredictor()
        else:
            predictor = SequencePredictor(model_type='gradient_boosting')

        # Reset state
        self.trades = []
        self.equity_curve = []
        self.positions = {}
        current_capital = self.initial_capital

        # Get historical data for all symbols
        print("Fetching historical data...")
        historical_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date - timedelta(days=365), end=end_date)
                if len(data) > 100:
                    historical_data[symbol] = data
                    print(f"  {symbol}: {len(data)} days")
                else:
                    print(f"  {symbol}: Insufficient data")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")

        if not historical_data:
            raise ValueError("No historical data available")

        # Generate trading dates
        trading_dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days

        print(f"Backtesting {len(trading_dates)} trading days...")

        # Walk-forward or fixed window training
        training_window = 252  # 1 year
        prediction_cache = {}

        for i, current_date in enumerate(trading_dates):
            if i % 50 == 0:
                print(f"Processing {current_date.date()}, Progress: {i/len(trading_dates)*100:.1f}%")

            # Update positions and calculate equity
            current_capital = self._update_positions(current_date, historical_data)
            self.equity_curve.append({
                'date': current_date,
                'equity': current_capital,
                'cash': current_capital - sum([pos['value'] for pos in self.positions.values()])
            })

            # Skip if not enough historical data
            if i < training_window:
                continue

            # Retrain model periodically (walk-forward)
            retrain_frequency = 21  # Retrain every month
            if walk_forward and (i % retrain_frequency == 0 or i == training_window):
                print(f"  Retraining model at {current_date.date()}")

                # Get training data up to current date
                train_end = current_date - timedelta(days=1)
                train_start = train_end - timedelta(days=500)

                try:
                    predictor.train(list(historical_data.keys())[:5], lookback_days=500)
                    prediction_cache = {}  # Clear cache after retraining
                except Exception as e:
                    print(f"  Training failed: {e}")
                    continue

            # Generate predictions
            signals = self._generate_signals(current_date, historical_data, predictor, prediction_cache)

            # Execute trades based on signals
            current_capital = self._execute_trades(current_date, signals, historical_data, current_capital)

        # Calculate final results
        print("Calculating results...")
        results = self._calculate_results(strategy_name, start_date, end_date)

        # Save to database
        self._save_results_to_db(strategy_name, results, predictor_type)

        print(f"Backtest completed!")
        print(f"Total Return: {results.total_return:.2%}")
        print(f"Annual Return: {results.annual_return:.2%}")
        print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {results.max_drawdown:.2%}")
        print(f"Win Rate: {results.win_rate:.1%}")
        print(f"Total Trades: {results.total_trades}")

        return results

    def _update_positions(self, current_date: datetime, historical_data: Dict) -> float:
        """Update position values and handle exits"""
        total_value = 0
        positions_to_remove = []

        for symbol, position in self.positions.items():
            if symbol not in historical_data:
                continue

            # Get current price
            symbol_data = historical_data[symbol]
            price_data = symbol_data[symbol_data.index.date <= current_date.date()]

            if len(price_data) == 0:
                continue

            current_price = price_data['Close'].iloc[-1]
            position['current_price'] = current_price
            position['value'] = position['quantity'] * current_price
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            position['days_held'] = (current_date - position['entry_date']).days

            total_value += position['value']

            # Check exit conditions
            exit_reason = self._check_exit_conditions(position, current_price, current_date)
            if exit_reason:
                self._close_position(symbol, position, current_price, current_date, exit_reason)
                positions_to_remove.append(symbol)

        # Remove closed positions
        for symbol in positions_to_remove:
            del self.positions[symbol]

        return total_value + self._get_cash_balance()

    def _generate_signals(self,
                         current_date: datetime,
                         historical_data: Dict,
                         predictor,
                         cache: Dict) -> Dict[str, Dict]:
        """Generate trading signals for current date"""
        signals = {}

        for symbol in historical_data.keys():
            if symbol in cache and (current_date - cache[symbol]['date']).days < 1:
                signals[symbol] = cache[symbol]['signal']
                continue

            try:
                # Get prediction
                prediction = predictor.predict(symbol)

                if prediction and prediction['action'] in ['BUY', 'SELL']:
                    signals[symbol] = {
                        'action': prediction['action'],
                        'confidence': prediction['confidence'],
                        'expected_return': prediction['expected_return']
                    }

                    cache[symbol] = {
                        'date': current_date,
                        'signal': signals[symbol]
                    }

            except Exception as e:
                continue

        return signals

    def _execute_trades(self,
                       current_date: datetime,
                       signals: Dict,
                       historical_data: Dict,
                       current_capital: float) -> float:
        """Execute trades based on signals"""

        for symbol, signal in signals.items():
            if symbol in self.positions:  # Already have position
                continue

            if signal['action'] != 'BUY':  # Only handle long positions for now
                continue

            if signal['confidence'] < 0.6:  # Minimum confidence threshold
                continue

            # Get current price
            symbol_data = historical_data[symbol]
            price_data = symbol_data[symbol_data.index.date <= current_date.date()]

            if len(price_data) == 0:
                continue

            current_price = price_data['Close'].iloc[-1]

            # Calculate position size
            risk_amount = current_capital * self.risk_per_trade
            stop_loss_pct = 0.05  # 5% stop loss
            position_size = risk_amount / (current_price * stop_loss_pct)

            # Apply slippage
            entry_price = current_price * (1 + self.slippage)
            total_cost = position_size * entry_price * (1 + self.commission)

            if total_cost > current_capital * 0.1:  # Max 10% per position
                position_size = (current_capital * 0.1) / entry_price / (1 + self.commission)

            if position_size < 1:  # Minimum 1 share
                continue

            # Open position
            self.positions[symbol] = {
                'entry_date': current_date,
                'entry_price': entry_price,
                'quantity': int(position_size),
                'stop_loss': entry_price * (1 - stop_loss_pct),
                'take_profit': entry_price * (1 + 0.10),  # 10% take profit
                'strategy': 'ML_Prediction',
                'signal_confidence': signal['confidence']
            }

        return current_capital

    def _check_exit_conditions(self, position: Dict, current_price: float, current_date: datetime) -> Optional[str]:
        """Check if position should be exited"""

        # Stop loss
        if current_price <= position['stop_loss']:
            return 'STOP_LOSS'

        # Take profit
        if current_price >= position['take_profit']:
            return 'TAKE_PROFIT'

        # Time-based exit (max 10 days)
        if position['days_held'] >= 10:
            return 'TIME_LIMIT'

        return None

    def _close_position(self, symbol: str, position: Dict, exit_price: float, exit_date: datetime, exit_reason: str):
        """Close a position and record the trade"""

        # Apply slippage and commission
        actual_exit_price = exit_price * (1 - self.slippage)
        pnl = (actual_exit_price - position['entry_price']) * position['quantity']
        pnl -= (position['entry_price'] + actual_exit_price) * position['quantity'] * self.commission

        pnl_percent = pnl / (position['entry_price'] * position['quantity'])

        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=exit_date,
            entry_price=position['entry_price'],
            exit_price=actual_exit_price,
            quantity=position['quantity'],
            side='long',
            strategy=position['strategy'],
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            hold_days=position['days_held']
        )

        self.trades.append(trade)

    def _get_cash_balance(self) -> float:
        """Calculate current cash balance"""
        total_invested = sum([pos['quantity'] * pos['entry_price'] for pos in self.positions.values()])
        return self.initial_capital - total_invested

    def _calculate_results(self, strategy_name: str, start_date: datetime, end_date: datetime) -> BacktestResults:
        """Calculate comprehensive backtest results"""

        if not self.equity_curve:
            raise ValueError("No equity curve data available")

        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('date', inplace=True)

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod()

        # Total and annual returns
        total_return = (equity_df['equity'].iloc[-1] - self.initial_capital) / self.initial_capital
        days = (end_date - start_date).days
        annual_return = (1 + total_return) ** (365.25 / days) - 1

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_returns = equity_df['returns'] - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Drawdown
        rolling_max = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Trade statistics
        if self.trades:
            winning_trades = [t for t in self.trades if t.pnl > 0]
            losing_trades = [t for t in self.trades if t.pnl <= 0]

            win_rate = len(winning_trades) / len(self.trades)
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss < 0 else 0

            best_trade = max([t.pnl for t in self.trades])
            worst_trade = min([t.pnl for t in self.trades])
            avg_hold_days = np.mean([t.hold_days for t in self.trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            best_trade = 0
            worst_trade = 0
            avg_hold_days = 0

        # Monthly returns
        monthly_returns = equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)

        return BacktestResults(
            trades=self.trades,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_hold_days=avg_hold_days,
            equity_curve=equity_df,
            monthly_returns=monthly_returns,
            drawdown_curve=drawdown
        )

    def _save_results_to_db(self, strategy_name: str, results: BacktestResults, predictor_type: str):
        """Save backtest results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        run_name = f"{strategy_name}_{predictor_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save run summary
        cursor.execute("""
            INSERT OR REPLACE INTO backtest_runs
            (run_name, strategy, start_date, end_date, initial_capital, final_capital,
             total_return, annual_return, sharpe_ratio, max_drawdown, win_rate,
             profit_factor, total_trades, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_name, strategy_name,
            results.equity_curve.index[0].isoformat(),
            results.equity_curve.index[-1].isoformat(),
            self.initial_capital,
            results.equity_curve['equity'].iloc[-1],
            results.total_return, results.annual_return, results.sharpe_ratio,
            results.max_drawdown, results.win_rate, results.profit_factor,
            results.total_trades, datetime.now().isoformat()
        ))

        # Save individual trades
        for trade in results.trades:
            cursor.execute("""
                INSERT INTO backtest_trades
                (run_name, symbol, entry_date, exit_date, entry_price, exit_price,
                 quantity, side, strategy, pnl, pnl_percent, exit_reason, hold_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_name, trade.symbol, trade.entry_date.isoformat(),
                trade.exit_date.isoformat(), trade.entry_price, trade.exit_price,
                trade.quantity, trade.side, trade.strategy, trade.pnl,
                trade.pnl_percent, trade.exit_reason, trade.hold_days
            ))

        conn.commit()
        conn.close()

        print(f"Results saved to database with run_name: {run_name}")

def main():
    """Example backtest"""
    backtester = Backtester(initial_capital=100000)

    # Define test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY']
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 1, 1)

    # Run backtest
    results = backtester.run_strategy_backtest(
        strategy_name='ML_Momentum',
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        predictor_type='random_forest',
        walk_forward=True
    )

    # Print detailed results
    print(f"\n=== BACKTEST RESULTS ===")
    print(f"Strategy: ML_Momentum")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Capital: ${backtester.initial_capital:,.0f}")
    print(f"Final Capital: ${results.equity_curve['equity'].iloc[-1]:,.0f}")
    print(f"Total Return: {results.total_return:.2%}")
    print(f"Annual Return: {results.annual_return:.2%}")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {results.max_drawdown:.2%}")
    print(f"Win Rate: {results.win_rate:.1%}")
    print(f"Profit Factor: {results.profit_factor:.2f}")
    print(f"Total Trades: {results.total_trades}")
    print(f"Average Win: ${results.avg_win:.2f}")
    print(f"Average Loss: ${results.avg_loss:.2f}")
    print(f"Best Trade: ${results.best_trade:.2f}")
    print(f"Worst Trade: ${results.worst_trade:.2f}")
    print(f"Average Hold Days: {results.avg_hold_days:.1f}")

if __name__ == "__main__":
    main()