"""
Paper Trading System
Automatically executes and tracks paper trades based on predictions
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import json
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

class PaperTrader:
    """Automated paper trading system"""
    
    def __init__(self, initial_capital: float = 10000):
        self.db = DatabaseManager()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # Active positions
        self.trade_history = []  # Completed trades
        self.max_positions = 5
        self.position_size_pct = 0.2  # 20% per position
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.max_hold_days = 7
        
        # Initialize database table
        self._init_database()
        
        # Load existing positions
        self.load_positions()
        
    def _init_database(self):
        """Initialize paper trading tables"""
        with self.db.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    entry_date DATETIME NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    position_size REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    exit_date DATETIME,
                    exit_price REAL,
                    exit_reason TEXT,
                    commission REAL DEFAULT 0,
                    pnl REAL,
                    pnl_percent REAL,
                    status TEXT DEFAULT 'OPEN',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS position_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL,
                    date DATETIME NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    unrealized_pnl REAL,
                    unrealized_pnl_pct REAL,
                    FOREIGN KEY (trade_id) REFERENCES paper_trades(id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS paper_portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    total_value REAL NOT NULL,
                    daily_return REAL,
                    total_return REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date)
                )
            """)
            
            conn.commit()
            
    def load_positions(self):
        """Load active positions from database"""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM paper_trades WHERE status = 'OPEN'
            """)
            
            for row in cursor.fetchall():
                self.positions[row['symbol']] = dict(row)
                
        print(f"Loaded {len(self.positions)} active positions")
        
    def check_predictions(self):
        """Check for new predictions and create trades"""
        today = datetime.now().strftime('%Y-%m-%d')
        predictions = self.db.get_predictions(date=today)
        
        if not predictions:
            print("No predictions for today")
            return
            
        trades_created = 0
        
        for pred in predictions:
            # Skip if position already exists
            if pred['ticker'] in self.positions:
                continue
                
            # Skip HOLD signals
            if pred['position'] == 'HOLD':
                continue
                
            # Check if we have room for more positions
            if len(self.positions) >= self.max_positions:
                print(f"Maximum positions reached ({self.max_positions})")
                break
                
            # Only trade high confidence predictions
            if pred['confidence'] >= 0.6:
                if self.create_trade(pred):
                    trades_created += 1
                    
        print(f"Created {trades_created} new trades")
        
    def create_trade(self, prediction: Dict) -> bool:
        """Create a new paper trade"""
        try:
            symbol = prediction['ticker']
            
            # Get current price
            price = self.get_current_price(symbol)
            if not price:
                print(f"Could not get price for {symbol}")
                return False
                
            # Calculate position size
            position_value = self.current_capital * self.position_size_pct
            quantity = int(position_value / price)
            
            if quantity <= 0:
                print(f"Insufficient capital for {symbol}")
                return False
                
            # Calculate stop loss and take profit
            if prediction['position'] == 'LONG':
                stop_loss = price * (1 - self.stop_loss_pct)
                take_profit = price * (1 + self.take_profit_pct)
            else:  # SHORT
                stop_loss = price * (1 + self.stop_loss_pct)
                take_profit = price * (1 - self.take_profit_pct)
                
            # Save to database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO paper_trades (
                        prediction_id, symbol, strategy, entry_date, entry_price,
                        quantity, position_size, stop_loss, take_profit, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                """, (
                    prediction['id'],
                    symbol,
                    prediction['strategy'],
                    datetime.now(),
                    price,
                    quantity,
                    position_value,
                    stop_loss,
                    take_profit
                ))
                conn.commit()
                
                # Add to positions
                self.positions[symbol] = {
                    'id': cursor.lastrowid,
                    'symbol': symbol,
                    'strategy': prediction['strategy'],
                    'position': prediction['position'],
                    'entry_date': datetime.now(),
                    'entry_price': price,
                    'quantity': quantity,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                }
                
            # Update capital
            self.current_capital -= position_value
            
            print(f"TRADE OPENED: {symbol} - {prediction['position']} "
                  f"{quantity} shares @ ${price:.2f} "
                  f"(SL: ${stop_loss:.2f}, TP: ${take_profit:.2f})")
            
            return True
            
        except Exception as e:
            print(f"Error creating trade: {e}")
            return False
            
    def update_positions(self):
        """Update all open positions with current prices"""
        if not self.positions:
            print("No open positions to update")
            return
            
        print(f"\nUpdating {len(self.positions)} positions...")
        
        for symbol, position in list(self.positions.items()):
            current_price = self.get_current_price(symbol)
            
            if not current_price:
                continue
                
            # Calculate P&L
            if position.get('position') == 'LONG':
                pnl = (current_price - position['entry_price']) * position['quantity']
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:  # SHORT
                pnl = (position['entry_price'] - current_price) * position['quantity']
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']
                
            # Check exit conditions
            should_exit = False
            exit_reason = None
            
            # Stop loss
            if position.get('position') == 'LONG':
                if current_price <= position['stop_loss']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position['take_profit']:
                    should_exit = True
                    exit_reason = 'TAKE_PROFIT'
            else:  # SHORT
                if current_price >= position['stop_loss']:
                    should_exit = True
                    exit_reason = 'STOP_LOSS'
                elif current_price <= position['take_profit']:
                    should_exit = True
                    exit_reason = 'TAKE_PROFIT'
                    
            # Check holding period
            if isinstance(position['entry_date'], str):
                entry_date = datetime.strptime(position['entry_date'], '%Y-%m-%d %H:%M:%S')
            else:
                entry_date = position['entry_date']
                
            days_held = (datetime.now() - entry_date).days
            
            if days_held >= self.max_hold_days:
                should_exit = True
                exit_reason = 'MAX_HOLD_PERIOD'
                
            # Update position history
            with self.db.get_connection() as conn:
                conn.execute("""
                    INSERT INTO position_history (
                        trade_id, date, price, value, unrealized_pnl, unrealized_pnl_pct
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    position['id'],
                    datetime.now(),
                    current_price,
                    current_price * position['quantity'],
                    pnl,
                    pnl_pct * 100
                ))
                conn.commit()
                
            print(f"{symbol}: ${current_price:.2f} | "
                  f"P&L: ${pnl:.2f} ({pnl_pct*100:.1f}%) | "
                  f"Days: {days_held}")
            
            # Exit if needed
            if should_exit:
                self.close_position(symbol, current_price, exit_reason)
                
    def close_position(self, symbol: str, exit_price: float, exit_reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        
        # Calculate final P&L
        if position.get('position') == 'LONG':
            pnl = (exit_price - position['entry_price']) * position['quantity']
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            pnl = (position['entry_price'] - exit_price) * position['quantity']
            pnl_pct = (position['entry_price'] - exit_price) / position['entry_price']
            
        # Update database
        with self.db.get_connection() as conn:
            conn.execute("""
                UPDATE paper_trades
                SET exit_date = ?, exit_price = ?, exit_reason = ?,
                    pnl = ?, pnl_percent = ?, status = 'CLOSED'
                WHERE id = ?
            """, (
                datetime.now(),
                exit_price,
                exit_reason,
                pnl,
                pnl_pct * 100,
                position['id']
            ))
            conn.commit()
            
        # Return capital
        self.current_capital += exit_price * position['quantity']
        
        # Remove from positions
        del self.positions[symbol]
        
        result = "WIN" if pnl > 0 else "LOSS"
        print(f"TRADE CLOSED: {symbol} - {result} ${pnl:.2f} ({pnl_pct*100:.1f}%) - {exit_reason}")
        
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d', interval='1m')
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to daily data
                info = ticker.info
                return info.get('regularMarketPrice') or info.get('previousClose')
                
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None
            
    def update_portfolio_value(self):
        """Update portfolio value in database"""
        # Calculate total position value
        positions_value = 0
        
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            if current_price:
                positions_value += current_price * position['quantity']
                
        total_value = self.current_capital + positions_value
        daily_return = (total_value - self.initial_capital) / self.initial_capital * 100
        
        # Save to database
        with self.db.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO paper_portfolio (
                    date, cash, positions_value, total_value, daily_return, total_return
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime('%Y-%m-%d'),
                self.current_capital,
                positions_value,
                total_value,
                daily_return,
                daily_return  # Same as total for now
            ))
            conn.commit()
            
        print(f"\nPORTFOLIO VALUE: ${total_value:.2f} ({daily_return:+.1f}%)")
        print(f"  Cash: ${self.current_capital:.2f}")
        print(f"  Positions: ${positions_value:.2f}")
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        with self.db.get_connection() as conn:
            # Get closed trades
            cursor = conn.execute("""
                SELECT * FROM paper_trades WHERE status = 'CLOSED'
            """)
            closed_trades = cursor.fetchall()
            
            if not closed_trades:
                return {"message": "No closed trades yet"}
                
            # Calculate statistics
            total_trades = len(closed_trades)
            wins = sum(1 for t in closed_trades if t['pnl'] > 0)
            losses = total_trades - wins
            
            total_pnl = sum(t['pnl'] for t in closed_trades)
            avg_win = np.mean([t['pnl'] for t in closed_trades if t['pnl'] > 0]) if wins > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in closed_trades if t['pnl'] <= 0]) if losses > 0 else 0
            
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            
            # Profit factor
            gross_profit = sum(t['pnl'] for t in closed_trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in closed_trades if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            return {
                'total_trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'best_trade': max(t['pnl'] for t in closed_trades),
                'worst_trade': min(t['pnl'] for t in closed_trades)
            }
            
    def run_daily_update(self):
        """Run complete daily update cycle"""
        print("\n" + "="*60)
        print(f"PAPER TRADING UPDATE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        # Check for new trades
        self.check_predictions()
        
        # Update existing positions
        self.update_positions()
        
        # Update portfolio value
        self.update_portfolio_value()
        
        # Show performance stats
        stats = self.get_performance_stats()
        if 'message' not in stats:
            print(f"\nPERFORMANCE STATS:")
            print(f"  Total Trades: {stats['total_trades']}")
            print(f"  Win Rate: {stats['win_rate']:.1f}%")
            print(f"  Total P&L: ${stats['total_pnl']:.2f}")
            print(f"  Profit Factor: {stats['profit_factor']:.2f}")
            
        print("="*60)
        

def main():
    """Run paper trading system"""
    trader = PaperTrader(initial_capital=10000)
    trader.run_daily_update()
    
if __name__ == "__main__":
    main()