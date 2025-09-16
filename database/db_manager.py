"""
Database Manager for Stock Predictor
Handles all database operations with SQLite
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from contextlib import contextmanager

class DatabaseManager:
    def __init__(self, db_path: str = "predictor.db"):
        """Initialize database manager"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database with schema"""
        schema_file = Path(__file__).parent / "schema.sql"

        with self.get_connection() as conn:
            # Read and execute schema
            if schema_file.exists():
                with open(schema_file, 'r') as f:
                    schema = f.read()
                conn.executescript(schema)
            else:
                # Fallback: create minimal schema
                self._create_minimal_schema(conn)

            conn.commit()
            print(f"Database initialized at {self.db_path}")

    def _create_minimal_schema(self, conn):
        """Create minimal schema if schema.sql not found"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                description TEXT
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS tickers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE NOT NULL,
                name TEXT
            )
        """)

    # ============ TICKER OPERATIONS ============

    def get_or_create_ticker(self, symbol: str, name: str = None) -> int:
        """Get ticker ID, creating if necessary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Try to get existing
            cursor.execute("SELECT id FROM tickers WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create new
            cursor.execute(
                "INSERT INTO tickers (symbol, name) VALUES (?, ?)",
                (symbol, name or symbol)
            )
            conn.commit()
            return cursor.lastrowid

    def get_ticker_id(self, symbol: str) -> Optional[int]:
        """Get ticker ID by symbol"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM tickers WHERE symbol = ?", (symbol,))
            result = cursor.fetchone()
            return result[0] if result else None

    # ============ STRATEGY OPERATIONS ============

    def get_or_create_strategy(self, name: str, description: str = None,
                               min_hold_days: int = 1, max_hold_days: int = 30) -> int:
        """Get strategy ID, creating if necessary"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Try to get existing
            cursor.execute("SELECT id FROM strategies WHERE name = ?", (name,))
            result = cursor.fetchone()

            if result:
                return result[0]

            # Create new
            cursor.execute("""
                INSERT INTO strategies (name, description, min_hold_days, max_hold_days)
                VALUES (?, ?, ?, ?)
            """, (name, description, min_hold_days, max_hold_days))
            conn.commit()
            return cursor.lastrowid

    # ============ PREDICTION OPERATIONS ============

    def save_prediction(self, date: str, strategy: str, ticker: str,
                       position: str, confidence: float, score: float = None,
                       target_1w: float = None, **kwargs) -> int:
        """Save a prediction to database"""
        strategy_id = self.get_or_create_strategy(strategy)
        ticker_id = self.get_or_create_ticker(ticker)

        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if prediction exists
            cursor.execute("""
                SELECT id FROM predictions
                WHERE date = ? AND strategy_id = ? AND ticker_id = ?
            """, (date, strategy_id, ticker_id))

            existing = cursor.fetchone()

            if existing:
                # Update existing
                cursor.execute("""
                    UPDATE predictions
                    SET position = ?, confidence = ?, score = ?, target_1w = ?,
                        created_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (position, confidence, score, target_1w, existing[0]))
                conn.commit()
                return existing[0]
            else:
                # Insert new
                cursor.execute("""
                    INSERT INTO predictions
                    (date, strategy_id, ticker_id, position, confidence, score, target_1w)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (date, strategy_id, ticker_id, position, confidence, score, target_1w))
                conn.commit()
                return cursor.lastrowid

    def get_predictions(self, date: str = None, strategy: str = None,
                       ticker: str = None) -> List[Dict]:
        """Get predictions with optional filters"""
        with self.get_connection() as conn:
            query = """
                SELECT
                    p.id, p.date, s.name as strategy, t.symbol as ticker,
                    p.position, p.confidence, p.score, p.target_1w,
                    p.created_at
                FROM predictions p
                JOIN strategies s ON p.strategy_id = s.id
                JOIN tickers t ON p.ticker_id = t.id
                WHERE 1=1
            """
            params = []

            if date:
                query += " AND p.date = ?"
                params.append(date)

            if strategy:
                query += " AND s.name = ?"
                params.append(strategy)

            if ticker:
                query += " AND t.symbol = ?"
                params.append(ticker)

            query += " ORDER BY p.date DESC, s.name"

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    # ============ TRADE OPERATIONS ============

    def create_trade(self, prediction_id: int, entry_date: str = None,
                    entry_price: float = None, status: str = 'PENDING') -> int:
        """Create a new trade from a prediction"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (prediction_id, entry_date, entry_price, status)
                VALUES (?, ?, ?, ?)
            """, (prediction_id, entry_date, entry_price, status))
            conn.commit()
            return cursor.lastrowid

    def update_trade(self, trade_id: int, **kwargs):
        """Update trade fields"""
        with self.get_connection() as conn:
            # Build update query dynamically
            fields = []
            values = []

            for field, value in kwargs.items():
                if field in ['exit_date', 'exit_price', 'status', 'pnl', 'pnl_percent', 'exit_reason']:
                    fields.append(f"{field} = ?")
                    values.append(value)

            if fields:
                fields.append("updated_at = CURRENT_TIMESTAMP")
                query = f"UPDATE trades SET {', '.join(fields)} WHERE id = ?"
                values.append(trade_id)
                conn.execute(query, values)
                conn.commit()

    def get_active_trades(self) -> List[Dict]:
        """Get all active trades"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    t.*, p.date as prediction_date, p.position, p.confidence,
                    s.name as strategy, tk.symbol as ticker
                FROM trades t
                JOIN predictions p ON t.prediction_id = p.id
                JOIN strategies s ON p.strategy_id = s.id
                JOIN tickers tk ON p.ticker_id = tk.id
                WHERE t.status = 'ACTIVE'
                ORDER BY t.entry_date DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    # ============ PRICE HISTORY OPERATIONS ============

    def save_price_history(self, ticker: str, date: str, close: float,
                          open: float = None, high: float = None,
                          low: float = None, volume: int = None):
        """Save price history data"""
        ticker_id = self.get_or_create_ticker(ticker)

        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO price_history
                (ticker_id, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker_id, date, open, high, low, close, volume))
            conn.commit()

    def get_price_history(self, ticker: str, start_date: str = None,
                         end_date: str = None) -> pd.DataFrame:
        """Get price history as DataFrame"""
        ticker_id = self.get_ticker_id(ticker)
        if not ticker_id:
            return pd.DataFrame()

        with self.get_connection() as conn:
            query = """
                SELECT date, open, high, low, close, volume
                FROM price_history
                WHERE ticker_id = ?
            """
            params = [ticker_id]

            if start_date:
                query += " AND date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND date <= ?"
                params.append(end_date)

            query += " ORDER BY date"

            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            return df

    # ============ PERFORMANCE OPERATIONS ============

    def save_performance_metrics(self, strategy: str, date: str, metrics: Dict):
        """Save performance metrics for a strategy"""
        strategy_id = self.get_or_create_strategy(strategy)

        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO performance_metrics
                (strategy_id, date, total_trades, winning_trades, losing_trades,
                 win_rate, avg_win, avg_loss, profit_factor, sharpe_ratio,
                 max_drawdown, total_return, total_pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, date,
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                metrics.get('win_rate'),
                metrics.get('avg_win'),
                metrics.get('avg_loss'),
                metrics.get('profit_factor'),
                metrics.get('sharpe_ratio'),
                metrics.get('max_drawdown'),
                metrics.get('total_return'),
                metrics.get('total_pnl')
            ))
            conn.commit()

    def get_strategy_performance(self, strategy: str = None) -> pd.DataFrame:
        """Get performance metrics for strategies"""
        with self.get_connection() as conn:
            query = """
                SELECT
                    s.name as strategy,
                    pm.*
                FROM performance_metrics pm
                JOIN strategies s ON pm.strategy_id = s.id
                WHERE 1=1
            """
            params = []

            if strategy:
                query += " AND s.name = ?"
                params.append(strategy)

            query += " ORDER BY pm.date DESC"

            return pd.read_sql_query(query, conn, params=params)

    # ============ VOLUME LEADERS OPERATIONS ============

    def save_volume_leaders(self, date: str, timeframe: str, leaders: List[Tuple[str, int, float]]):
        """Save volume leaders for a specific date and timeframe"""
        with self.get_connection() as conn:
            for rank, (ticker, volume, price) in enumerate(leaders, 1):
                ticker_id = self.get_or_create_ticker(ticker)

                conn.execute("""
                    INSERT OR REPLACE INTO volume_leaders
                    (date, timeframe, rank, ticker_id, avg_volume, avg_price)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (date, timeframe, rank, ticker_id, volume, price))

            conn.commit()

    def get_volume_leaders(self, date: str, timeframe: str = 'daily') -> List[str]:
        """Get volume leaders for a specific date"""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT t.symbol
                FROM volume_leaders vl
                JOIN tickers t ON vl.ticker_id = t.id
                WHERE vl.date = ? AND vl.timeframe = ?
                ORDER BY vl.rank
                LIMIT 10
            """, (date, timeframe))

            return [row[0] for row in cursor.fetchall()]

    # ============ BACKTEST OPERATIONS ============

    def save_backtest_result(self, strategy: str, start_date: str, end_date: str,
                            results: Dict) -> int:
        """Save backtest results"""
        strategy_id = self.get_or_create_strategy(strategy)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO backtest_results
                (strategy_id, start_date, end_date, initial_capital, final_capital,
                 total_return, annual_return, sharpe_ratio, max_drawdown,
                 total_trades, winning_trades, losing_trades, win_rate,
                 profit_factor, avg_trade_return, best_trade, worst_trade,
                 parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                strategy_id, start_date, end_date,
                results.get('initial_capital', 10000),
                results.get('final_capital'),
                results.get('total_return'),
                results.get('annual_return'),
                results.get('sharpe_ratio'),
                results.get('max_drawdown'),
                results.get('total_trades'),
                results.get('winning_trades'),
                results.get('losing_trades'),
                results.get('win_rate'),
                results.get('profit_factor'),
                results.get('avg_trade_return'),
                results.get('best_trade'),
                results.get('worst_trade'),
                json.dumps(results.get('parameters', {}))
            ))
            conn.commit()
            return cursor.lastrowid

    # ============ UTILITY OPERATIONS ============

    def backup_database(self, backup_path: str = None):
        """Create database backup"""
        if not backup_path:
            backup_path = f"{self.db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        with self.get_connection() as conn:
            backup = sqlite3.connect(backup_path)
            conn.backup(backup)
            backup.close()

        print(f"Database backed up to {backup_path}")
        return backup_path

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        with self.get_connection() as conn:
            stats = {}

            tables = ['strategies', 'tickers', 'predictions', 'trades',
                     'price_history', 'performance_metrics', 'volume_leaders',
                     'backtest_results']

            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            # Get date range
            cursor = conn.execute("SELECT MIN(date), MAX(date) FROM predictions")
            result = cursor.fetchone()
            if result[0]:
                stats['date_range'] = f"{result[0]} to {result[1]}"

            return stats

    def optimize_database(self):
        """Optimize database performance"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            print("Database optimized")


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    """Get database manager instance"""
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance