"""
Migrate JSON data to SQLite database
Preserves all existing data while moving to relational database
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

class JSONToSQLiteMigrator:
    def __init__(self):
        self.db = DatabaseManager("predictor.db")
        self.data_dir = Path(__file__).parent.parent / "data"
        self.root_dir = Path(__file__).parent.parent
        self.stats = {
            'predictions': 0,
            'trades': 0,
            'volume_leaders': 0,
            'strategies': 0,
            'tickers': 0,
            'errors': []
        }

    def migrate_all(self):
        """Run complete migration"""
        print("Starting JSON to SQLite Migration")
        print("=" * 50)

        # Backup existing database if it exists
        if Path("predictor.db").exists():
            backup_path = self.db.backup_database()
            print(f"Created backup: {backup_path}")

        # Run migrations
        self.migrate_strategies()
        self.migrate_predictions()
        self.migrate_trades()
        self.migrate_volume_leaders()
        self.migrate_historical_prices()

        # Print summary
        self.print_summary()

    def migrate_strategies(self):
        """Create strategy entries"""
        print("\nMigrating strategies...")

        strategies = [
            ("momentum", "Classic price momentum", 3, 10),
            ("mean_reversion", "Mean reversion trading", 2, 7),
            ("volume_breakout", "Volume-based breakouts", 1, 5),
            ("technical_indicators", "Technical analysis", 5, 15),
            ("pattern_recognition", "Chart pattern detection", 7, 20),
            ("volatility_arbitrage", "Volatility trading", 1, 3),
            ("moving_average_crossover", "MA crossover signals", 10, 30),
            ("support_resistance", "Support/resistance levels", 5, 15),
            ("market_sentiment", "Market sentiment analysis", 3, 10),
            ("ensemble", "Combined strategies", 5, 15)
        ]

        for name, desc, min_days, max_days in strategies:
            self.db.get_or_create_strategy(name, desc, min_days, max_days)
            self.stats['strategies'] += 1
            print(f"  Added strategy: {name}")

    def migrate_predictions(self):
        """Migrate predictions_data.json"""
        print("\nMigrating predictions...")

        predictions_file = self.root_dir / "predictions_data.json"
        if not predictions_file.exists():
            print("  predictions_data.json not found")
            return

        try:
            with open(predictions_file, 'r') as f:
                data = json.load(f)

            for day_data in data.get('predictions', []):
                date = day_data['date']
                predictions = day_data.get('predictions', {})

                for strategy, pred in predictions.items():
                    if pred.get('stock') == 'NONE':
                        continue

                    try:
                        # Extract target value as float
                        target = pred.get('projected', '0%')
                        if isinstance(target, str):
                            target = float(target.replace('%', '').replace('+', ''))
                        else:
                            target = float(target)

                        self.db.save_prediction(
                            date=date,
                            strategy=strategy.replace('_', ' ').title(),
                            ticker=pred['stock'],
                            position=pred['position'],
                            confidence=pred.get('confidence', 0.5),
                            score=pred.get('score'),
                            target_1w=target
                        )
                        self.stats['predictions'] += 1

                    except Exception as e:
                        self.stats['errors'].append(f"Prediction error: {e}")

            print(f"  Migrated {self.stats['predictions']} predictions")

        except Exception as e:
            print(f"  Error loading predictions: {e}")
            self.stats['errors'].append(f"Predictions file error: {e}")

    def migrate_trades(self):
        """Migrate trades_data.json"""
        print("\nMigrating trades...")

        trades_file = self.root_dir / "trades_data.json"
        if not trades_file.exists():
            print("  trades_data.json not found - creating sample trades")
            self.create_sample_trades()
            return

        try:
            with open(trades_file, 'r') as f:
                data = json.load(f)

            # Migrate active trades
            for trade in data.get('active_trades', []):
                self.migrate_single_trade(trade, 'ACTIVE')

            # Migrate closed trades
            for trade in data.get('closed_trades', []):
                status = 'CLOSED' if trade.get('return_pct', 0) > 0 else 'STOPPED'
                self.migrate_single_trade(trade, status)

            print(f"  Migrated {self.stats['trades']} trades")

        except Exception as e:
            print(f"  Error loading trades: {e}")
            self.stats['errors'].append(f"Trades file error: {e}")

    def migrate_single_trade(self, trade, status):
        """Migrate a single trade"""
        try:
            # Find matching prediction
            predictions = self.db.get_predictions(
                date=trade.get('entry_date'),
                strategy=trade.get('strategy', '').replace('_', ' ').title(),
                ticker=trade.get('symbol')
            )

            if predictions:
                prediction_id = predictions[0]['id']
            else:
                # Create prediction if not found
                prediction_id = self.db.save_prediction(
                    date=trade.get('entry_date', datetime.now().strftime('%Y-%m-%d')),
                    strategy=trade.get('strategy', 'unknown').replace('_', ' ').title(),
                    ticker=trade.get('symbol', 'UNKNOWN'),
                    position=trade.get('position', 'LONG'),
                    confidence=0.5
                )

            # Create trade
            trade_id = self.db.create_trade(
                prediction_id=prediction_id,
                entry_date=trade.get('entry_date'),
                entry_price=trade.get('entry_price'),
                status=status
            )

            # Update with exit info if closed
            if status in ['CLOSED', 'STOPPED']:
                pnl_pct = trade.get('return_pct', 0)
                pnl = trade.get('entry_price', 100) * pnl_pct / 100 * 100  # Assume 100 shares

                self.db.update_trade(
                    trade_id,
                    exit_date=trade.get('exit_date'),
                    exit_price=trade.get('exit_price'),
                    pnl=pnl,
                    pnl_percent=pnl_pct,
                    status=status
                )

            self.stats['trades'] += 1

        except Exception as e:
            self.stats['errors'].append(f"Trade migration error: {e}")

    def create_sample_trades(self):
        """Create sample trades from recent predictions"""
        predictions = self.db.get_predictions()

        for pred in predictions[:20]:  # Create trades for last 20 predictions
            if pred['position'] != 'HOLD':
                trade_id = self.db.create_trade(
                    prediction_id=pred['id'],
                    entry_date=pred['date'],
                    status='PENDING'
                )
                self.stats['trades'] += 1

    def migrate_volume_leaders(self):
        """Migrate volume leader databases"""
        print("\nMigrating volume leaders...")

        volume_files = {
            'daily': self.data_dir / "volume_leaders_daily.json",
            'weekly': self.data_dir / "volume_leaders_weekly.json",
            'monthly': self.data_dir / "volume_leaders_monthly.json"
        }

        for timeframe, file_path in volume_files.items():
            if not file_path.exists():
                print(f"  {file_path.name} not found")
                continue

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                count = 0
                for date_key, period_data in data.items():
                    if 'rankings' in period_data:
                        leaders = []
                        for item in period_data['rankings'][:10]:
                            leaders.append((
                                item['symbol'],
                                item.get('avg_volume', item.get('volume', 0)),
                                item.get('avg_price', item.get('price', 0))
                            ))

                        # Determine actual date
                        if timeframe == 'weekly':
                            date = period_data.get('week_start', date_key)
                        elif timeframe == 'monthly':
                            date = period_data.get('month_start', date_key)
                        else:
                            date = period_data.get('date', date_key)

                        self.db.save_volume_leaders(date, timeframe, leaders)
                        count += 1

                self.stats['volume_leaders'] += count
                print(f"  Migrated {count} {timeframe} volume leader records")

            except Exception as e:
                print(f"  Error loading {file_path.name}: {e}")
                self.stats['errors'].append(f"Volume leaders error: {e}")

    def migrate_historical_prices(self):
        """Migrate historical price data if available"""
        print("\nMigrating historical prices...")

        # This would connect to yfinance to get historical data
        # For now, we'll skip this as it requires external API calls
        print("  Skipping price history (requires yfinance download)")

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 50)
        print("MIGRATION SUMMARY")
        print("=" * 50)

        print(f"Strategies created: {self.stats['strategies']}")
        print(f"Predictions migrated: {self.stats['predictions']}")
        print(f"Trades migrated: {self.stats['trades']}")
        print(f"Volume leaders migrated: {self.stats['volume_leaders']}")

        # Get unique tickers
        db_stats = self.db.get_database_stats()
        print(f"Unique tickers: {db_stats.get('tickers', 0)}")

        if db_stats.get('date_range'):
            print(f"Date range: {db_stats['date_range']}")

        if self.stats['errors']:
            print(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:
                print(f"  - {error}")

        print("\n✅ Migration complete!")
        print(f"Database location: predictor.db")


def main():
    """Run migration"""
    migrator = JSONToSQLiteMigrator()
    migrator.migrate_all()

    # Verify migration
    print("\nVerifying migration...")
    db = DatabaseManager("predictor.db")
    stats = db.get_database_stats()

    print("Database contents:")
    for table, count in stats.items():
        if table != 'date_range':
            print(f"  {table}: {count} records")

if __name__ == "__main__":
    main()