"""
Multiple Volume Databases Builder

Creates three volume databases:
1. Daily: Top 10 highest volume stocks for prior trading day
2. Weekly: Top 10 highest volume stocks for prior trading week
3. Monthly: Top 10 highest volume stocks for prior trading month

Each database goes back 2-3 years to show historical changes in volume leaders.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path
import numpy as np

class MultiVolumeDatabase:
    def __init__(self):
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)

        # Expanded stock universe - major stocks from past 5 years
        self.stock_universe = [
            # Current mega caps
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'BRK-A',
            # Large cap tech
            'AVGO', 'ORCL', 'CRM', 'ADBE', 'NFLX', 'INTC', 'CSCO', 'AMD', 'NOW', 'INTU',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'USB', 'PNC', 'TFC', 'BLK', 'SCHW',
            # Healthcare
            'JNJ', 'UNH', 'PG', 'HD', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
            # Consumer/Retail
            'WMT', 'MA', 'V', 'DIS', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST',
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'MPC', 'VLO', 'PSX', 'HAL',
            # Telecom/Utilities
            'VZ', 'T', 'TMUS', 'NEE', 'SO', 'DUK', 'AEP', 'EXC', 'XEL', 'PCG',
            # Industrial
            'BA', 'CAT', 'MMM', 'GE', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'DE',
            # High volume/volatile stocks
            'F', 'GE', 'AAL', 'DAL', 'CCL', 'NCLH', 'MGM', 'SIRI', 'SNAP', 'UBER', 'LYFT',
            'PLTR', 'COIN', 'ROKU', 'ZM', 'PTON', 'AMC', 'GME', 'BB', 'NOK',
            # ETFs for reference
            'SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLK', 'XLV', 'VTI', 'EFA', 'EEM'
        ]

        # Remove duplicates and sort
        self.stock_universe = sorted(list(set(self.stock_universe)))
        print(f"Stock universe: {len(self.stock_universe)} symbols")

    def get_trading_days(self, start_date, end_date):
        """Get actual trading days in a period (excludes weekends/holidays)"""
        # Use SPY as proxy for trading days
        spy = yf.Ticker('SPY')
        hist = spy.history(start=start_date, end=end_date + timedelta(days=1))
        return hist.index.date.tolist()

    def fetch_volume_data(self, symbols, start_date, end_date):
        """Fetch volume data for multiple symbols"""
        volume_data = {}

        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                hist = stock.history(start=start_date, end=end_date + timedelta(days=1))

                if not hist.empty and len(hist) > 0:
                    # Get basic info for filtering
                    try:
                        info = stock.info
                        market_cap = info.get('marketCap', 0)
                    except:
                        market_cap = 0

                    volume_data[symbol] = {
                        'volume': hist['Volume'],
                        'close': hist['Close'],
                        'market_cap': market_cap,
                        'latest_price': hist['Close'].iloc[-1] if len(hist) > 0 else 0
                    }

            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue

        return volume_data

    def rank_by_period(self, volume_data, period_start, period_end, period_type):
        """Rank stocks by volume for a specific period"""
        rankings = []

        for symbol, data in volume_data.items():
            try:
                # Filter data for the period
                period_data = data['volume'].loc[period_start:period_end]
                period_prices = data['close'].loc[period_start:period_end]

                if len(period_data) == 0:
                    continue

                # Calculate average volume and price for the period
                avg_volume = period_data.mean()
                avg_price = period_prices.mean()
                total_volume = period_data.sum()
                latest_price = data['latest_price']

                # Apply filters
                if avg_price < 10:  # Minimum price
                    continue

                if data['market_cap'] and data['market_cap'] < 1e9:  # $1B minimum
                    continue

                rankings.append({
                    'symbol': symbol,
                    'avg_volume': int(avg_volume),
                    'total_volume': int(total_volume),
                    'avg_price': round(avg_price, 2),
                    'latest_price': round(latest_price, 2),
                    'dollar_volume': int(avg_volume * avg_price),
                    'market_cap': data['market_cap'],
                    'trading_days': len(period_data)
                })

            except Exception as e:
                print(f"Error ranking {symbol}: {e}")
                continue

        # Sort by average volume
        rankings.sort(key=lambda x: x['avg_volume'], reverse=True)
        return rankings[:10]  # Top 10

    def build_daily_database(self, days_back=500):
        """Build database of daily volume leaders"""
        print("Building daily volume database...")

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back + 10)  # Extra buffer

        # Get trading days
        trading_days = self.get_trading_days(start_date, end_date)
        trading_days = trading_days[-days_back:]  # Last N trading days

        # Fetch all volume data once
        print(f"Fetching volume data for {len(self.stock_universe)} stocks...")
        volume_data = self.fetch_volume_data(self.stock_universe, start_date, end_date)

        daily_database = {}

        for i, day in enumerate(trading_days):
            if i % 50 == 0:
                print(f"Processing day {i+1}/{len(trading_days)}: {day}")

            try:
                # Rank stocks for this specific day
                rankings = self.rank_by_period(volume_data, day, day, 'daily')

                daily_database[day.strftime('%Y-%m-%d')] = {
                    'date': day.strftime('%Y-%m-%d'),
                    'top_10_symbols': [r['symbol'] for r in rankings],
                    'rankings': rankings,
                    'period_type': 'daily'
                }

            except Exception as e:
                print(f"Error processing {day}: {e}")
                continue

        return daily_database

    def build_weekly_database(self, weeks_back=100):
        """Build database of weekly volume leaders"""
        print("Building weekly volume database...")

        end_date = datetime.now().date()
        start_date = end_date - timedelta(weeks=weeks_back + 4)  # Extra buffer

        weekly_database = {}

        # Fetch all volume data once
        volume_data = self.fetch_volume_data(self.stock_universe, start_date, end_date)

        for week_num in range(weeks_back):
            week_end = end_date - timedelta(weeks=week_num)
            week_start = week_end - timedelta(days=6)  # 7-day week

            if week_num % 20 == 0:
                print(f"Processing week {week_num+1}/{weeks_back}: {week_start} to {week_end}")

            try:
                rankings = self.rank_by_period(volume_data, week_start, week_end, 'weekly')

                weekly_database[week_start.strftime('%Y-%m-%d')] = {
                    'week_start': week_start.strftime('%Y-%m-%d'),
                    'week_end': week_end.strftime('%Y-%m-%d'),
                    'top_10_symbols': [r['symbol'] for r in rankings],
                    'rankings': rankings,
                    'period_type': 'weekly'
                }

            except Exception as e:
                print(f"Error processing week {week_start}: {e}")
                continue

        return weekly_database

    def build_monthly_database(self, months_back=24):
        """Build database of monthly volume leaders"""
        print("Building monthly volume database...")

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=months_back * 31 + 30)  # Extra buffer

        monthly_database = {}

        # Fetch all volume data once
        volume_data = self.fetch_volume_data(self.stock_universe, start_date, end_date)

        for month_num in range(months_back):
            # Calculate month boundaries
            month_end = datetime(end_date.year, end_date.month, 1).date() - timedelta(days=month_num * 31)
            month_start = datetime(month_end.year, month_end.month, 1).date()

            # Adjust to last day of month
            if month_end.month == 12:
                month_end = datetime(month_end.year + 1, 1, 1).date() - timedelta(days=1)
            else:
                month_end = datetime(month_end.year, month_end.month + 1, 1).date() - timedelta(days=1)

            print(f"Processing month {month_num+1}/{months_back}: {month_start} to {month_end}")

            try:
                rankings = self.rank_by_period(volume_data, month_start, month_end, 'monthly')

                monthly_database[month_start.strftime('%Y-%m')] = {
                    'month_start': month_start.strftime('%Y-%m-%d'),
                    'month_end': month_end.strftime('%Y-%m-%d'),
                    'month': month_start.strftime('%Y-%m'),
                    'top_10_symbols': [r['symbol'] for r in rankings],
                    'rankings': rankings,
                    'period_type': 'monthly'
                }

            except Exception as e:
                print(f"Error processing month {month_start}: {e}")
                continue

        return monthly_database

    def save_databases(self, daily_db, weekly_db, monthly_db):
        """Save all databases"""

        # Save individual databases
        daily_file = self.output_dir / "volume_leaders_daily.json"
        weekly_file = self.output_dir / "volume_leaders_weekly.json"
        monthly_file = self.output_dir / "volume_leaders_monthly.json"

        with open(daily_file, 'w') as f:
            json.dump(daily_db, f, indent=2, default=str)

        with open(weekly_file, 'w') as f:
            json.dump(weekly_db, f, indent=2, default=str)

        with open(monthly_file, 'w') as f:
            json.dump(monthly_db, f, indent=2, default=str)

        # Create combined summary
        summary = {
            'created_at': datetime.now().isoformat(),
            'methodology': 'Average volume ranking with $10 min price, $1B min market cap',
            'daily_periods': len(daily_db),
            'weekly_periods': len(weekly_db),
            'monthly_periods': len(monthly_db),
            'stock_universe_size': len(self.stock_universe),
            'files': {
                'daily': str(daily_file),
                'weekly': str(weekly_file),
                'monthly': str(monthly_file)
            }
        }

        summary_file = self.output_dir / "volume_databases_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return daily_file, weekly_file, monthly_file, summary_file

def main():
    """Build all volume databases"""
    print("Building Multiple Volume Databases")
    print("=" * 40)

    builder = MultiVolumeDatabase()

    # Build databases (reduced timeframes for faster execution)
    daily_db = builder.build_daily_database(days_back=100)      # ~5 months of daily data
    weekly_db = builder.build_weekly_database(weeks_back=50)    # ~1 year of weekly data
    monthly_db = builder.build_monthly_database(months_back=24) # 2 years of monthly data

    # Save databases
    daily_file, weekly_file, monthly_file, summary_file = builder.save_databases(daily_db, weekly_db, monthly_db)

    print(f"\n✅ Volume databases created successfully!")
    print(f"📅 Daily database: {daily_file} ({len(daily_db)} trading days)")
    print(f"📊 Weekly database: {weekly_file} ({len(weekly_db)} weeks)")
    print(f"📈 Monthly database: {monthly_file} ({len(monthly_db)} months)")
    print(f"📋 Summary: {summary_file}")

    # Show sample results
    print(f"\n📋 SAMPLE RESULTS:")

    # Latest daily leaders
    latest_daily = max(daily_db.keys())
    daily_leaders = daily_db[latest_daily]['top_10_symbols'][:5]
    print(f"Latest daily leaders ({latest_daily}): {daily_leaders}")

    # Latest weekly leaders
    latest_weekly = max(weekly_db.keys())
    weekly_leaders = weekly_db[latest_weekly]['top_10_symbols'][:5]
    print(f"Latest weekly leaders: {weekly_leaders}")

    # Latest monthly leaders
    latest_monthly = max(monthly_db.keys())
    monthly_leaders = monthly_db[latest_monthly]['top_10_symbols'][:5]
    print(f"Latest monthly leaders: {monthly_leaders}")

if __name__ == "__main__":
    main()