"""
Simple Volume Database Creator

Creates lightweight volume databases without timezone issues:
- Last 30 trading days (daily leaders)
- Last 20 weeks (weekly leaders)
- Last 12 months (monthly leaders)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

def create_volume_databases():
    """Create simple volume databases"""

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    # Core stock universe (high volume stocks)
    stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY',
        'INTC', 'AMD', 'F', 'GE', 'AAL', 'BAC', 'JPM', 'PFE', 'XOM',
        'SIRI', 'SNAP', 'UBER', 'QQQ', 'IWM', 'XLF', 'PLTR', 'GME'
    ]

    print(f"Fetching data for {len(stocks)} stocks...")

    # Get 6 months of data (covers all our needs)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    # Fetch all data at once
    all_data = {}
    for symbol in stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty:
                # Convert timezone-aware index to naive
                hist.index = hist.index.tz_localize(None)
                all_data[symbol] = hist

        except Exception as e:
            print(f"Error with {symbol}: {e}")
            continue

    print(f"Successfully fetched data for {len(all_data)} stocks")

    # Create databases
    databases = {
        'daily': create_daily_leaders(all_data, days=30),
        'weekly': create_weekly_leaders(all_data, weeks=20),
        'monthly': create_monthly_leaders(all_data, months=12)
    }

    # Save databases
    files_created = []
    for db_type, db_data in databases.items():
        filename = output_dir / f"volume_leaders_{db_type}.json"

        with open(filename, 'w') as f:
            json.dump(db_data, f, indent=2, default=str)

        files_created.append(filename)
        print(f"Created {filename} with {len(db_data)} periods")

    # Create summary
    summary = {
        'created_at': datetime.now().isoformat(),
        'databases': {
            'daily': {'periods': len(databases['daily']), 'description': 'Top 10 volume leaders by trading day'},
            'weekly': {'periods': len(databases['weekly']), 'description': 'Top 10 volume leaders by week'},
            'monthly': {'periods': len(databases['monthly']), 'description': 'Top 10 volume leaders by month'}
        },
        'stock_universe': stocks,
        'methodology': 'Average volume ranking, min $10 price',
        'files': [str(f) for f in files_created]
    }

    summary_file = output_dir / "volume_databases_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {summary_file}")
    return databases, summary

def create_daily_leaders(all_data, days=30):
    """Create daily volume leaders"""
    daily_leaders = {}

    # Get last N trading days
    end_date = datetime.now().date()

    for day_offset in range(days):
        target_date = end_date - timedelta(days=day_offset)

        # Find data for this date (or closest trading day)
        day_rankings = []

        for symbol, hist in all_data.items():
            try:
                # Find closest trading day
                available_dates = [d.date() for d in hist.index]
                if not available_dates:
                    continue

                # Find the closest date <= target_date
                valid_dates = [d for d in available_dates if d <= target_date]
                if not valid_dates:
                    continue

                closest_date = max(valid_dates)
                closest_datetime = datetime.combine(closest_date, datetime.min.time())

                if closest_datetime in hist.index:
                    row = hist.loc[closest_datetime]

                    if row['Close'] >= 10:  # Price filter
                        day_rankings.append({
                            'symbol': symbol,
                            'volume': int(row['Volume']),
                            'price': round(row['Close'], 2),
                            'date_used': closest_date.strftime('%Y-%m-%d')
                        })

            except Exception as e:
                continue

        # Sort by volume and take top 10
        day_rankings.sort(key=lambda x: x['volume'], reverse=True)
        top_10 = day_rankings[:10]

        if top_10:  # Only add if we have data
            daily_leaders[target_date.strftime('%Y-%m-%d')] = {
                'date': target_date.strftime('%Y-%m-%d'),
                'top_10_symbols': [r['symbol'] for r in top_10],
                'rankings': top_10
            }

    return daily_leaders

def create_weekly_leaders(all_data, weeks=20):
    """Create weekly volume leaders"""
    weekly_leaders = {}

    end_date = datetime.now().date()

    for week_offset in range(weeks):
        week_end = end_date - timedelta(weeks=week_offset)
        week_start = week_end - timedelta(days=6)

        week_rankings = []

        for symbol, hist in all_data.items():
            try:
                # Get data for this week
                week_mask = (hist.index.date >= week_start) & (hist.index.date <= week_end)
                week_data = hist[week_mask]

                if len(week_data) > 0:
                    avg_volume = week_data['Volume'].mean()
                    avg_price = week_data['Close'].mean()

                    if avg_price >= 10:
                        week_rankings.append({
                            'symbol': symbol,
                            'avg_volume': int(avg_volume),
                            'avg_price': round(avg_price, 2),
                            'trading_days': len(week_data)
                        })

            except Exception as e:
                continue

        # Sort by average volume
        week_rankings.sort(key=lambda x: x['avg_volume'], reverse=True)
        top_10 = week_rankings[:10]

        if top_10:
            weekly_leaders[week_start.strftime('%Y-%m-%d')] = {
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'top_10_symbols': [r['symbol'] for r in top_10],
                'rankings': top_10
            }

    return weekly_leaders

def create_monthly_leaders(all_data, months=12):
    """Create monthly volume leaders"""
    monthly_leaders = {}

    end_date = datetime.now().date()

    for month_offset in range(months):
        # Calculate month start/end
        year = end_date.year
        month = end_date.month - month_offset

        # Handle year rollover
        while month <= 0:
            month += 12
            year -= 1

        month_start = datetime(year, month, 1).date()

        # Last day of month
        if month == 12:
            month_end = datetime(year + 1, 1, 1).date() - timedelta(days=1)
        else:
            month_end = datetime(year, month + 1, 1).date() - timedelta(days=1)

        month_rankings = []

        for symbol, hist in all_data.items():
            try:
                # Get data for this month
                month_mask = (hist.index.date >= month_start) & (hist.index.date <= month_end)
                month_data = hist[month_mask]

                if len(month_data) > 0:
                    avg_volume = month_data['Volume'].mean()
                    avg_price = month_data['Close'].mean()

                    if avg_price >= 10:
                        month_rankings.append({
                            'symbol': symbol,
                            'avg_volume': int(avg_volume),
                            'avg_price': round(avg_price, 2),
                            'trading_days': len(month_data)
                        })

            except Exception as e:
                continue

        # Sort by average volume
        month_rankings.sort(key=lambda x: x['avg_volume'], reverse=True)
        top_10 = month_rankings[:10]

        if top_10:
            monthly_leaders[month_start.strftime('%Y-%m')] = {
                'month': month_start.strftime('%Y-%m'),
                'month_start': month_start.strftime('%Y-%m-%d'),
                'month_end': month_end.strftime('%Y-%m-%d'),
                'top_10_symbols': [r['symbol'] for r in top_10],
                'rankings': top_10
            }

    return monthly_leaders

def main():
    """Main function"""
    print("Creating Volume Leader Databases")
    print("=" * 35)

    databases, summary = create_volume_databases()

    print(f"\nVolume databases created successfully!")

    # Show sample results
    print(f"\nSample Results:")

    if databases['daily']:
        latest_daily = max(databases['daily'].keys())
        daily_top5 = databases['daily'][latest_daily]['top_10_symbols'][:5]
        print(f"Latest daily leaders ({latest_daily}): {daily_top5}")

    if databases['weekly']:
        latest_weekly = max(databases['weekly'].keys())
        weekly_top5 = databases['weekly'][latest_weekly]['top_10_symbols'][:5]
        print(f"Latest weekly leaders: {weekly_top5}")

    if databases['monthly']:
        latest_monthly = max(databases['monthly'].keys())
        monthly_top5 = databases['monthly'][latest_monthly]['top_10_symbols'][:5]
        print(f"Latest monthly leaders: {monthly_top5}")

    print(f"\nFiles ready for GitHub Pages!")

if __name__ == "__main__":
    main()