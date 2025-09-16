"""
Analyze the total universe of stocks that appeared in volume databases
"""

import json
from pathlib import Path

def analyze_universe_size():
    """Count unique tickers across all volume databases"""

    data_dir = Path("data")
    all_tickers = set()

    # Load each database
    databases = {
        'daily': data_dir / "volume_leaders_daily.json",
        'weekly': data_dir / "volume_leaders_weekly.json",
        'monthly': data_dir / "volume_leaders_monthly.json"
    }

    stats = {}

    for db_name, db_path in databases.items():
        if db_path.exists():
            with open(db_path, 'r') as f:
                data = json.load(f)

            db_tickers = set()
            period_count = 0

            for period_key, period_data in data.items():
                if 'top_10_symbols' in period_data:
                    db_tickers.update(period_data['top_10_symbols'])
                    period_count += 1

            stats[db_name] = {
                'unique_tickers': len(db_tickers),
                'periods': period_count,
                'tickers': sorted(list(db_tickers))
            }

            all_tickers.update(db_tickers)

    print("VOLUME UNIVERSE ANALYSIS")
    print("=" * 40)

    for db_name, info in stats.items():
        print(f"\n{db_name.upper()} DATABASE:")
        print(f"  Periods tracked: {info['periods']}")
        print(f"  Unique tickers: {info['unique_tickers']}")
        print(f"  Tickers: {', '.join(info['tickers'][:10])}")
        if len(info['tickers']) > 10:
            print(f"           {', '.join(info['tickers'][10:])}")

    print(f"\nTOTAL UNIQUE TICKERS ACROSS ALL DATABASES: {len(all_tickers)}")
    print(f"Complete universe: {', '.join(sorted(all_tickers))}")

    # Analyze ticker frequency
    ticker_frequency = {}
    for db_name, db_path in databases.items():
        if db_path.exists():
            with open(db_path, 'r') as f:
                data = json.load(f)

            for period_key, period_data in data.items():
                if 'top_10_symbols' in period_data:
                    for ticker in period_data['top_10_symbols']:
                        ticker_frequency[ticker] = ticker_frequency.get(ticker, 0) + 1

    # Sort by frequency
    sorted_tickers = sorted(ticker_frequency.items(), key=lambda x: x[1], reverse=True)

    print(f"\nMOST CONSISTENT VOLUME LEADERS:")
    for ticker, count in sorted_tickers[:15]:
        print(f"  {ticker:6} appeared {count:3} times")

    return all_tickers, stats

if __name__ == "__main__":
    all_tickers, stats = analyze_universe_size()