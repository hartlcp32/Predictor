"""
Quick Volume Sample - Test the volume ranking methodology

This script demonstrates the approach by getting volume leaders
for just the past month to validate the methodology before
running the full 10-year database build.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json

def get_current_volume_leaders(days_back=30):
    """Get volume leaders for the past month as a test"""

    # Broader stock universe for testing
    test_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
        'JPM', 'JNJ', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'BAC', 'XOM', 'CVX',
        'INTC', 'CSCO', 'PFE', 'KO', 'WFC', 'GE', 'F', 'AAL', 'SIRI', 'SPY',
        'AMD', 'CRM', 'NFLX', 'PYPL', 'ADBE', 'T', 'VZ', 'WMT', 'TGT'
    ]

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    volume_data = []

    print(f"Fetching volume data from {start_date.date()} to {end_date.date()}...")

    for symbol in test_stocks:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            if not hist.empty:
                avg_volume = hist['Volume'].mean()
                avg_price = hist['Close'].mean()
                dollar_volume = avg_volume * avg_price
                latest_price = hist['Close'].iloc[-1]

                # Get basic info for market cap estimate
                try:
                    info = stock.info
                    market_cap = info.get('marketCap', 0)
                except:
                    market_cap = 0

                volume_data.append({
                    'symbol': symbol,
                    'avg_daily_volume': int(avg_volume),
                    'avg_price': round(avg_price, 2),
                    'dollar_volume': int(dollar_volume),
                    'latest_price': round(latest_price, 2),
                    'market_cap': market_cap
                })

        except Exception as e:
            print(f"Error with {symbol}: {e}")
            continue

    # Filter and rank
    # Remove stocks under $10 and small market caps
    filtered = [s for s in volume_data if s['avg_price'] > 10 and s['market_cap'] > 1e9]

    # Sort by average volume
    volume_ranked = sorted(filtered, key=lambda x: x['avg_daily_volume'], reverse=True)

    # Sort by dollar volume (alternative ranking)
    dollar_ranked = sorted(filtered, key=lambda x: x['dollar_volume'], reverse=True)

    return {
        'by_volume': volume_ranked[:15],
        'by_dollar_volume': dollar_ranked[:15],
        'period': f"{start_date.date()} to {end_date.date()}",
        'methodology': 'Average daily volume over past 30 days'
    }

def main():
    """Test volume ranking methodology"""
    print("Testing Volume Ranking Methodology...")
    print("=" * 50)

    results = get_current_volume_leaders()

    print(f"\nRESULTS FOR PERIOD: {results['period']}")
    print(f"Methodology: {results['methodology']}")

    print(f"\nTOP 10 BY AVERAGE VOLUME:")
    print(f"{'Rank':<4} {'Symbol':<8} {'Avg Volume':<15} {'Avg Price':<12} {'Market Cap':<12}")
    print("-" * 60)

    for i, stock in enumerate(results['by_volume'][:10], 1):
        volume_str = f"{stock['avg_daily_volume']:,}"
        price_str = f"${stock['avg_price']}"
        mcap_str = f"${stock['market_cap']/1e9:.1f}B" if stock['market_cap'] > 0 else "N/A"

        print(f"{i:<4} {stock['symbol']:<8} {volume_str:<15} {price_str:<12} {mcap_str:<12}")

    print(f"\nTOP 10 BY DOLLAR VOLUME:")
    print(f"{'Rank':<4} {'Symbol':<8} {'Dollar Volume':<15} {'Price':<12} {'Volume':<15}")
    print("-" * 70)

    for i, stock in enumerate(results['by_dollar_volume'][:10], 1):
        dollar_str = f"${stock['dollar_volume']:,}"
        price_str = f"${stock['avg_price']}"
        volume_str = f"{stock['avg_daily_volume']:,}"

        print(f"{i:<4} {stock['symbol']:<8} {dollar_str:<15} {price_str:<12} {volume_str:<15}")

    # Save sample results
    output_file = "data/volume_sample_30day.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSample results saved to: {output_file}")

    print(f"\nRECOMMENDED APPROACH:")
    print("- Use average volume (not dollar volume) for consistency")
    print("- Filter: Price >$10, Market Cap >$1B")
    print("- Weekly snapshots for 10-year database")
    print("- This methodology captures real market volume patterns")

    # Show how rankings differ
    volume_top10 = [s['symbol'] for s in results['by_volume'][:10]]
    dollar_top10 = [s['symbol'] for s in results['by_dollar_volume'][:10]]

    print(f"\nRANKING COMPARISON:")
    print(f"Volume ranking:  {volume_top10}")
    print(f"Dollar ranking:  {dollar_top10}")
    print(f"Overlap: {len(set(volume_top10) & set(dollar_top10))}/10 stocks")

if __name__ == "__main__":
    main()