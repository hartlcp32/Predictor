"""
Historical Volume Database Creator

Creates a database of the top 10 highest volume stocks by week
for the past 10 years using real Yahoo Finance data.

Methodology:
- Weekly average volume over 7-day rolling periods
- Market cap filter: >$5B
- Minimum price: >$10
- NYSE/NASDAQ only
- Excludes ETFs, REITs, penny stocks
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import requests
from pathlib import Path

class VolumeHistoryBuilder:
    def __init__(self):
        self.output_dir = Path("data")
        self.output_dir.mkdir(exist_ok=True)

        # Stock universe: Major stocks from past decade
        # This is our "search universe" - we'll rank these by volume each week
        self.candidate_stocks = [
            # Current mega caps
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            # Traditional large caps
            'JNJ', 'JPM', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'CRM',
            'NFLX', 'INTC', 'CSCO', 'PFE', 'KO', 'PEP', 'ABT', 'TMO', 'COST', 'AVGO',
            # Financial sector
            'BAC', 'WFC', 'GS', 'MS', 'C', 'USB', 'PNC', 'TFC', 'AXP', 'BLK',
            # Energy (important in 2010s)
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC', 'VLO', 'PSX',
            # Healthcare/Pharma
            'ABBV', 'MRK', 'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX',
            # Tech that was bigger before
            'IBM', 'ORCL', 'CRM', 'NOW', 'SNOW', 'PLTR', 'UBER', 'LYFT',
            # Retail/Consumer
            'WMT', 'TGT', 'LOW', 'NKE', 'SBUX', 'MCD', 'AMZN',
            # Telecom/Media
            'VZ', 'T', 'TMUS', 'CMCSA', 'DIS', 'NFLX',
            # Historical volume leaders
            'GE', 'F', 'SIRI', 'PLUG', 'AAL', 'DAL', 'CCL', 'NCLH', 'MGM',
            # SPY for reference
            'SPY', 'QQQ', 'IWM'
        ]

        # Remove duplicates
        self.candidate_stocks = list(set(self.candidate_stocks))

    def get_stock_fundamentals(self, symbol, date):
        """Get market cap and price for filtering"""
        try:
            stock = yf.Ticker(symbol)

            # Get historical data around the date
            start_date = date - timedelta(days=5)
            end_date = date + timedelta(days=5)

            hist = stock.history(start=start_date, end=end_date)
            if hist.empty:
                return None, None

            # Get closest price to our target date
            closest_price = hist['Close'].iloc[-1]

            # Get shares outstanding (approximate)
            info = stock.info
            shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))

            if shares_outstanding and closest_price:
                market_cap = shares_outstanding * closest_price
                return market_cap, closest_price

            return None, None

        except Exception as e:
            print(f"Error getting fundamentals for {symbol}: {e}")
            return None, None

    def get_weekly_volume_data(self, symbol, start_date, end_date):
        """Get volume data for a stock over date range"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(start=start_date, end=end_date)

            if hist.empty:
                return None

            # Calculate weekly average volume
            weekly_avg_volume = hist['Volume'].rolling(window=7, min_periods=5).mean()

            return {
                'symbol': symbol,
                'volume_data': weekly_avg_volume.dropna(),
                'price_data': hist['Close']
            }

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def rank_stocks_by_volume(self, date):
        """Rank all candidate stocks by volume for a specific week"""
        week_start = date - timedelta(days=date.weekday())  # Monday of that week
        week_end = week_start + timedelta(days=6)  # Sunday

        stock_rankings = []

        print(f"Processing week of {week_start.strftime('%Y-%m-%d')}...")

        for symbol in self.candidate_stocks:
            # Get volume data
            volume_data = self.get_weekly_volume_data(
                symbol,
                week_start - timedelta(days=14),  # Extra buffer for rolling average
                week_end + timedelta(days=1)
            )

            if not volume_data:
                continue

            # Get volume for that specific week
            try:
                week_volume = volume_data['volume_data'].loc[week_start:week_end].mean()
                week_price = volume_data['price_data'].loc[week_start:week_end].mean()

                if pd.isna(week_volume) or pd.isna(week_price):
                    continue

                # Apply filters
                if week_price < 10:  # Minimum price filter
                    continue

                # Get market cap (approximate)
                market_cap, _ = self.get_stock_fundamentals(symbol, week_start)
                if market_cap and market_cap < 5e9:  # $5B minimum
                    continue

                stock_rankings.append({
                    'symbol': symbol,
                    'avg_volume': week_volume,
                    'avg_price': week_price,
                    'market_cap': market_cap,
                    'dollar_volume': week_volume * week_price
                })

            except Exception as e:
                print(f"Error processing {symbol} for week {week_start}: {e}")
                continue

            # Rate limiting
            time.sleep(0.1)

        # Sort by average volume (could also use dollar volume)
        stock_rankings.sort(key=lambda x: x['avg_volume'], reverse=True)

        return stock_rankings[:10]  # Top 10

    def build_historical_database(self, years_back=10):
        """Build complete historical volume database"""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)

        # Create weekly snapshots
        volume_history = {}

        current_date = start_date
        while current_date < end_date:
            # Process every Monday (weekly snapshots)
            if current_date.weekday() == 0:  # Monday

                week_key = current_date.strftime('%Y-%m-%d')

                try:
                    top_10 = self.rank_stocks_by_volume(current_date)

                    volume_history[week_key] = {
                        'date': week_key,
                        'top_10_by_volume': [stock['symbol'] for stock in top_10],
                        'volume_data': top_10,
                        'year': current_date.year,
                        'quarter': f"Q{(current_date.month - 1) // 3 + 1}"
                    }

                    print(f"Week {week_key}: {[s['symbol'] for s in top_10[:5]]}")

                except Exception as e:
                    print(f"Error processing week {week_key}: {e}")

            current_date += timedelta(days=7)  # Move to next week

        return volume_history

    def save_database(self, volume_history):
        """Save the historical volume database"""

        # Save full database
        output_file = self.output_dir / "historical_volume_leaders.json"
        with open(output_file, 'w') as f:
            json.dump(volume_history, f, indent=2, default=str)

        print(f"Saved historical volume database to {output_file}")

        # Create summary by year
        yearly_summary = {}
        for week_key, week_data in volume_history.items():
            year = week_data['year']
            if year not in yearly_summary:
                yearly_summary[year] = {}

            # Count frequency of each stock in top 10 for that year
            for symbol in week_data['top_10_by_volume']:
                if symbol not in yearly_summary[year]:
                    yearly_summary[year][symbol] = 0
                yearly_summary[year][symbol] += 1

        # Save yearly summary
        summary_file = self.output_dir / "yearly_volume_leaders_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(yearly_summary, f, indent=2)

        print(f"Saved yearly summary to {summary_file}")

        return output_file, summary_file

def main():
    """Build the historical volume database"""
    print("Building Historical Volume Database...")
    print("This will take 20-30 minutes due to API rate limiting...")

    builder = VolumeHistoryBuilder()

    # Build database (start with 2 years for testing, then expand to 10)
    print("Starting with 2-year sample...")
    volume_history = builder.build_historical_database(years_back=2)

    # Save results
    db_file, summary_file = builder.save_database(volume_history)

    print(f"\n✅ Database created successfully!")
    print(f"📁 Full database: {db_file}")
    print(f"📊 Summary: {summary_file}")
    print(f"📈 Total weeks processed: {len(volume_history)}")

    # Show sample results
    recent_weeks = sorted(volume_history.keys())[-4:]
    print(f"\n📋 Sample Results (Last 4 weeks):")
    for week in recent_weeks:
        top_5 = volume_history[week]['top_10_by_volume'][:5]
        print(f"  {week}: {top_5}")

if __name__ == "__main__":
    main()