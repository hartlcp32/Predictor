"""
Volume-Based Historical Universe Manager

Provides historically accurate stock universes based on actual trading volume data.
Replaces static stock lists with dynamic volume leaders from specific time periods.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class VolumeUniverse:
    def __init__(self):
        self.data_dir = Path("data")
        self.volume_databases = self._load_volume_databases()

    def _load_volume_databases(self):
        """Load all volume databases"""
        databases = {}

        db_files = {
            'daily': self.data_dir / "volume_leaders_daily.json",
            'weekly': self.data_dir / "volume_leaders_weekly.json",
            'monthly': self.data_dir / "volume_leaders_monthly.json"
        }

        for db_type, file_path in db_files.items():
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        databases[db_type] = json.load(f)
                    print(f"Loaded {db_type} volume database: {len(databases[db_type])} periods")
                except Exception as e:
                    print(f"Error loading {db_type} database: {e}")
                    databases[db_type] = {}
            else:
                print(f"Warning: {file_path} not found")
                databases[db_type] = {}

        return databases

    def get_universe_for_date(self, target_date, lookback_type='weekly', fallback_to_current=True):
        """
        Get the appropriate stock universe for a specific date

        Args:
            target_date: Date string (YYYY-MM-DD) or datetime object
            lookback_type: 'daily', 'weekly', or 'monthly'
            fallback_to_current: If True, use current top volume stocks if historical data unavailable

        Returns:
            List of stock symbols that were volume leaders at that time
        """

        if isinstance(target_date, str):
            target_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        elif isinstance(target_date, datetime):
            target_date = target_date.date()

        database = self.volume_databases.get(lookback_type, {})

        if lookback_type == 'daily':
            return self._get_daily_universe(target_date, database, fallback_to_current)
        elif lookback_type == 'weekly':
            return self._get_weekly_universe(target_date, database, fallback_to_current)
        elif lookback_type == 'monthly':
            return self._get_monthly_universe(target_date, database, fallback_to_current)
        else:
            raise ValueError(f"Invalid lookback_type: {lookback_type}")

    def _get_daily_universe(self, target_date, database, fallback_to_current):
        """Get universe based on daily volume leaders"""

        date_str = target_date.strftime('%Y-%m-%d')

        # Try exact date first
        if date_str in database:
            return database[date_str]['top_10_symbols']

        # Try nearby dates (within 5 days)
        for days_back in range(1, 6):
            nearby_date = target_date - timedelta(days=days_back)
            nearby_str = nearby_date.strftime('%Y-%m-%d')

            if nearby_str in database:
                print(f"Using volume data from {nearby_str} for {date_str}")
                return database[nearby_str]['top_10_symbols']

        return self._get_fallback_universe(fallback_to_current)

    def _get_weekly_universe(self, target_date, database, fallback_to_current):
        """Get universe based on weekly volume leaders"""

        # Find the week that contains our target date
        for week_start_str, week_data in database.items():
            week_start = datetime.strptime(week_start_str, '%Y-%m-%d').date()
            week_end = datetime.strptime(week_data['week_end'], '%Y-%m-%d').date()

            if week_start <= target_date <= week_end:
                return week_data['top_10_symbols']

        # Find closest week
        closest_week = None
        min_distance = float('inf')

        for week_start_str, week_data in database.items():
            week_start = datetime.strptime(week_start_str, '%Y-%m-%d').date()
            distance = abs((target_date - week_start).days)

            if distance < min_distance:
                min_distance = distance
                closest_week = week_data

        if closest_week and min_distance <= 14:  # Within 2 weeks
            return closest_week['top_10_symbols']

        return self._get_fallback_universe(fallback_to_current)

    def _get_monthly_universe(self, target_date, database, fallback_to_current):
        """Get universe based on monthly volume leaders"""

        target_month = target_date.strftime('%Y-%m')

        # Try exact month
        if target_month in database:
            return database[target_month]['top_10_symbols']

        # Find closest month
        closest_month = None
        min_distance = float('inf')

        for month_str, month_data in database.items():
            month_start = datetime.strptime(month_data['month_start'], '%Y-%m-%d').date()
            distance = abs((target_date - month_start).days)

            if distance < min_distance:
                min_distance = distance
                closest_month = month_data

        if closest_month and min_distance <= 45:  # Within ~1.5 months
            return closest_month['top_10_symbols']

        return self._get_fallback_universe(fallback_to_current)

    def _get_fallback_universe(self, use_current):
        """Get fallback universe when historical data unavailable"""

        if use_current:
            # Use the most recent data we have
            for db_type in ['daily', 'weekly', 'monthly']:
                database = self.volume_databases.get(db_type, {})
                if database:
                    latest_key = max(database.keys())
                    return database[latest_key]['top_10_symbols']

        # Ultimate fallback: current static universe
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'SPY', 'INTC', 'F']

    def get_universe_evolution(self, start_date, end_date, lookback_type='weekly'):
        """
        Show how the volume leader universe evolved over time

        Returns:
            Dictionary with dates as keys and stock lists as values
        """

        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        evolution = {}
        current_date = start_date

        while current_date <= end_date:
            universe = self.get_universe_for_date(current_date, lookback_type)
            evolution[current_date.strftime('%Y-%m-%d')] = universe

            # Move to next period
            if lookback_type == 'daily':
                current_date += timedelta(days=7)  # Weekly samples
            elif lookback_type == 'weekly':
                current_date += timedelta(weeks=2)  # Bi-weekly samples
            elif lookback_type == 'monthly':
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)

        return evolution

    def analyze_universe_stability(self, lookback_type='weekly', periods=10):
        """
        Analyze how stable the volume leader universe is over time

        Returns:
            Dictionary with stability metrics
        """

        database = self.volume_databases.get(lookback_type, {})
        if not database:
            return {"error": f"No {lookback_type} database available"}

        # Get recent periods
        sorted_periods = sorted(database.keys(), reverse=True)[:periods]

        if len(sorted_periods) < 2:
            return {"error": "Insufficient data for analysis"}

        # Calculate overlap between consecutive periods
        overlaps = []
        all_symbols = set()
        symbol_frequency = {}

        for i, period in enumerate(sorted_periods):
            symbols = set(database[period]['top_10_symbols'])
            all_symbols.update(symbols)

            # Count frequency
            for symbol in symbols:
                symbol_frequency[symbol] = symbol_frequency.get(symbol, 0) + 1

            # Calculate overlap with previous period
            if i > 0:
                prev_symbols = set(database[sorted_periods[i-1]]['top_10_symbols'])
                overlap = len(symbols & prev_symbols)
                overlaps.append(overlap)

        # Calculate metrics
        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0

        # Most consistent stocks
        consistent_stocks = [(symbol, count) for symbol, count in symbol_frequency.items()]
        consistent_stocks.sort(key=lambda x: x[1], reverse=True)

        return {
            'periods_analyzed': len(sorted_periods),
            'average_overlap': round(avg_overlap, 1),
            'overlap_percentage': round(avg_overlap / 10 * 100, 1),
            'total_unique_symbols': len(all_symbols),
            'most_consistent': consistent_stocks[:10],
            'stability_score': round(avg_overlap / 10 * 100, 1)  # 0-100 scale
        }

def main():
    """Test the volume universe system"""
    print("Testing Volume Universe System")
    print("=" * 30)

    universe = VolumeUniverse()

    # Test getting universe for specific dates
    test_dates = [
        '2025-09-15',  # Recent
        '2025-08-01',  # Last month
        '2025-06-01',  # Earlier this year
    ]

    for date in test_dates:
        try:
            daily_universe = universe.get_universe_for_date(date, 'daily')
            weekly_universe = universe.get_universe_for_date(date, 'weekly')
            monthly_universe = universe.get_universe_for_date(date, 'monthly')

            print(f"\nUniverse for {date}:")
            print(f"  Daily:   {daily_universe}")
            print(f"  Weekly:  {weekly_universe}")
            print(f"  Monthly: {monthly_universe}")
        except Exception as e:
            print(f"Error processing {date}: {e}")

    # Analyze stability
    print(f"\nStability Analysis:")
    for lookback in ['daily', 'weekly', 'monthly']:
        analysis = universe.analyze_universe_stability(lookback)
        if 'error' not in analysis:
            print(f"\n{lookback.title()} stability:")
            print(f"  Average overlap: {analysis['average_overlap']}/10 stocks ({analysis['overlap_percentage']}%)")
            print(f"  Most consistent: {[s[0] for s in analysis['most_consistent'][:5]]}")
        else:
            print(f"\n{lookback.title()}: {analysis['error']}")

if __name__ == "__main__":
    main()