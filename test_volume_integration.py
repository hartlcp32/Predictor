"""
Test script for volume universe integration

Demonstrates how the historical volume databases work with the prediction system.
"""

import sys
import os
sys.path.append('predictors')

from volume_universe import VolumeUniverse
from datetime import datetime

def test_volume_integration():
    """Test the volume universe integration"""
    print("Testing Volume Universe Integration")
    print("=" * 40)

    # Initialize volume universe
    try:
        universe = VolumeUniverse()
        print("Volume universe loaded successfully")
    except Exception as e:
        print(f"Error loading volume universe: {e}")
        return

    # Test different dates and methods
    test_dates = [
        '2025-09-15',  # Today
        '2025-08-15',  # Last month
        '2025-07-15',  # Two months ago
    ]

    print(f"\nVolume Leader Evolution:")

    for date in test_dates:
        try:
            daily_leaders = universe.get_universe_for_date(date, 'daily')
            weekly_leaders = universe.get_universe_for_date(date, 'weekly')
            monthly_leaders = universe.get_universe_for_date(date, 'monthly')

            print(f"\n{date}:")
            print(f"  Daily:   {daily_leaders[:5]} + 5 more")
            print(f"  Weekly:  {weekly_leaders[:5]} + 5 more")
            print(f"  Monthly: {monthly_leaders[:5]} + 5 more")

            # Compare to static universe
            static_universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'JNJ']
            weekly_set = set(weekly_leaders)
            static_set = set(static_universe)

            overlap = len(weekly_set & static_set)
            print(f"  Overlap with static universe: {overlap}/10 stocks ({overlap*10}%)")

        except Exception as e:
            print(f"  Error: {e}")

    # Analyze stability
    print(f"\nStability Analysis:")
    for timeframe in ['daily', 'weekly', 'monthly']:
        analysis = universe.analyze_universe_stability(timeframe, periods=10)
        if 'error' not in analysis:
            print(f"\n{timeframe.title()}:")
            print(f"  Stability: {analysis['stability_score']}% (higher = more stable)")
            print(f"  Most consistent: {[s[0] for s in analysis['most_consistent'][:5]]}")

    # Show evolution over time
    print(f"\nVolume Leader Evolution (Weekly):")
    evolution = universe.get_universe_evolution('2025-08-01', '2025-09-15', 'weekly')

    for date, leaders in list(evolution.items())[:3]:
        print(f"  {date}: {leaders[:5]}")

    print(f"\nIntegration test complete!")
    print(f"Key Benefits:")
    print(f"  - Historically accurate stock universes")
    print(f"  - Reflects actual market volume patterns")
    print(f"  - Avoids survivorship bias")
    print(f"  - Updates automatically with new data")

def demonstrate_prediction_impact():
    """Show how this affects predictions"""
    print(f"\n" + "="*50)
    print("PREDICTION SYSTEM INTEGRATION")
    print("="*50)

    universe = VolumeUniverse()

    # Compare universes for different periods
    current_universe = universe.get_universe_for_date('2025-09-15', 'weekly')
    past_universe = universe.get_universe_for_date('2025-06-15', 'weekly')

    print(f"\nUniverse Comparison:")
    print(f"Current (Sep 2025): {current_universe}")
    print(f"Past (Jun 2025):    {past_universe}")

    # Show differences
    current_set = set(current_universe)
    past_set = set(past_universe)

    new_leaders = current_set - past_set
    dropped_leaders = past_set - current_set

    if new_leaders:
        print(f"\nNew volume leaders: {list(new_leaders)}")
    if dropped_leaders:
        print(f"Dropped from top 10: {list(dropped_leaders)}")

    print(f"\nIntegration Benefits:")
    print(f"  - Predictions use actual market leaders")
    print(f"  - No more static 'mega cap bias'")
    print(f"  - Captures emerging market trends")
    print(f"  - Better reflects retail trading activity")

if __name__ == "__main__":
    test_volume_integration()
    demonstrate_prediction_impact()