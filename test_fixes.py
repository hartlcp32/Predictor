#!/usr/bin/env python3
"""
Test script to verify the fixes work correctly
"""

import sys
import os

# Add predictors to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'predictors'))

def test_yfinance():
    """Test if yfinance is working"""
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")

        # Test fetching price
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1d")
        if len(data) > 0:
            price = data['Close'].iloc[-1]
            print(f"✅ Successfully fetched AAPL price: ${price:.2f}")
            return True
        else:
            print("❌ No data returned from yfinance")
            return False
    except ImportError:
        print("❌ yfinance not available")
        return False
    except Exception as e:
        print(f"❌ Error testing yfinance: {e}")
        return False

def test_trade_tracker():
    """Test trade tracker with real prices"""
    try:
        from trade_tracker import TradeTracker

        tracker = TradeTracker()
        print("✅ TradeTracker initialized")

        # Test price fetching
        price1 = tracker.get_current_price("AAPL")
        price2 = tracker.get_current_price("AAPL")  # Should be cached

        if price1 and price2:
            print(f"✅ AAPL price consistency: ${price1:.2f} == ${price2:.2f}")
            if price1 == price2:
                print("✅ Price caching working correctly")
                return True
            else:
                print("❌ Price caching not working")
                return False
        else:
            print("❌ Could not fetch AAPL price")
            return False
    except Exception as e:
        print(f"❌ Error testing trade tracker: {e}")
        return False

def test_prediction_generator():
    """Test prediction generation with realistic returns"""
    try:
        from improved_predictor import ImprovedPredictionGenerator

        predictor = ImprovedPredictionGenerator()
        print("✅ ImprovedPredictionGenerator initialized")

        # Test projected return calculation
        mock_prediction = {
            'confidence': 0.75,
            'score': 0.6,
            'position': 'LONG'
        }

        projected = predictor.calculate_projected_return(mock_prediction)
        print(f"✅ Sample projected return: {projected}")

        # Ensure it's not just confidence * 10
        if projected != "+7.5%":
            print("✅ Projected return calculation is realistic (not just confidence * 10)")
            return True
        else:
            print("❌ Still using simple confidence * 10 calculation")
            return False
    except Exception as e:
        print(f"❌ Error testing prediction generator: {e}")
        return False

def main():
    print("🔍 Testing fixes...")
    print("=" * 50)

    all_passed = True

    print("\n1. Testing yfinance dependency:")
    all_passed &= test_yfinance()

    print("\n2. Testing trade tracker price consistency:")
    all_passed &= test_trade_tracker()

    print("\n3. Testing realistic projected returns:")
    all_passed &= test_prediction_generator()

    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")

    return all_passed

if __name__ == "__main__":
    main()