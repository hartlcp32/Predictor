#!/usr/bin/env python3
"""
Simple test to verify data fetching capability
"""

try:
    import yfinance as yf
    import pandas as pd
    print("✅ Dependencies available")

    # Test single stock fetch
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="5d")
    print(f"✅ Fetched {len(data)} days of AAPL data")
    print(f"Latest close: ${data['Close'].iloc[-1]:.2f}")

except ImportError as e:
    print(f"❌ Missing dependency: {e}")
except Exception as e:
    print(f"❌ Error: {e}")