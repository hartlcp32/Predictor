#!/usr/bin/env python3
"""
System update script - Run comprehensive updates to all prediction components
"""

import os
import sys
from datetime import datetime

def main():
    print("🚀 STOCK PREDICTOR SYSTEM UPDATE")
    print("=" * 50)

    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    try:
        # 1. Generate new predictions
        print("\n📊 Generating new predictions...")
        from generate_predictions import PredictionGenerator
        generator = PredictionGenerator()
        generator.run()

        # 2. Update performance tracking
        print("\n📈 Updating performance tracking...")
        from predictors.performance_tracker import PerformanceTracker
        tracker = PerformanceTracker('docs')
        tracker.print_performance_summary()

        # 3. Update risk analysis
        print("\n⚖️ Updating risk analysis...")
        from predictors.risk_analyzer import RiskAnalyzer
        analyzer = RiskAnalyzer('docs')
        analyzer.print_risk_summary()

        # 4. Update trade tracking
        print("\n🎯 Updating trade tracking...")
        from predictors.trade_tracker import TradeTracker
        trade_tracker = TradeTracker('docs')
        trade_tracker.print_trading_summary()

        print("\n✅ System update completed successfully!")
        print("\n📝 Updated files:")
        print("   - docs/predictions_data.json")
        print("   - docs/performance_data.json")
        print("   - docs/risk_analysis.json")

        print("\n🌐 Your GitHub Pages site will automatically update with:")
        print("   - New daily predictions")
        print("   - Real performance metrics")
        print("   - Interactive charts and analytics")
        print("   - Strategy comparison dashboard")
        print("   - Risk analysis and alerts")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install yfinance pandas numpy scikit-learn")

    except Exception as e:
        print(f"❌ Error during update: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())