"""Quick test of system components"""

from datetime import datetime
from database.db_manager import DatabaseManager
from tracking.performance_tracker import PerformanceTracker
from api.database_api import DatabaseAPI

print("\n" + "="*50)
print("STOCK PREDICTOR SYSTEM TEST")
print("="*50)

# Check database
db = DatabaseManager()
stats = db.get_database_stats()
print(f"\nDatabase Status:")
print(f"  Predictions: {stats['predictions']} records")
print(f"  Trades: {stats['trades']} records")
print(f"  Tickers: {stats['tickers']} symbols")

# Check active trades
active = db.get_active_trades()
print(f"\nActive Trades: {len(active)}")

# Generate API files for web interface
api = DatabaseAPI()
api.generate_api_endpoints()
print(f"\nAPI endpoints updated in api_data/")

# Export for web compatibility
api.export_predictions_json()
api.export_trades_json()
print(f"JSON files updated for web interface")

print(f"\nTest complete! Check:")
print(f"  - predictions_data.json")
print(f"  - trades_data.json") 
print(f"  - api_data/ folder")
print(f"\nWeb interface: https://hartlcp32.github.io/Predictor/")