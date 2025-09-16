"""
Database API for Web Interface
Provides JSON endpoints for the web interface to access SQLite database
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

class DatabaseAPI:
    """API for web interface to access database"""
    
    def __init__(self, db_path: str = "predictor.db"):
        """Initialize API"""
        self.db = DatabaseManager(db_path)
        
    def get_latest_predictions(self, limit: int = 20) -> Dict[str, Any]:
        """Get latest predictions in web-compatible format"""
        # Get today's date
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Get predictions
        predictions = self.db.get_predictions(date=today)
        
        # If no predictions today, get yesterday's
        if not predictions:
            predictions = self.db.get_predictions(date=yesterday)
            date_used = yesterday
        else:
            date_used = today
            
        # Format for web interface
        formatted_predictions = {}
        
        for pred in predictions:
            strategy = pred['strategy'].lower().replace(' ', '_')
            
            if strategy not in formatted_predictions:
                formatted_predictions[strategy] = {
                    'stock': pred['ticker'],
                    'position': pred['position'],
                    'confidence': pred['confidence'],
                    'projected': f"+{pred.get('target_1w', 0):.1f}%" if pred.get('target_1w', 0) > 0 else f"{pred.get('target_1w', 0):.1f}%",
                    'score': pred.get('score')
                }
                
        return {
            'date': date_used,
            'predictions': formatted_predictions,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get active trades for entry tracker"""
        trades = self.db.get_active_trades()
        
        formatted_trades = []
        for trade in trades:
            formatted_trades.append({
                'id': trade['id'],
                'symbol': trade['ticker'],
                'position': trade['position'],
                'entry_date': trade.get('entry_date', trade['prediction_date']),
                'entry_price': trade.get('entry_price', 0),
                'shares': trade.get('shares', 100),
                'strategy': trade['strategy'],
                'confidence': trade['confidence'],
                'status': trade['status']
            })
            
        return formatted_trades
        
    def get_closed_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get closed trades history"""
        with self.db.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    t.*, p.date as prediction_date, p.position, p.confidence,
                    s.name as strategy, tk.symbol as ticker
                FROM trades t
                JOIN predictions p ON t.prediction_id = p.id
                JOIN strategies s ON p.strategy_id = s.id
                JOIN tickers tk ON p.ticker_id = tk.id
                WHERE t.status IN ('CLOSED', 'STOPPED')
                ORDER BY t.exit_date DESC
                LIMIT ?
            """, (limit,))
            
            trades = [dict(row) for row in cursor.fetchall()]
            
        formatted_trades = []
        for trade in trades:
            formatted_trades.append({
                'symbol': trade['ticker'],
                'position': trade['position'],
                'entry_date': trade['entry_date'],
                'entry_price': trade['entry_price'],
                'exit_date': trade['exit_date'],
                'exit_price': trade['exit_price'],
                'return_pct': trade.get('pnl_percent', 0),
                'pnl': trade.get('pnl', 0),
                'strategy': trade['strategy'],
                'exit_reason': trade.get('exit_reason', 'Manual close')
            })
            
        return formatted_trades
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics"""
        with self.db.get_connection() as conn:
            # Get trade statistics
            cursor = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl_percent) as avg_return,
                    SUM(pnl) as total_pnl,
                    MAX(pnl_percent) as best_trade,
                    MIN(pnl_percent) as worst_trade
                FROM trades
                WHERE status IN ('CLOSED', 'STOPPED')
            """)
            
            stats = dict(cursor.fetchone())
            
            # Calculate additional metrics
            if stats['total_trades'] and stats['total_trades'] > 0:
                stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
            else:
                stats['win_rate'] = 0
                
            # Get active trades count
            cursor = conn.execute("SELECT COUNT(*) FROM trades WHERE status = 'ACTIVE'")
            stats['active_trades'] = cursor.fetchone()[0]
            
        return stats
        
    def get_strategy_performance(self) -> List[Dict[str, Any]]:
        """Get performance by strategy"""
        perf_df = self.db.get_strategy_performance()
        
        if perf_df.empty:
            return []
            
        # Group by strategy and calculate metrics
        strategies = []
        for strategy in perf_df['strategy'].unique():
            strat_data = perf_df[perf_df['strategy'] == strategy]
            
            strategies.append({
                'name': strategy,
                'total_trades': int(strat_data['total_trades'].sum()),
                'win_rate': float(strat_data['win_rate'].mean()),
                'avg_return': float(strat_data['total_return'].mean()),
                'sharpe_ratio': float(strat_data['sharpe_ratio'].mean()),
                'total_pnl': float(strat_data['total_pnl'].sum())
            })
            
        return sorted(strategies, key=lambda x: x['total_pnl'], reverse=True)
        
    def export_predictions_json(self, output_file: str = "predictions_data.json"):
        """Export predictions to JSON format for backward compatibility"""
        # Get all predictions
        predictions = self.db.get_predictions()
        
        # Group by date
        predictions_by_date = {}
        for pred in predictions:
            date = pred['date']
            if date not in predictions_by_date:
                predictions_by_date[date] = {'date': date, 'predictions': {}}
                
            strategy = pred['strategy'].lower().replace(' ', '_')
            predictions_by_date[date]['predictions'][strategy] = {
                'stock': pred['ticker'],
                'position': pred['position'],
                'confidence': pred['confidence'],
                'projected': f"+{pred.get('target_1w', 0):.1f}%" if pred.get('target_1w', 0) > 0 else f"{pred.get('target_1w', 0):.1f}%",
                'score': pred.get('score')
            }
            
        # Convert to list and sort by date
        predictions_list = list(predictions_by_date.values())
        predictions_list.sort(key=lambda x: x['date'], reverse=True)
        
        # Save to file
        data = {
            'predictions': predictions_list,
            'last_updated': datetime.now().isoformat(),
            'source': 'database'
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        return len(predictions_list)
        
    def export_trades_json(self, output_file: str = "trades_data.json"):
        """Export trades to JSON format for backward compatibility"""
        active_trades = self.get_active_trades()
        closed_trades = self.get_closed_trades(limit=100)
        
        data = {
            'active_trades': active_trades,
            'closed_trades': closed_trades,
            'last_updated': datetime.now().isoformat(),
            'source': 'database'
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        return len(active_trades) + len(closed_trades)
        
    def generate_api_endpoints(self, output_dir: str = "api_data"):
        """Generate static JSON files for web interface"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Latest predictions
        with open(output_path / "latest_predictions.json", 'w') as f:
            json.dump(self.get_latest_predictions(), f, indent=2)
            
        # Active trades
        with open(output_path / "active_trades.json", 'w') as f:
            json.dump(self.get_active_trades(), f, indent=2)
            
        # Closed trades
        with open(output_path / "closed_trades.json", 'w') as f:
            json.dump(self.get_closed_trades(), f, indent=2)
            
        # Performance metrics
        with open(output_path / "performance_metrics.json", 'w') as f:
            json.dump(self.get_performance_metrics(), f, indent=2, default=str)
            
        # Strategy performance
        with open(output_path / "strategy_performance.json", 'w') as f:
            json.dump(self.get_strategy_performance(), f, indent=2)
            
        print(f"API endpoints generated in {output_path}")
        

def main():
    """Test API and generate files"""
    api = DatabaseAPI()
    
    print("\n" + "="*60)
    print("DATABASE API TEST")
    print("="*60)
    
    # Get latest predictions
    predictions = api.get_latest_predictions()
    print(f"\nLatest predictions: {len(predictions['predictions'])} strategies")
    
    # Get active trades
    active = api.get_active_trades()
    print(f"Active trades: {len(active)}")
    
    # Get performance
    metrics = api.get_performance_metrics()
    print(f"\nPerformance Metrics:")
    print(f"  Total trades: {metrics.get('total_trades', 0)}")
    print(f"  Win rate: {metrics.get('win_rate', 0):.1f}%")
    total_pnl = metrics.get('total_pnl', 0) or 0
    print(f"  Total P&L: ${total_pnl:,.2f}")
    
    # Export JSON files
    print("\nExporting JSON files...")
    api.export_predictions_json()
    api.export_trades_json()
    
    # Generate API endpoints
    print("\nGenerating API endpoints...")
    api.generate_api_endpoints()
    
    print("\nAPI test complete!")
    
if __name__ == "__main__":
    main()