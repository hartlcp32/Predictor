"""
Export paper trading data for web interface
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager

def export_paper_trading_data():
    """Export paper trading data to JSON for web display"""
    db = DatabaseManager()
    
    data = {
        'last_updated': datetime.now().isoformat(),
        'portfolio': {},
        'positions': [],
        'trades': [],
        'stats': {},
        'equity_curve': []
    }
    
    # Get portfolio value
    with db.get_connection() as conn:
        # Latest portfolio snapshot
        cursor = conn.execute("""
            SELECT * FROM paper_portfolio
            ORDER BY date DESC LIMIT 1
        """)
        portfolio = cursor.fetchone()
        
        if portfolio:
            data['portfolio'] = {
                'total_value': portfolio['total_value'],
                'cash': portfolio['cash'],
                'positions_value': portfolio['positions_value'],
                'daily_return': portfolio['daily_return'],
                'total_return': portfolio['total_return']
            }
        else:
            # Default values
            data['portfolio'] = {
                'total_value': 10000,
                'cash': 10000,
                'positions_value': 0,
                'daily_return': 0,
                'total_return': 0
            }
        
        # Get open positions
        cursor = conn.execute("""
            SELECT pt.*, 
                   (SELECT price FROM position_history 
                    WHERE trade_id = pt.id 
                    ORDER BY date DESC LIMIT 1) as current_price,
                   (SELECT unrealized_pnl FROM position_history 
                    WHERE trade_id = pt.id 
                    ORDER BY date DESC LIMIT 1) as current_pnl,
                   (SELECT unrealized_pnl_pct FROM position_history 
                    WHERE trade_id = pt.id 
                    ORDER BY date DESC LIMIT 1) as current_pnl_pct,
                   julianday('now') - julianday(entry_date) as days_held
            FROM paper_trades pt
            WHERE status = 'OPEN'
            ORDER BY entry_date DESC
        """)
        
        for row in cursor.fetchall():
            position = dict(row)
            data['positions'].append({
                'symbol': position['symbol'],
                'strategy': position['strategy'],
                'position': 'LONG',  # Default, update based on your logic
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': position['current_price'] or position['entry_price'],
                'pnl': position['current_pnl'] or 0,
                'pnl_percent': position['current_pnl_pct'] or 0,
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'days_held': int(position['days_held'] or 0)
            })
        
        # Get closed trades (last 20)
        cursor = conn.execute("""
            SELECT *, 
                   julianday(exit_date) - julianday(entry_date) as days_held
            FROM paper_trades
            WHERE status = 'CLOSED'
            ORDER BY exit_date DESC
            LIMIT 20
        """)
        
        for row in cursor.fetchall():
            trade = dict(row)
            data['trades'].append({
                'date': trade['exit_date'][:10] if trade['exit_date'] else '',
                'symbol': trade['symbol'],
                'strategy': trade['strategy'],
                'side': 'LONG',  # Default
                'entry_price': trade['entry_price'],
                'exit_price': trade['exit_price'],
                'pnl': trade['pnl'],
                'pnl_percent': trade['pnl_percent'],
                'days_held': int(trade['days_held'] or 0),
                'exit_reason': trade['exit_reason']
            })
        
        # Calculate statistics
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl <= 0 THEN pnl END) as avg_loss,
                SUM(pnl) as total_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM paper_trades
            WHERE status = 'CLOSED'
        """)
        
        stats_row = cursor.fetchone()
        if stats_row and stats_row['total_trades']:
            stats = dict(stats_row)
            win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
            
            # Calculate profit factor
            cursor = conn.execute("""
                SELECT 
                    SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                    SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
                FROM paper_trades
                WHERE status = 'CLOSED'
            """)
            pf_row = cursor.fetchone()
            profit_factor = pf_row['gross_profit'] / pf_row['gross_loss'] if pf_row['gross_loss'] > 0 else 0
            
            data['stats'] = {
                'total_trades': stats['total_trades'],
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': round(win_rate, 1),
                'avg_win': stats['avg_win'] or 0,
                'avg_loss': stats['avg_loss'] or 0,
                'profit_factor': round(profit_factor, 2),
                'best_trade': stats['best_trade'] or 0,
                'worst_trade': stats['worst_trade'] or 0,
                'total_pnl': stats['total_pnl'] or 0
            }
        
        # Get equity curve (last 30 days)
        cursor = conn.execute("""
            SELECT date, total_value
            FROM paper_portfolio
            ORDER BY date DESC
            LIMIT 30
        """)
        
        equity_data = cursor.fetchall()
        data['equity_curve'] = [
            {'date': row['date'], 'value': row['total_value']}
            for row in reversed(equity_data)
        ]
    
    # Save to file
    output_file = Path('docs/paper_trading_data.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    print(f"Paper trading data exported to {output_file}")
    print(f"  Open Positions: {len(data['positions'])}")
    print(f"  Closed Trades: {len(data['trades'])}")
    print(f"  Win Rate: {data['stats'].get('win_rate', 0)}%")
    
    return data

def main():
    export_paper_trading_data()

if __name__ == "__main__":
    main()