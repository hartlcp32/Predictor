"""
Automated Report Generator
Generates comprehensive HTML and PDF reports for performance tracking
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from tracking.performance_tracker import PerformanceTracker
from backtesting.backtest_engine import BacktestEngine, BacktestConfig

class ReportGenerator:
    """Generate comprehensive performance reports"""
    
    def __init__(self, db_path: str = "predictor.db"):
        """Initialize report generator"""
        self.db = DatabaseManager(db_path)
        self.tracker = PerformanceTracker(db_path)
        self.output_dir = Path('reports')
        self.output_dir.mkdir(exist_ok=True)
        
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        print("\nGenerating daily report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_date = datetime.now().strftime('%Y-%m-%d')
        
        # Get data
        stats = self.db.get_database_stats()
        active_trades = self.db.get_active_trades()
        recent_predictions = self.db.get_predictions(date=report_date)
        
        # Get performance metrics
        perf_report = self.tracker.generate_performance_report()
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Daily Performance Report - {report_date}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background-color: #0a0a0a;
            color: #00ff00;
            padding: 20px;
            margin: 0;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1, h2 {{
            color: #00ff00;
            text-shadow: 0 0 10px #00ff00;
            border-bottom: 2px solid #00ff00;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: rgba(0, 255, 0, 0.1);
            border: 1px solid #00ff00;
            padding: 15px;
            border-radius: 5px;
        }}
        .metric-title {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .positive {{ color: #00ff00; }}
        .negative {{ color: #ff0000; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #00ff00;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background: rgba(0, 255, 0, 0.2);
        }}
        .chart-container {{
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #00ff00;
            background: rgba(0, 255, 0, 0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DAILY PERFORMANCE REPORT</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>OVERALL METRICS</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">Total Trades</div>
                <div class="metric-value">{perf_report['overall']['total_trades']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Win Rate</div>
                <div class="metric-value {('positive' if perf_report['overall']['overall_win_rate'] > 50 else 'negative')}">
                    {perf_report['overall']['overall_win_rate']:.1f}%
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total P&L</div>
                <div class="metric-value {('positive' if perf_report['overall']['total_pnl'] > 0 else 'negative')}">
                    ${perf_report['overall']['total_pnl']:,.2f}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Active Positions</div>
                <div class="metric-value">{len(active_trades)}</div>
            </div>
        </div>
        
        <h2>ACTIVE POSITIONS</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Strategy</th>
                <th>Position</th>
                <th>Entry Date</th>
                <th>Entry Price</th>
                <th>Current P&L</th>
            </tr>
"""
        
        for trade in active_trades[:10]:  # Show top 10
            html_content += f"""
            <tr>
                <td>{trade['ticker']}</td>
                <td>{trade['strategy']}</td>
                <td class="{'negative' if trade['position'] == 'SHORT' else 'positive'}">{trade['position']}</td>
                <td>{trade.get('entry_date', 'PENDING')}</td>
                <td>${trade.get('entry_price', 0):.2f}</td>
                <td>-</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>TODAY'S PREDICTIONS</h2>
        <table>
            <tr>
                <th>Ticker</th>
                <th>Strategy</th>
                <th>Position</th>
                <th>Confidence</th>
                <th>Target</th>
            </tr>
"""
        
        for pred in recent_predictions[:10]:  # Show top 10
            html_content += f"""
            <tr>
                <td>{pred['ticker']}</td>
                <td>{pred['strategy']}</td>
                <td class="{'negative' if pred['position'] == 'SHORT' else 'positive' if pred['position'] == 'LONG' else ''}">
                    {pred['position']}
                </td>
                <td>{pred['confidence']*100:.0f}%</td>
                <td>{pred.get('target_1w', 0):.1f}%</td>
            </tr>
"""
        
        html_content += f"""
        </table>
        
        <h2>DATABASE STATISTICS</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">Strategies</div>
                <div class="metric-value">{stats.get('strategies', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Tickers</div>
                <div class="metric-value">{stats.get('tickers', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Predictions</div>
                <div class="metric-value">{stats.get('predictions', 0)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Historical Trades</div>
                <div class="metric-value">{stats.get('trades', 0)}</div>
            </div>
        </div>
        
        <p style="margin-top: 50px; color: #888; text-align: center;">
            Stock Predictor v2.0 | Database-Driven Analytics
        </p>
    </div>
</body>
</html>
"""
        
        # Save HTML report
        report_path = self.output_dir / f"daily_report_{timestamp}.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        print(f"Daily report saved to {report_path}")
        return str(report_path)
        
    def generate_strategy_comparison_report(self) -> str:
        """Generate strategy comparison report"""
        print("\nGenerating strategy comparison report...")
        
        # Get performance data for all strategies
        perf_df = self.db.get_strategy_performance()
        
        if perf_df.empty:
            print("No performance data available")
            return ""
            
        # Create visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate by Strategy', 'Average Return', 
                          'Sharpe Ratio', 'Total P&L'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                  [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # Group by strategy
        strategy_stats = perf_df.groupby('strategy').agg({
            'win_rate': 'mean',
            'total_return': 'mean',
            'sharpe_ratio': 'mean',
            'total_pnl': 'sum'
        }).reset_index()
        
        # Add traces
        fig.add_trace(
            go.Bar(x=strategy_stats['strategy'], y=strategy_stats['win_rate'],
                  marker_color='green', name='Win Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategy_stats['strategy'], y=strategy_stats['total_return'],
                  marker_color='blue', name='Avg Return'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=strategy_stats['strategy'], y=strategy_stats['sharpe_ratio'],
                  marker_color='purple', name='Sharpe'),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(x=strategy_stats['strategy'], y=strategy_stats['total_pnl'],
                  marker_color='orange', name='Total P&L'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            template='plotly_dark',
            title_text="Strategy Performance Comparison",
            title_font_size=20
        )
        
        # Save as HTML
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"strategy_comparison_{timestamp}.html"
        fig.write_html(str(report_path))
        
        print(f"Strategy comparison saved to {report_path}")
        return str(report_path)
        
    def generate_backtest_report(self, start_date: str, end_date: str) -> str:
        """Generate comprehensive backtest report"""
        print(f"\nGenerating backtest report for {start_date} to {end_date}...")
        
        # Run backtests for all strategies
        results = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT name FROM strategies")
            strategies = [row[0] for row in cursor.fetchall()]
            
        for strategy in strategies:
            engine = BacktestEngine()
            metrics = engine.run_backtest(start_date, end_date, strategy)
            
            if metrics.get('total_trades', 0) > 0:
                results[strategy] = metrics
                
        # Create comparison chart
        if results:
            fig = go.Figure()
            
            strategies_list = list(results.keys())
            returns = [results[s]['total_return'] for s in strategies_list]
            sharpes = [results[s]['sharpe_ratio'] for s in strategies_list]
            win_rates = [results[s]['win_rate'] for s in strategies_list]
            
            # Create bubble chart
            fig.add_trace(go.Scatter(
                x=returns,
                y=sharpes,
                mode='markers+text',
                marker=dict(
                    size=[w/2 for w in win_rates],
                    color=returns,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Return %")
                ),
                text=strategies_list,
                textposition="top center"
            ))
            
            fig.update_layout(
                title="Strategy Backtest Results",
                xaxis_title="Total Return (%)",
                yaxis_title="Sharpe Ratio",
                template='plotly_dark',
                height=600
            )
            
            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = self.output_dir / f"backtest_report_{timestamp}.html"
            fig.write_html(str(report_path))
            
            print(f"Backtest report saved to {report_path}")
            return str(report_path)
            
        return ""
        
    def generate_weekly_summary(self) -> str:
        """Generate weekly performance summary"""
        print("\nGenerating weekly summary...")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        # Get trades from past week
        with self.db.get_connection() as conn:
            query = """
                SELECT t.*, s.name as strategy, tk.symbol as ticker
                FROM trades t
                JOIN predictions p ON t.prediction_id = p.id
                JOIN strategies s ON p.strategy_id = s.id
                JOIN tickers tk ON p.ticker_id = tk.id
                WHERE t.entry_date >= ?
                ORDER BY t.entry_date DESC
            """
            cursor = conn.execute(query, (start_date.strftime('%Y-%m-%d'),))
            week_trades = [dict(row) for row in cursor.fetchall()]
            
        # Calculate weekly metrics
        total_trades = len(week_trades)
        closed_trades = [t for t in week_trades if t['status'] in ['CLOSED', 'STOPPED']]
        winning_trades = [t for t in closed_trades if t.get('pnl_percent', 0) > 0]
        
        weekly_stats = {
            'period': f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            'total_trades': total_trades,
            'closed_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'win_rate': (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0,
            'total_pnl': sum(t.get('pnl', 0) for t in closed_trades),
            'active_trades': len([t for t in week_trades if t['status'] == 'ACTIVE'])
        }
        
        # Create summary HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Weekly Summary - {end_date.strftime('%Y-%m-%d')}</title>
    <style>
        body {{
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            padding: 20px;
        }}
        .summary-box {{
            border: 2px solid #00ff00;
            padding: 20px;
            margin: 20px 0;
            background: rgba(0, 255, 0, 0.05);
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            border-bottom: 1px solid rgba(0, 255, 0, 0.3);
        }}
        .positive {{ color: #00ff00; }}
        .negative {{ color: #ff0000; }}
    </style>
</head>
<body>
    <h1>WEEKLY PERFORMANCE SUMMARY</h1>
    <p>{weekly_stats['period']}</p>
    
    <div class="summary-box">
        <div class="stat-row">
            <span>Total Trades Opened:</span>
            <span>{weekly_stats['total_trades']}</span>
        </div>
        <div class="stat-row">
            <span>Trades Closed:</span>
            <span>{weekly_stats['closed_trades']}</span>
        </div>
        <div class="stat-row">
            <span>Win Rate:</span>
            <span class="{('positive' if weekly_stats['win_rate'] > 50 else 'negative')}">
                {weekly_stats['win_rate']:.1f}%
            </span>
        </div>
        <div class="stat-row">
            <span>Weekly P&L:</span>
            <span class="{('positive' if weekly_stats['total_pnl'] > 0 else 'negative')}">
                ${weekly_stats['total_pnl']:,.2f}
            </span>
        </div>
        <div class="stat-row">
            <span>Active Positions:</span>
            <span>{weekly_stats['active_trades']}</span>
        </div>
    </div>
    
    <h2>TOP PERFORMING TRADES</h2>
    <div class="summary-box">
"""
        
        # Add top trades
        top_trades = sorted(closed_trades, key=lambda x: x.get('pnl_percent', 0), reverse=True)[:5]
        for trade in top_trades:
            html_content += f"""
        <div class="stat-row">
            <span>{trade['ticker']} ({trade['strategy']})</span>
            <span class="positive">+{trade.get('pnl_percent', 0):.2f}%</span>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"weekly_summary_{timestamp}.html"
        
        with open(report_path, 'w') as f:
            f.write(html_content)
            
        print(f"Weekly summary saved to {report_path}")
        return str(report_path)
        
    def run_all_reports(self):
        """Generate all reports"""
        print("\n" + "#"*60)
        print("GENERATING ALL REPORTS")
        print("#"*60)
        
        # Generate daily report
        self.generate_daily_report()
        
        # Generate weekly summary
        self.generate_weekly_summary()
        
        # Generate strategy comparison
        self.generate_strategy_comparison_report()
        
        # Generate backtest report for last 30 days
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        self.generate_backtest_report(start_date, end_date)
        
        print("\nAll reports generated successfully!")
        

def main():
    """Run report generation"""
    generator = ReportGenerator()
    generator.run_all_reports()
    
if __name__ == "__main__":
    main()