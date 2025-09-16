"""
Automated Scheduler for Stock Predictor
Runs predictions, tracking, and reporting on schedule
"""

import schedule
import time
import sys
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from database.db_manager import DatabaseManager
from tracking.performance_tracker import PerformanceTracker
# from reporting.report_generator import ReportGenerator  # Skip if plotly not installed
from predictors.improved_predictor import ImprovedPredictionGenerator

class Scheduler:
    """Main scheduler for automated tasks"""
    
    def __init__(self):
        """Initialize scheduler"""
        self.db = DatabaseManager()
        self.tracker = PerformanceTracker()
        # self.reporter = ReportGenerator()  # Skip if plotly not installed
        self.log_file = Path('logs') / 'scheduler.log'
        self.log_file.parent.mkdir(exist_ok=True)
        
    def log(self, message: str):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + '\n')
            
    def run_morning_tasks(self):
        """Tasks to run every morning before market open"""
        self.log("Starting morning tasks...")
        
        try:
            # 1. Generate new predictions
            self.log("Generating predictions...")
            generator = ImprovedPredictionGenerator()
            generator.generate_all_predictions()

            # 2. Update volume universe
            self.log("Updating volume universe...")
            subprocess.run([sys.executable, 'scripts/simple_volume_db.py'], check=False)
            
            # 3. Monitor predictions and create trades
            self.log("Creating trades from predictions...")
            self.tracker.monitor_predictions()
            
            # 4. Generate morning report (skip if plotly not installed)
            self.log("Morning report generation skipped (plotly not installed)")
            
            self.log("Morning tasks completed successfully")
            
        except Exception as e:
            self.log(f"Error in morning tasks: {e}")
            
    def run_market_hours_tasks(self):
        """Tasks to run during market hours"""
        self.log("Running market hours update...")
        
        try:
            # Update active trades with current prices
            self.tracker.update_all_trades()
            
            # Check for stop losses and take profits
            active_trades = self.db.get_active_trades()
            self.log(f"Monitoring {len(active_trades)} active trades")
            
        except Exception as e:
            self.log(f"Error in market hours tasks: {e}")
            
    def run_evening_tasks(self):
        """Tasks to run after market close"""
        self.log("Starting evening tasks...")
        
        try:
            # 1. Final trade updates
            self.log("Final trade updates...")
            self.tracker.update_all_trades()
            
            # 2. Calculate daily performance
            self.log("Calculating performance metrics...")
            self.tracker.calculate_strategy_performance()
            
            # 3. Generate evening reports (skip if plotly not installed)
            self.log("Evening report generation skipped (plotly not installed)")
            
            # 4. Backup database
            self.log("Backing up database...")
            self.db.backup_database()
            
            # 5. Optimize database
            self.log("Optimizing database...")
            self.db.optimize_database()
            
            self.log("Evening tasks completed successfully")
            
        except Exception as e:
            self.log(f"Error in evening tasks: {e}")
            
    def run_weekly_tasks(self):
        """Tasks to run weekly"""
        self.log("Starting weekly tasks...")
        
        try:
            # 1. Weekly summary (skip if plotly not installed)
            self.log("Weekly reports skipped (plotly not installed)")

            # 2. Update database API for web interface
            self.log("Updating web interface data...")
            from api.database_api import DatabaseAPI
            api = DatabaseAPI()
            api.generate_api_endpoints()
            api.export_predictions_json()
            api.export_trades_json()
            
            # 4. Clean up old logs and reports
            self.cleanup_old_files()
            
            self.log("Weekly tasks completed successfully")
            
        except Exception as e:
            self.log(f"Error in weekly tasks: {e}")
            
    def cleanup_old_files(self, days_to_keep: int = 30):
        """Clean up old log and report files"""
        self.log(f"Cleaning up files older than {days_to_keep} days...")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean logs
        for log_file in Path('logs').glob('*.log'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                log_file.unlink()
                self.log(f"Deleted old log: {log_file}")
                
        # Clean reports
        for report_file in Path('reports').glob('*.html'):
            if report_file.stat().st_mtime < cutoff_date.timestamp():
                report_file.unlink()
                self.log(f"Deleted old report: {report_file}")
                
    def run_continuous(self):
        """Run scheduler continuously"""
        self.log("="*60)
        self.log("STOCK PREDICTOR SCHEDULER STARTED")
        self.log("="*60)
        
        # Schedule tasks
        schedule.every().day.at("08:00").do(self.run_morning_tasks)
        schedule.every().day.at("09:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("10:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("11:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("12:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("13:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("14:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("15:30").do(self.run_market_hours_tasks)
        schedule.every().day.at("16:30").do(self.run_evening_tasks)
        schedule.every().sunday.at("18:00").do(self.run_weekly_tasks)
        
        self.log("Scheduled tasks:")
        self.log("  - Morning tasks: 8:00 AM")
        self.log("  - Market updates: Every hour 9:30 AM - 3:30 PM")
        self.log("  - Evening tasks: 4:30 PM")
        self.log("  - Weekly tasks: Sunday 6:00 PM")
        
        # Run initial tasks
        self.log("\nRunning initial tasks...")
        self.run_morning_tasks()
        
        # Keep running
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.log("\nScheduler stopped by user")
                break
                
            except Exception as e:
                self.log(f"Scheduler error: {e}")
                time.sleep(60)  # Continue after error
                
    def run_once(self):
        """Run all tasks once (for testing)"""
        self.log("Running all tasks once...")
        
        self.run_morning_tasks()
        self.run_market_hours_tasks()
        self.run_evening_tasks()
        
        self.log("All tasks completed")
        

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Predictor Scheduler')
    parser.add_argument('--once', action='store_true', help='Run all tasks once and exit')
    parser.add_argument('--morning', action='store_true', help='Run morning tasks only')
    parser.add_argument('--evening', action='store_true', help='Run evening tasks only')
    parser.add_argument('--weekly', action='store_true', help='Run weekly tasks only')
    
    args = parser.parse_args()
    
    scheduler = Scheduler()
    
    if args.once:
        scheduler.run_once()
    elif args.morning:
        scheduler.run_morning_tasks()
    elif args.evening:
        scheduler.run_evening_tasks()
    elif args.weekly:
        scheduler.run_weekly_tasks()
    else:
        scheduler.run_continuous()
        
if __name__ == "__main__":
    main()