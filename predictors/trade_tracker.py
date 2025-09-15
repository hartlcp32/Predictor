#!/usr/bin/env python3
"""
Trade tracking system with specific entry, target, and stop-loss price levels
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

try:
    import yfinance as yf
    import pandas as pd
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not installed. Using mock data for trade tracking.")


@dataclass
class TradeSetup:
    """Complete trade setup with all price levels"""
    symbol: str
    strategy: str
    position: str  # LONG or SHORT
    entry_price: float
    target_price: float
    stop_loss_price: float
    entry_date: str
    exit_date: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "TARGET", "STOP_LOSS", "TIME_EXIT"
    status: str = "PENDING"  # PENDING, ACTIVE, CLOSED, STOPPED
    return_pct: Optional[float] = None
    days_held: Optional[int] = None


class TradeTracker:
    def __init__(self, data_dir='docs'):
        self.data_dir = data_dir
        self.trades_file = os.path.join(data_dir, 'trades_data.json')
        self.predictions_file = os.path.join(data_dir, 'predictions_data.json')
        self.price_cache = {}  # Cache prices to ensure consistency

    def get_current_price(self, symbol: str, use_cache=True) -> Optional[float]:
        """Get current stock price with caching for consistency"""
        # Return cached price if available and requested
        if use_cache and symbol in self.price_cache:
            return self.price_cache[symbol]

        if not YFINANCE_AVAILABLE:
            print(f"Warning: yfinance not available, cannot fetch real price for {symbol}")
            return None

        try:
            ticker = yf.Ticker(symbol)
            # Try recent data first
            data = ticker.history(period="1d", interval="1m")
            if len(data) > 0:
                price = float(data['Close'].iloc[-1])
                self.price_cache[symbol] = price
                return price

            # Fallback to daily data
            data = ticker.history(period="2d")
            if len(data) > 0:
                price = float(data['Close'].iloc[-1])
                self.price_cache[symbol] = price
                return price

            print(f"No price data available for {symbol}")
            return None
        except Exception as e:
            print(f"Error fetching price for {symbol}: {e}")
            return None

    def clear_price_cache(self):
        """Clear price cache to get fresh prices"""
        self.price_cache = {}

    def calculate_price_levels(self, symbol: str, strategy_name: str, position: str,
                             confidence: float, strategy_config: Dict) -> Dict:
        """Calculate entry, target, and stop-loss price levels"""
        current_price = self.get_current_price(symbol)
        if not current_price:
            return None

        # Get strategy-specific parameters
        target_profit = strategy_config.get('target_profit', 0.08)  # 8% default
        stop_loss = strategy_config.get('stop_loss', 0.05)  # 5% default

        # Adjust targets based on confidence
        confidence_multiplier = confidence * 1.5  # Scale confidence effect
        adjusted_target = target_profit * confidence_multiplier
        adjusted_stop = stop_loss * (2 - confidence)  # Lower confidence = wider stops

        if position == "LONG":
            entry_price = current_price
            target_price = current_price * (1 + adjusted_target)
            stop_loss_price = current_price * (1 - adjusted_stop)
        else:  # SHORT
            entry_price = current_price
            target_price = current_price * (1 - adjusted_target)
            stop_loss_price = current_price * (1 + adjusted_stop)

        return {
            'entry_price': round(entry_price, 2),
            'target_price': round(target_price, 2),
            'stop_loss_price': round(stop_loss_price, 2),
            'target_profit_pct': adjusted_target * 100,
            'stop_loss_pct': adjusted_stop * 100,
            'risk_reward_ratio': adjusted_target / adjusted_stop
        }

    def create_trade_setups_from_predictions(self, predictions_data: Dict) -> List[TradeSetup]:
        """Convert daily predictions into trade setups with price levels"""
        strategy_configs = {
            'momentum': {'target_profit': 0.08, 'stop_loss': 0.05, 'max_days': 10},
            'mean_reversion': {'target_profit': 0.06, 'stop_loss': 0.04, 'max_days': 5},
            'volume_breakout': {'target_profit': 0.10, 'stop_loss': 0.06, 'max_days': 3},
            'technical_indicators': {'target_profit': 0.12, 'stop_loss': 0.07, 'max_days': 15},
            'pattern_recognition': {'target_profit': 0.08, 'stop_loss': 0.05, 'max_days': 12},
            'volatility_arbitrage': {'target_profit': 0.09, 'stop_loss': 0.06, 'max_days': 8},
            'moving_average_crossover': {'target_profit': 0.07, 'stop_loss': 0.04, 'max_days': 20},
            'support_resistance': {'target_profit': 0.08, 'stop_loss': 0.05, 'max_days': 10},
            'market_sentiment': {'target_profit': 0.09, 'stop_loss': 0.05, 'max_days': 12},
            'ensemble': {'target_profit': 0.10, 'stop_loss': 0.06, 'max_days': 14}
        }

        trade_setups = []

        # Get the latest predictions
        if not predictions_data.get('predictions'):
            return trade_setups

        latest_day = predictions_data['predictions'][0]
        date = latest_day['date']

        for strategy_name, prediction in latest_day['predictions'].items():
            if prediction.get('stock') == 'NONE' or prediction.get('position') == 'HOLD':
                continue

            symbol = prediction['stock']
            position = prediction['position']
            confidence = prediction.get('confidence', 0.5)

            strategy_config = strategy_configs.get(strategy_name, strategy_configs['ensemble'])

            # Calculate price levels
            price_levels = self.calculate_price_levels(
                symbol, strategy_name, position, confidence, strategy_config
            )

            if price_levels:
                trade_setup = TradeSetup(
                    symbol=symbol,
                    strategy=strategy_name,
                    position=position,
                    entry_price=price_levels['entry_price'],
                    target_price=price_levels['target_price'],
                    stop_loss_price=price_levels['stop_loss_price'],
                    entry_date=date,
                    status="PENDING"  # Start as PENDING until entry price is hit
                )
                trade_setups.append(trade_setup)

                print(f"📋 {strategy_name}: {symbol} {position}")
                print(f"   Entry: ${price_levels['entry_price']}")
                print(f"   Target: ${price_levels['target_price']} (+{price_levels['target_profit_pct']:.1f}%)")
                print(f"   Stop: ${price_levels['stop_loss_price']} (-{price_levels['stop_loss_pct']:.1f}%)")
                print(f"   R:R = 1:{price_levels['risk_reward_ratio']:.1f}")

        return trade_setups

    def check_active_trades(self) -> List[Dict]:
        """Check all active trades for entry and exit conditions"""
        # Clear price cache to get fresh prices for all checks
        self.clear_price_cache()

        trades_data = self.load_trades_data()
        updates = []

        for trade_data in trades_data.get('active_trades', []):
            if trade_data['status'] not in ['PENDING', 'ACTIVE']:
                continue

            trade = TradeSetup(**trade_data)
            current_price = self.get_current_price(trade.symbol)

            if not current_price:
                continue

            # Calculate days since position was created
            entry_date = datetime.strptime(trade.entry_date, '%Y-%m-%d')
            days_since_created = (datetime.now() - entry_date).days

            # First check if PENDING position should become ACTIVE
            if trade.status == "PENDING":
                entry_triggered = False

                if trade.position == "LONG":
                    # For LONG: Enter when price goes UP to or above entry price
                    entry_triggered = current_price >= trade.entry_price
                else:  # SHORT
                    # For SHORT: Enter when price goes DOWN to or below entry price
                    entry_triggered = current_price <= trade.entry_price

                if entry_triggered:
                    # Position becomes ACTIVE
                    update = {
                        'trade': trade,
                        'status_change': 'PENDING_TO_ACTIVE',
                        'entry_triggered_at': current_price,
                        'days_to_entry': days_since_created
                    }
                    updates.append(update)

                    print(f"🟢 ENTRY TRIGGERED: {trade.symbol} {trade.position}")
                    print(f"   Entry Level: ${trade.entry_price} → Current: ${current_price}")
                    print(f"   Days to entry: {days_since_created}")

                # Skip exit checks for PENDING positions
                continue

            # Check exit conditions for ACTIVE positions
            days_held = days_since_created  # For now, use days since created
            exit_triggered = False
            exit_reason = None

            if trade.position == "LONG":
                if current_price >= trade.target_price:
                    exit_triggered = True
                    exit_reason = "TARGET"
                elif current_price <= trade.stop_loss_price:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"
            else:  # SHORT
                if current_price <= trade.target_price:
                    exit_triggered = True
                    exit_reason = "TARGET"
                elif current_price >= trade.stop_loss_price:
                    exit_triggered = True
                    exit_reason = "STOP_LOSS"

            # Time-based exit (max holding period)
            max_days = 14  # Default max holding period
            if days_held >= max_days:
                exit_triggered = True
                exit_reason = "TIME_EXIT"

            if exit_triggered:
                # Calculate return
                if trade.position == "LONG":
                    return_pct = (current_price - trade.entry_price) / trade.entry_price * 100
                else:  # SHORT
                    return_pct = (trade.entry_price - current_price) / trade.entry_price * 100

                update = {
                    'trade': trade,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'return_pct': return_pct,
                    'days_held': days_held,
                    'exit_date': datetime.now().strftime('%Y-%m-%d')
                }
                updates.append(update)

                print(f"🚨 EXIT SIGNAL: {trade.symbol} {trade.strategy}")
                print(f"   Reason: {exit_reason}")
                print(f"   Entry: ${trade.entry_price} → Exit: ${current_price}")
                print(f"   Return: {return_pct:+.1f}% in {days_held} days")

        return updates

    def generate_trade_alerts(self) -> List[Dict]:
        """Generate trading alerts for active positions"""
        trades_data = self.load_trades_data()
        alerts = []

        for trade_data in trades_data.get('active_trades', []):
            if trade_data['status'] != 'ACTIVE':
                continue

            trade = TradeSetup(**trade_data)
            current_price = self.get_current_price(trade.symbol)

            if not current_price:
                continue

            # Calculate distance to targets
            if trade.position == "LONG":
                target_distance = (trade.target_price - current_price) / current_price * 100
                stop_distance = (current_price - trade.stop_loss_price) / current_price * 100
            else:  # SHORT
                target_distance = (current_price - trade.target_price) / current_price * 100
                stop_distance = (trade.stop_loss_price - current_price) / current_price * 100

            # Generate alerts based on proximity to levels
            if abs(target_distance) < 2:  # Within 2% of target
                alerts.append({
                    'type': 'TARGET_NEAR',
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'message': f"{trade.symbol} within 2% of target (${trade.target_price})",
                    'current_price': current_price,
                    'distance': target_distance
                })

            if abs(stop_distance) < 1:  # Within 1% of stop loss
                alerts.append({
                    'type': 'STOP_NEAR',
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'message': f"{trade.symbol} within 1% of stop loss (${trade.stop_loss_price})",
                    'current_price': current_price,
                    'distance': stop_distance
                })

        return alerts

    def load_trades_data(self) -> Dict:
        """Load existing trades data"""
        if os.path.exists(self.trades_file):
            with open(self.trades_file, 'r') as f:
                return json.load(f)
        return {'active_trades': [], 'closed_trades': [], 'statistics': {}}

    def save_trades_data(self, trades_data: Dict):
        """Save trades data to file"""
        with open(self.trades_file, 'w') as f:
            json.dump(trades_data, f, indent=2, default=str)

    def update_trades_system(self):
        """Complete update of the trading system"""
        print("\n🎯 UPDATING TRADE TRACKING SYSTEM")
        print("=" * 50)

        # Load current data
        trades_data = self.load_trades_data()

        # Load predictions to create new trades
        if os.path.exists(self.predictions_file):
            with open(self.predictions_file, 'r') as f:
                predictions_data = json.load(f)

            # Create trade setups from latest predictions
            new_trades = self.create_trade_setups_from_predictions(predictions_data)

            # Add new trades to active trades (convert to dict for JSON)
            for trade in new_trades:
                trade_dict = {
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'position': trade.position,
                    'entry_price': trade.entry_price,
                    'target_price': trade.target_price,
                    'stop_loss_price': trade.stop_loss_price,
                    'entry_date': trade.entry_date,
                    'status': trade.status
                }
                trades_data['active_trades'].append(trade_dict)

        # Check for entry triggers and exits on all trades
        all_updates = self.check_active_trades()

        # Process updates
        for update in all_updates:
            trade = update['trade']

            if update.get('status_change') == 'PENDING_TO_ACTIVE':
                # Update status from PENDING to ACTIVE
                for active_trade in trades_data['active_trades']:
                    if (active_trade['symbol'] == trade.symbol and
                        active_trade['strategy'] == trade.strategy and
                        active_trade['entry_date'] == trade.entry_date):
                        active_trade['status'] = 'ACTIVE'
                        print(f"✅ {trade.symbol} {trade.strategy} status changed to ACTIVE")
                        break

            elif 'exit_price' in update:
                # Process exit (move from active to closed)

                # Find and remove from active trades
                trades_data['active_trades'] = [
                    t for t in trades_data['active_trades']
                    if not (t['symbol'] == trade.symbol and t['strategy'] == trade.strategy and t['entry_date'] == trade.entry_date)
                ]

                # Add to closed trades
                closed_trade = {
                    'symbol': trade.symbol,
                    'strategy': trade.strategy,
                    'position': trade.position,
                    'entry_price': trade.entry_price,
                    'target_price': trade.target_price,
                    'stop_loss_price': trade.stop_loss_price,
                    'entry_date': trade.entry_date,
                    'exit_date': update['exit_date'],
                    'exit_price': update['exit_price'],
                    'exit_reason': update['exit_reason'],
                    'return_pct': update['return_pct'],
                    'days_held': update['days_held'],
                    'status': 'CLOSED'
                }
                trades_data['closed_trades'].append(closed_trade)

        # Generate current alerts
        alerts = self.generate_trade_alerts()
        trades_data['current_alerts'] = alerts

        # Calculate statistics
        trades_data['statistics'] = self.calculate_trade_statistics(trades_data)
        trades_data['last_updated'] = datetime.now().isoformat()

        # Save updated data
        self.save_trades_data(trades_data)

        print(f"\n✅ Trade tracking updated:")
        print(f"   Active trades: {len(trades_data['active_trades'])}")
        print(f"   Closed trades: {len(trades_data['closed_trades'])}")
        print(f"   Current alerts: {len(alerts)}")

        return trades_data

    def calculate_trade_statistics(self, trades_data: Dict) -> Dict:
        """Calculate trading performance statistics"""
        closed_trades = trades_data.get('closed_trades', [])

        if not closed_trades:
            return {}

        returns = [trade['return_pct'] for trade in closed_trades]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r <= 0]

        stats = {
            'total_trades': len(closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(closed_trades) if closed_trades else 0,
            'avg_win': sum(wins) / len(wins) if wins else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'avg_return': sum(returns) / len(returns),
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
        }

        return stats

    def print_trading_summary(self):
        """Print comprehensive trading summary"""
        trades_data = self.update_trades_system()

        print("\n📊 TRADE TRACKING SUMMARY")
        print("=" * 50)

        # Active trades
        active_trades = trades_data.get('active_trades', [])
        if active_trades:
            print(f"\n🎯 ACTIVE TRADES ({len(active_trades)}):")
            for trade in active_trades:
                current_price = self.get_current_price(trade['symbol'])
                if current_price:
                    if trade['position'] == 'LONG':
                        unrealized = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                    else:
                        unrealized = (trade['entry_price'] - current_price) / trade['entry_price'] * 100

                    print(f"   {trade['symbol']} {trade['position']} - {trade['strategy']}")
                    print(f"      Entry: ${trade['entry_price']} | Current: ${current_price} | P&L: {unrealized:+.1f}%")
                    print(f"      Target: ${trade['target_price']} | Stop: ${trade['stop_loss_price']}")

        # Recent alerts
        alerts = trades_data.get('current_alerts', [])
        if alerts:
            print(f"\n🚨 CURRENT ALERTS ({len(alerts)}):")
            for alert in alerts:
                print(f"   {alert['type']}: {alert['message']}")

        # Statistics
        stats = trades_data.get('statistics', {})
        if stats:
            print(f"\n📈 PERFORMANCE STATS:")
            print(f"   Total Trades: {stats.get('total_trades', 0)}")
            print(f"   Win Rate: {stats.get('win_rate', 0):.1%}")
            print(f"   Avg Return: {stats.get('avg_return', 0):+.1f}%")
            print(f"   Best Trade: {stats.get('best_trade', 0):+.1f}%")
            print(f"   Profit Factor: {stats.get('profit_factor', 0):.2f}")

        print(f"\n✅ Trade data saved to: {self.trades_file}")


if __name__ == "__main__":
    tracker = TradeTracker()
    tracker.print_trading_summary()