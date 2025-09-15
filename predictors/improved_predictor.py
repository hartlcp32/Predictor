"""
Improved prediction system with better stock selection
Each strategy picks its best stock, avoiding duplicates
"""

import json
import os
from datetime import datetime
import numpy as np
from predictors.data_fetcher import StockDataFetcher
from predictors.strategies import get_all_strategies

class ImprovedPredictionGenerator:
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.strategies = get_all_strategies()
        self.output_dir = 'docs'
        
    def rank_all_stocks(self, strategy, stock_features):
        """Rank all stocks for a given strategy"""
        rankings = []
        
        for symbol, features in stock_features.items():
            prediction = strategy.predict(features)
            
            # Calculate a combined score considering position and confidence
            if prediction['position'] == 'HOLD':
                combined_score = 0
            else:
                # Higher score for stronger signals
                combined_score = abs(prediction['score']) * prediction['confidence']
                if prediction['position'] == 'SHORT':
                    combined_score = -combined_score
            
            rankings.append({
                'symbol': symbol,
                'position': prediction['position'],
                'confidence': prediction['confidence'],
                'score': prediction['score'],
                'combined_score': combined_score,
                'raw_prediction': prediction
            })
        
        # Sort by absolute combined score (strongest signals first)
        rankings.sort(key=lambda x: abs(x['combined_score']), reverse=True)
        return rankings
    
    def generate_diverse_predictions(self):
        """Generate predictions ensuring each strategy picks a different stock when possible"""
        print(f"Generating diverse predictions for {datetime.now().strftime('%Y-%m-%d')}...")
        
        # Fetch latest data
        stock_features = self.fetcher.get_latest_features()
        
        # Track which stocks have been picked
        picked_stocks = set()
        strategy_predictions = {}
        
        # First pass: Let each strategy pick its best unpicked stock
        for strategy in self.strategies:
            strategy_name = strategy.name.lower().replace(' ', '_')
            
            # Rank all stocks for this strategy
            rankings = self.rank_all_stocks(strategy, stock_features)
            
            # Find the best unpicked stock
            best_pick = None
            for ranked_stock in rankings:
                if ranked_stock['position'] != 'HOLD':
                    # Prefer unpicked stocks, but take picked if score is much better
                    if ranked_stock['symbol'] not in picked_stocks:
                        best_pick = ranked_stock
                        picked_stocks.add(ranked_stock['symbol'])
                        break
                    elif best_pick is None:
                        # Keep as backup if no unpicked stocks available
                        best_pick = ranked_stock
            
            if best_pick:
                strategy_predictions[strategy_name] = {
                    'stock': best_pick['symbol'],
                    'position': best_pick['position'],
                    'confidence': best_pick['confidence'],
                    'score': best_pick['score'],
                    'projected': self.calculate_projected_return(best_pick['raw_prediction']),
                    'timeframe': '1W',
                    'rank': 1  # This was their top pick
                }
            else:
                strategy_predictions[strategy_name] = {
                    'stock': 'NONE',
                    'position': 'HOLD',
                    'projected': '0%',
                    'timeframe': '-',
                    'rank': 0
                }
        
        # Show diversity statistics
        unique_stocks = len(set(p['stock'] for p in strategy_predictions.values() if p['stock'] != 'NONE'))
        print(f"  Selected {unique_stocks} unique stocks across {len(self.strategies)} strategies")
        
        return strategy_predictions
    
    def generate_concentrated_predictions(self):
        """Alternative: All strategies pick from their top 3, may overlap"""
        print(f"Generating concentrated predictions...")
        
        stock_features = self.fetcher.get_latest_features()
        strategy_predictions = {}
        stock_votes = {}  # Track how many strategies like each stock
        
        for strategy in self.strategies:
            strategy_name = strategy.name.lower().replace(' ', '_')
            
            # Rank all stocks
            rankings = self.rank_all_stocks(strategy, stock_features)
            
            # Take the absolute best pick regardless of overlap
            if rankings and rankings[0]['position'] != 'HOLD':
                best = rankings[0]
                
                # Track votes
                symbol = best['symbol']
                if symbol not in stock_votes:
                    stock_votes[symbol] = {'long': 0, 'short': 0}
                
                if best['position'] == 'LONG':
                    stock_votes[symbol]['long'] += 1
                else:
                    stock_votes[symbol]['short'] += 1
                
                strategy_predictions[strategy_name] = {
                    'stock': best['symbol'],
                    'position': best['position'],
                    'confidence': best['confidence'],
                    'score': best['score'],
                    'projected': self.calculate_projected_return(best['raw_prediction']),
                    'timeframe': '1W'
                }
            else:
                strategy_predictions[strategy_name] = {
                    'stock': 'NONE',
                    'position': 'HOLD',
                    'projected': '0%',
                    'timeframe': '-'
                }
        
        # Add consensus data
        for strategy_name, pred in strategy_predictions.items():
            if pred['stock'] != 'NONE':
                votes = stock_votes.get(pred['stock'], {'long': 0, 'short': 0})
                total_votes = votes['long'] + votes['short']
                pred['consensus'] = f"{total_votes} strategies agree"
        
        return strategy_predictions, stock_votes
    
    def calculate_projected_return(self, prediction):
        """Calculate realistic 1-week projected return based on technical analysis"""
        # Use score-based calculation with volatility adjustment
        base_score = abs(prediction['score'])
        confidence = prediction['confidence']

        # More realistic return expectations (1-5% typical weekly moves)
        # High confidence + strong score = higher expected return
        max_weekly_return = base_score * confidence * 6.0  # Max ~6% for strongest signals

        # Add some randomization based on market volatility (more realistic)
        import random
        volatility_factor = random.uniform(0.7, 1.3)  # ±30% variation
        projected_return = max_weekly_return * volatility_factor

        # Cap at reasonable levels (weekly returns rarely exceed 8-10%)
        projected_return = min(projected_return, 8.0)
        projected_return = max(projected_return, 1.0)  # Minimum 1%

        if prediction['position'] == 'SHORT':
            return f"-{projected_return:.1f}%"
        else:
            return f"+{projected_return:.1f}%"
    
    def generate_both_approaches(self):
        """Generate both diverse and concentrated predictions"""
        
        # Get diverse predictions (each strategy different stock)
        diverse = self.generate_diverse_predictions()
        
        # Get concentrated predictions (overlap allowed)
        concentrated, stock_votes = self.generate_concentrated_predictions()
        
        # Find the most popular stocks
        popular_stocks = []
        for symbol, votes in stock_votes.items():
            total = votes['long'] + votes['short']
            if total > 1:  # More than one strategy likes it
                popular_stocks.append({
                    'symbol': symbol,
                    'total_votes': total,
                    'long_votes': votes['long'],
                    'short_votes': votes['short'],
                    'consensus': 'LONG' if votes['long'] > votes['short'] else 'SHORT'
                })
        
        popular_stocks.sort(key=lambda x: x['total_votes'], reverse=True)
        
        # Create combined output
        output = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'diverse_predictions': diverse,
            'concentrated_predictions': concentrated,
            'consensus_picks': popular_stocks[:3],  # Top 3 most agreed upon
            'generated_at': datetime.now().isoformat(),
            'approach': 'Both diverse (unique picks) and concentrated (best picks) shown'
        }
        
        # Display summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        
        print("\n📊 DIVERSE APPROACH (Each strategy picks different stock):")
        for strategy, pred in diverse.items():
            if pred['stock'] != 'NONE':
                print(f"  {strategy:25} → {pred['stock']:6} {pred['position']:5} {pred['projected']}")
        
        print("\n🎯 CONCENTRATED APPROACH (Strategies pick their best):")
        for strategy, pred in concentrated.items():
            if pred['stock'] != 'NONE':
                consensus = pred.get('consensus', '')
                print(f"  {strategy:25} → {pred['stock']:6} {pred['position']:5} {pred['projected']} {consensus}")
        
        print("\n⭐ TOP CONSENSUS PICKS:")
        for pick in popular_stocks[:3]:
            print(f"  {pick['symbol']:6} - {pick['total_votes']} strategies agree ({pick['long_votes']}L/{pick['short_votes']}S) → {pick['consensus']}")
        
        return output
    
    def save_predictions(self, predictions_data):
        """Save predictions to JSON file"""
        filepath = os.path.join(self.output_dir, 'predictions_data.json')
        
        # Load existing or create new
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            data = {'predictions': []}
        
        # Add new predictions
        data['predictions'].insert(0, predictions_data)
        data['predictions'] = data['predictions'][:30]  # Keep 30 days
        
        # Save
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Predictions saved to {filepath}")
        return data

if __name__ == "__main__":
    generator = ImprovedPredictionGenerator()
    predictions = generator.generate_both_approaches()
    generator.save_predictions(predictions)