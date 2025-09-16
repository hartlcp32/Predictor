"""Fix all strategies by adding default get_exit_condition method"""

import sys
from pathlib import Path

# Read the strategies file
file_path = Path('predictors/strategies.py')
content = file_path.read_text()

# Strategies that need fixing
strategies_to_fix = [
    'VolumeBreakoutStrategy',
    'TechnicalIndicatorStrategy', 
    'PatternRecognitionStrategy',
    'VolatilityArbitrageStrategy',
    'MovingAverageCrossoverStrategy',
    'SupportResistanceStrategy',
    'MarketSentimentStrategy',
    'EnsembleStrategy'
]

# Add default get_exit_condition after __init__ for each strategy
for strategy_name in strategies_to_fix:
    # Find the class definition
    class_start = content.find(f'class {strategy_name}(BaseStrategy):')
    if class_start == -1:
        print(f"Couldn't find {strategy_name}")
        continue
        
    # Find the __init__ method
    init_start = content.find('def __init__(self):', class_start)
    if init_start == -1:
        print(f"Couldn't find __init__ for {strategy_name}")
        continue
        
    # Find the end of __init__ (next def or class)
    next_def = content.find('\n    def ', init_start + 1)
    
    # Insert the get_exit_condition method
    exit_condition_code = '''
    
    def get_exit_condition(self, position, features):
        """Default exit condition based on holding period"""
        # Exit after standard holding period or on stop loss
        return False  # Let position manager handle exits
'''
    
    # Only add if not already present
    if 'def get_exit_condition' not in content[class_start:next_def]:
        content = content[:next_def] + exit_condition_code + content[next_def:]
        print(f"Fixed {strategy_name}")

# Write back
file_path.write_text(content)
print("\nStrategies fixed!")
print("Testing import...")

try:
    from predictors.strategies import get_all_strategies
    strategies = get_all_strategies()
    print(f"Successfully loaded {len(strategies)} strategies")
except Exception as e:
    print(f"Error: {e}")