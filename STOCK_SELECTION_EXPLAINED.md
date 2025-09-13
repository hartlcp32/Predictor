# How Stock Selection Works

## The 10 Stocks Pool
We focus on the 10 most liquid, widely-traded stocks:
- **AAPL** (Apple)
- **MSFT** (Microsoft) 
- **GOOGL** (Google)
- **AMZN** (Amazon)
- **NVDA** (Nvidia)
- **META** (Meta/Facebook)
- **TSLA** (Tesla)
- **BRK-B** (Berkshire Hathaway)
- **JPM** (JPMorgan Chase)
- **JNJ** (Johnson & Johnson)

## Two Selection Approaches

### 1. Diverse Selection (Default)
Each strategy picks a DIFFERENT stock to maximize variety:

```
Strategy 1: Momentum        → Evaluates all 10 → Picks AAPL (best signal)
Strategy 2: Mean Reversion  → Evaluates all 10 → Picks MSFT (best unpicked)
Strategy 3: Volume Breakout → Evaluates all 10 → Picks NVDA (best unpicked)
...and so on
```

**Pros:** 
- Maximum diversification
- Each strategy gets its preferred pick
- Reduces concentration risk

**Cons:**
- Might miss if multiple strategies agree on one stock

### 2. Concentrated Selection (Alternative)
Each strategy picks its ABSOLUTE BEST, overlap allowed:

```
Strategy 1: Momentum        → Picks AAPL (score: 0.8)
Strategy 2: Mean Reversion  → Picks TSLA (score: 0.7)
Strategy 3: Volume Breakout → Picks AAPL (score: 0.9)
Strategy 4: Technical       → Picks AAPL (score: 0.6)
...
Result: AAPL picked by 3 strategies (high consensus)
```

**Pros:**
- Identifies stocks with multiple bullish/bearish signals
- Stronger conviction when strategies agree
- Natural validation through consensus

**Cons:**
- Less diversification
- Could amplify errors if multiple strategies wrong

## How Each Strategy Evaluates Stocks

Each strategy calculates a score for every stock based on its indicators:

### Example: Momentum Strategy evaluating AAPL
1. Check 5-day moving average vs 20-day → **+0.3 points**
2. Check if price above 20-day MA → **+0.2 points**
3. Check volume surge → **+0.1 points**
4. **Total Score: 0.6 → Position: LONG**

### Example: Mean Reversion evaluating TSLA
1. Check RSI (oversold at 25) → **+0.4 points**
2. Price 8% below 20-day MA → **+0.3 points**
3. Near lower Bollinger Band → **+0.3 points**
4. **Total Score: 1.0 → Position: LONG (oversold bounce)**

## The Selection Process

```python
For each strategy:
    1. Calculate scores for all 10 stocks
    2. Rank them by signal strength
    3. Pick the highest scoring that signals LONG or SHORT
    4. If diverse mode: skip already-picked stocks
    5. If concentrated: always pick the best
```

## Real Example Output

**Monday Predictions:**
```
Momentum        → AAPL  LONG  +4.5%  (1W)
Mean Reversion  → TSLA  LONG  +3.2%  (1W)  
Volume Breakout → NVDA  LONG  +5.8%  (1W)
Technical       → MSFT  SHORT -2.7%  (1W)
Pattern         → GOOGL LONG  +3.1%  (1W)
Volatility      → AMZN  SHORT -4.2%  (1W)
MA Crossover    → META  LONG  +3.8%  (1W)
Support/Resist  → JPM   LONG  +2.4%  (1W)
Sentiment       → BRK-B HOLD  0%     (-)
Ensemble        → AAPL  LONG  +4.2%  (1W)

Consensus: AAPL (3 strategies), NVDA (2 strategies)
```

## Why These 10 Stocks?

1. **Liquidity**: Highest daily volume = easy to trade
2. **Data Quality**: Most analyzed = reliable indicators
3. **Market Leaders**: Represent major sectors
4. **Lower Spreads**: Tight bid/ask = better execution
5. **Options Available**: Can hedge positions
6. **Global Names**: Less susceptible to single-event risk

## Customization Options

You can modify the stock list in `data_fetcher.py`:
```python
self.stocks = ['AAPL', 'MSFT', ...]  # Change to any tickers
```

Or use sector-specific lists:
- **Tech Focus**: AAPL, MSFT, GOOGL, NVDA, META, CRM, ORCL, ADBE, INTC, AMD
- **Financial**: JPM, BAC, WFC, GS, MS, C, USB, PNC, TFC, COF
- **Healthcare**: JNJ, UNH, PFE, ABBV, TMO, MRK, ABT, DHR, CVS, MDT