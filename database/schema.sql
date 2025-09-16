-- Stock Predictor Database Schema
-- SQLite Database Design

-- Strategies table
CREATE TABLE IF NOT EXISTS strategies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    description TEXT,
    min_hold_days INTEGER DEFAULT 1,
    max_hold_days INTEGER DEFAULT 30,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tickers table
CREATE TABLE IF NOT EXISTS tickers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    name TEXT,
    sector TEXT,
    market_cap REAL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    strategy_id INTEGER NOT NULL,
    ticker_id INTEGER NOT NULL,
    position TEXT CHECK(position IN ('LONG', 'SHORT', 'HOLD')) NOT NULL,
    confidence REAL CHECK(confidence >= 0 AND confidence <= 1),
    score REAL,
    target_1w REAL,
    target_3d REAL,
    target_2w REAL,
    target_1m REAL,
    timeframe TEXT DEFAULT '1W',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    FOREIGN KEY (ticker_id) REFERENCES tickers(id),
    UNIQUE(date, strategy_id, ticker_id)
);

-- Trades table (tracks actual trades based on predictions)
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER NOT NULL,
    entry_date DATE,
    entry_price REAL,
    exit_date DATE,
    exit_price REAL,
    status TEXT CHECK(status IN ('PENDING', 'ACTIVE', 'CLOSED', 'STOPPED', 'CANCELLED')) DEFAULT 'PENDING',
    shares INTEGER DEFAULT 100,
    pnl REAL,
    pnl_percent REAL,
    exit_reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id)
);

-- Price history table
CREATE TABLE IF NOT EXISTS price_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker_id INTEGER NOT NULL,
    date DATE NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL NOT NULL,
    volume INTEGER,
    adjusted_close REAL,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id),
    UNIQUE(ticker_id, date)
);

-- Performance metrics table (daily snapshots)
CREATE TABLE IF NOT EXISTS performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    date DATE NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL,
    avg_win REAL,
    avg_loss REAL,
    profit_factor REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    total_return REAL,
    total_pnl REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id),
    UNIQUE(strategy_id, date)
);

-- Volume leaders table
CREATE TABLE IF NOT EXISTS volume_leaders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    timeframe TEXT CHECK(timeframe IN ('daily', 'weekly', 'monthly')) NOT NULL,
    rank INTEGER NOT NULL,
    ticker_id INTEGER NOT NULL,
    avg_volume INTEGER,
    avg_price REAL,
    dollar_volume REAL,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id),
    UNIQUE(date, timeframe, rank)
);

-- Backtest results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id INTEGER NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_capital REAL DEFAULT 10000,
    final_capital REAL,
    total_return REAL,
    annual_return REAL,
    sharpe_ratio REAL,
    sortino_ratio REAL,
    max_drawdown REAL,
    max_drawdown_duration INTEGER,
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    win_rate REAL,
    profit_factor REAL,
    avg_trade_return REAL,
    best_trade REAL,
    worst_trade REAL,
    avg_holding_days REAL,
    commission_paid REAL,
    parameters TEXT,  -- JSON string of strategy parameters
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (strategy_id) REFERENCES strategies(id)
);

-- Consensus picks table (tracks when multiple strategies agree)
CREATE TABLE IF NOT EXISTS consensus_picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    ticker_id INTEGER NOT NULL,
    long_votes INTEGER DEFAULT 0,
    short_votes INTEGER DEFAULT 0,
    total_votes INTEGER DEFAULT 0,
    consensus_position TEXT,
    avg_confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker_id) REFERENCES tickers(id),
    UNIQUE(date, ticker_id)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(date);
CREATE INDEX IF NOT EXISTS idx_predictions_strategy ON predictions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker_id);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_date ON trades(entry_date);
CREATE INDEX IF NOT EXISTS idx_price_history_date ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_price_history_ticker ON price_history(ticker_id);
CREATE INDEX IF NOT EXISTS idx_performance_date ON performance_metrics(date);
CREATE INDEX IF NOT EXISTS idx_volume_leaders_date ON volume_leaders(date);

-- Create views for common queries
CREATE VIEW IF NOT EXISTS active_positions AS
SELECT
    t.id as trade_id,
    p.date as prediction_date,
    s.name as strategy,
    tk.symbol as ticker,
    p.position,
    p.confidence,
    t.entry_date,
    t.entry_price,
    t.shares,
    t.status
FROM trades t
JOIN predictions p ON t.prediction_id = p.id
JOIN strategies s ON p.strategy_id = s.id
JOIN tickers tk ON p.ticker_id = tk.id
WHERE t.status = 'ACTIVE';

CREATE VIEW IF NOT EXISTS strategy_performance AS
SELECT
    s.name as strategy,
    COUNT(t.id) as total_trades,
    SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN t.pnl < 0 THEN 1 ELSE 0 END) as losses,
    ROUND(AVG(t.pnl_percent), 2) as avg_return,
    ROUND(SUM(t.pnl), 2) as total_pnl,
    ROUND(MAX(t.pnl_percent), 2) as best_trade,
    ROUND(MIN(t.pnl_percent), 2) as worst_trade
FROM strategies s
LEFT JOIN predictions p ON s.id = p.strategy_id
LEFT JOIN trades t ON p.id = t.prediction_id
WHERE t.status = 'CLOSED'
GROUP BY s.id;