# Tax Loss Harvesting System - Design Document

## Overview

This system implements an automated tax loss harvesting (TLH) strategy for S&P 500 index tracking using a 100-stock sampling approach. The system is designed with a unified architecture that allows the same core strategy logic to be used across three execution modes: backtesting, paper trading, and live trading.

---

## 1. Key Algorithmic Decisions

### 1.1 Portfolio Construction

**Decision: 100-Stock Sampling from S&P 500**

The system uses a subset of 100 stocks from the S&P 500 rather than all 500 stocks. This approach:
- Reduces transaction costs and complexity
- Maintains diversification through sector allocation
- Allows for meaningful tax loss harvesting opportunities

**Selection Algorithm:**
1. **Top 50 by Market Cap**: Selects the 50 largest companies by market capitalization to capture the majority of index weight
2. **Sector Diversification**: Distributes the remaining 50 positions across sectors, selecting up to 5 stocks per sector from the remaining pool
3. **Market Cap Tiers**: Categorizes stocks into three tiers:
   - **Mega Cap**: > $200B market cap
   - **Large Cap**: $10B - $200B market cap
   - **Mid Cap**: < $10B market cap

**Rationale**: This hybrid approach balances index representation (via large caps) with diversification (via sector distribution), ensuring the portfolio tracks the S&P 500 while maintaining sufficient liquidity and trading opportunities.

### 1.2 Loss Detection Threshold

**Decision: 0.5% Loss Threshold**

The system harvests losses when a position declines by 0.5% or more from its cost basis.

**Calculation:**
```
pnl_pct = (current_value - cost_basis) / cost_basis
should_harvest = pnl_pct < -0.005  # 0.5% threshold
```

**Rationale**: 
- Too low (e.g., 0.1%): Excessive trading, high transaction costs, wash sale complications
- Too high (e.g., 2%): Misses many harvesting opportunities, reduces tax alpha
- 0.5%: Balances harvesting frequency with transaction costs and wash sale management

### 1.3 Replacement Stock Selection

**Decision: Multi-Stage Filtering with Correlation-Based Selection**

The replacement selection algorithm uses a cascading filter approach:

**Stage 1: Filtering**
1. Exclude symbols currently in portfolio
2. Exclude symbols in wash sale period (30 days)
3. Exclude the symbol being sold

**Stage 2: Sector and Market Cap Matching**
1. **Preferred**: Same sector + same market cap tier
2. **Fallback 1**: Same market cap tier only
3. **Fallback 2**: All available candidates

**Stage 3: Correlation Analysis**
- Calculates 90-day rolling correlation of daily returns between sold stock and candidates
- Requires minimum 20 overlapping trading days
- Selects candidate with highest correlation to maintain index tracking

**Stage 4: Market Cap Fallback**
- If correlation calculation fails (insufficient data), falls back to selecting highest market cap candidate
- Last resort: first available symbol

**Rationale**: 
- **Sector matching**: Maintains sector exposure similar to the original position
- **Market cap tier matching**: Preserves size factor exposure
- **Correlation-based selection**: Ensures replacement stock moves similarly to the sold stock, maintaining index tracking
- **Cascading fallbacks**: Ensures a replacement is always found if candidates exist

### 1.4 Wash Sale Management

**Decision: 30-Day Wash Sale Window**

The system tracks wash sales and prevents repurchasing the same symbol within 30 days of a loss sale.

**Implementation:**
- Records sale date when a loss is realized
- Checks if symbol is in wash sale period before allowing purchase
- Uses calendar days (not trading days) for compliance with IRS rules

**Rationale**: IRS wash sale rules disallow tax deduction if substantially identical securities are purchased within 30 days before or after a loss sale. The 30-day window ensures compliance while allowing the system to find suitable replacements.

### 1.5 Correlation Calculation

**Decision: 90-Day Rolling Correlation of Daily Returns**

**Method:**
1. Retrieves historical prices for target and all candidates over 90-day lookback period
2. Calculates daily percentage returns: `returns = prices.pct_change()`
3. Aligns returns on common trading dates
4. Calculates Pearson correlation coefficient
5. Requires minimum 20 overlapping trading days for validity

**Data Quality Handling:**
- Removes NaN and infinite values
- Only uses dates where all symbols have data
- Returns empty series if insufficient data

**Rationale**: 
- 90 days provides sufficient data for stable correlation estimates
- Daily returns capture short-term co-movement
- Minimum data requirement ensures statistical validity
- Alignment on common dates prevents spurious correlations from missing data

### 1.6 Harvest Frequency

**Decision: Weekly Harvesting (Configurable)**

The system can run harvest cycles at different frequencies:
- **Daily**: Check and harvest every trading day
- **Weekly**: Check and harvest on Mondays (default)
- **Monthly**: Check and harvest on first trading day of month

**Rationale**: 
- **Weekly**: Balances responsiveness to market movements with transaction cost management
- Prevents over-trading while capturing most significant loss opportunities
- Allows time for positions to develop meaningful losses

---

## 2. Architecture

### 2.1 Design Principles

The architecture follows these key principles:

1. **Separation of Concerns**: Strategy logic is completely separate from execution
2. **Stateless Strategy**: Core strategy is pure functions with no side effects
3. **Unified Interface**: All execution modes implement the same interface
4. **Testability**: Strategy can be tested independently of execution
5. **Consistency**: Same strategy logic across all execution modes

### 2.2 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Core Strategy (strategy/core.py)           │
│  - Stock selection                                      │
│  - Loss detection                                       │
│  - Replacement finding                                  │
│  - Wash sale tracking                                   │
│  - Correlation calculation                              │
│                                                         │
│  Pure business logic - no I/O, no API calls            │
└─────────────────────────────────────────────────────────┘
                        ▲
                        │ uses
        ┌───────────────┼───────────────┐
        │               │               │
┌───────┴──────┐ ┌──────┴──────┐ ┌─────┴──────┐
│  Backtest    │ │   Paper     │ │   Live     │
│   Engine     │ │   Engine    │ │   Engine   │
│              │ │             │ │            │
│ Uses:        │ │ Uses:       │ │ Uses:      │
│ - Historical │ │ - Alpaca    │ │ - Alpaca   │
│   data       │ │   Paper API │ │   Live API │
│ - Simulated  │ │ - Real      │ │ - Real     │
│   trades     │ │   orders    │ │   orders   │
└─────────────┘ └─────────────┘ └────────────┘
```

### 2.3 Component Overview

#### 2.3.1 Core Strategy (`strategy/core.py`)

**TLHStrategy Class** - Stateless strategy implementation

**Key Methods:**
- `select_stocks()`: Portfolio construction algorithm
- `should_harvest()`: Loss detection logic
- `find_replacement()`: Replacement selection algorithm
- `calculate_correlations()`: Correlation calculation
- `is_wash_sale_active()`: Wash sale checking

**Characteristics:**
- All methods are static (no instance state)
- Pure functions: inputs → outputs, no side effects
- No file I/O, no API calls, no logging
- Can be tested in isolation

#### 2.3.2 Execution Interface (`strategy/execution.py`)

**ExecutionEngine Abstract Base Class** - Defines common interface

**Required Methods:**
- `get_price(symbol, date)`: Get current/historical price
- `get_prices_range(symbol, start, end)`: Get historical price data
- `execute_buy(symbol, quantity, price, date)`: Execute buy order
- `execute_sell(symbol, quantity, price, date)`: Execute sell order
- `get_positions()`: Get current portfolio positions
- `get_metadata(symbols)`: Get stock metadata
- `get_available_symbols()`: Get list of tradeable symbols

**Purpose**: Ensures all execution engines provide the same interface, allowing strategy to work with any execution mode.

#### 2.3.3 Execution Engines

**BacktestExecutionEngine** (`backtesting/backtest_execution_engine.py`)
- Uses historical data from `BacktestDataLoader`
- Simulates trades (no real API calls)
- Tracks cash, positions, transaction costs
- Supports date-based backtesting

**PaperTradeExecutionEngine** (`strategy/paper_trade_engine.py`)
- Uses Alpaca Paper Trading API
- Real API calls but no real money
- Same interface as backtest engine
- Useful for testing with real market data

**LiveTradeExecutionEngine** (`strategy/live_trade_engine.py`)
- Uses Alpaca Live Trading API
- Real API calls with real money
- Additional safety checks:
  - Trade confirmation prompts
  - Daily trade limits (50 trades/day)
  - Enhanced logging and warnings

#### 2.3.4 Configuration (`strategy/config.py`)

**TLHConfig Class** - Unified configuration

**Key Parameters:**
- `NUM_STOCKS = 100`: Portfolio size
- `HARVEST_THRESHOLD = 0.005`: 0.5% loss threshold
- `WASH_SALE_DAYS = 30`: Wash sale window
- `CORRELATION_LOOKBACK_DAYS = 90`: Correlation lookback period
- `HARVEST_FREQUENCY = 'weekly'`: Harvest schedule
- `SHORT_TERM_TAX_RATE = 0.37`: Tax rate for backtesting
- `SPREAD_BPS = 1`: Transaction cost (1 basis point)

**Rationale**: Single source of truth for all configuration ensures consistency across execution modes.

#### 2.3.5 Data Models (`strategy/models.py`)

**Common Data Structures:**
- `Position`: Portfolio position (symbol, quantity, cost_basis, purchase_date)
- `StockMetadata`: Stock information (symbol, name, sector, industry, market_cap)

**Purpose**: Shared data models ensure compatibility across execution engines.

### 2.4 Data Flow

#### 2.4.1 Backtest Flow

```
1. Load historical data (BacktestDataLoader)
   ↓
2. Initialize BacktestExecutionEngine with data
   ↓
3. For each trading date:
   a. Check if harvest should run (frequency check)
   b. For each position:
      - Get current price
      - Calculate P&L using TLHStrategy.should_harvest()
      - If loss > threshold:
         * Find replacement using TLHStrategy.find_replacement()
         * Execute sell via BacktestExecutionEngine
         * Execute buy via BacktestExecutionEngine
   c. Record portfolio value
   ↓
4. Calculate results (returns, tax alpha, risk metrics)
   ↓
5. Generate visualizations and reports
```

#### 2.4.2 Paper/Live Trading Flow

```
1. Initialize execution engine (PaperTradeExecutionEngine or LiveTradeExecutionEngine)
   ↓
2. Load current positions from execution engine
   ↓
3. For each position:
   a. Get current price from execution engine
   b. Calculate P&L using TLHStrategy.should_harvest()
   c. If loss > threshold:
      * Find replacement using TLHStrategy.find_replacement()
      * Execute sell via execution engine
      * Execute buy via execution engine
      * Update local position tracking
   ↓
4. Save positions and transactions to disk
```

### 2.5 File Structure

```
edi/
├── strategy/
│   ├── core.py                    # TLHStrategy (pure logic)
│   ├── config.py                  # Unified configuration
│   ├── execution.py               # ExecutionEngine interface
│   ├── models.py                  # Common data models
│   ├── paper_trade_engine.py      # Paper trading execution
│   ├── live_trade_engine.py       # Live trading execution
│   └── di_tlh_sp500_sampling.py   # CLI wrapper
│
├── backtesting/
│   ├── backtest_engine.py         # Backtest runner
│   ├── backtest_execution_engine.py  # Backtest execution
│   └── backtest_data_downloader.py   # Historical data download
│
├── data/                          # Runtime data (positions, transactions)
├── backtest_data/                 # Historical price data
└── backtest_results/              # Backtest output files
```

### 2.6 Key Design Patterns

1. **Strategy Pattern**: Core strategy is separated from execution
2. **Adapter Pattern**: Execution engines adapt different data sources to common interface
3. **Template Method**: Execution engines implement same interface with different behaviors
4. **Dependency Injection**: Strategy receives execution engine as dependency

---

## 3. Usage Manual

### 3.1 Prerequisites

**Required Environment Variables:**
```bash
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"  # or live URL
export FMP_API_KEY="your_fmp_key"  # For metadata
```

**Or use the provided setup script:**
```bash
source setup_env.sh
```

**Python Dependencies:**
- pandas, numpy
- requests
- yfinance (for backtest data)
- matplotlib, seaborn (for backtest visualization)

### 3.2 Execution Modes

The system supports three execution modes:

1. **Backtest**: Test strategy on historical data
2. **Paper Trading**: Test with real market data but no real money
3. **Live Trading**: Execute with real money (use with caution!)

### 3.3 Backtesting

**Purpose**: Validate strategy performance on historical data before risking real money.

**Setup:**
```bash
# Download historical data (first time only)
cd backtesting
python backtest_data_downloader.py
```

**Run Backtest:**
```bash
cd backtesting
python backtest_engine.py
```

**Output:**
- Console: Performance metrics, trade statistics
- Files:
  - `backtest_results/backtest_results_YYYYMMDD_HHMMSS.json`: Results data
  - `backtest_results/backtest_results_YYYYMMDD_HHMMSS.png`: Performance charts
  - `backtest_results/backtest_results_YYYYMMDD_HHMMSS_trades.csv`: Trade history

**Metrics Calculated:**
- Total return vs S&P 500 benchmark
- Annualized return
- Tax alpha (value of harvested losses)
- Sharpe ratio
- Maximum drawdown
- Tracking error
- Transaction costs impact

### 3.4 Paper Trading

**Purpose**: Test strategy with real market data and API calls, but no real money at risk.

**Initial Setup:**
```bash
python strategy/di_tlh_sp500_sampling.py --setup --capital 10000 --mode paper
```

This will:
1. Fetch S&P 500 constituent list
2. Download stock metadata (sector, market cap, etc.)
3. Select 100 stocks using core strategy
4. Buy initial positions via Alpaca Paper Trading API

**Run Harvest Cycle:**
```bash
python strategy/di_tlh_sp500_sampling.py --harvest --mode paper
```

This will:
1. Check all current positions for losses
2. Harvest losses exceeding threshold
3. Find and buy replacement stocks
4. Update positions and transaction records

**Data Storage:**
- `data/positions.json`: Current portfolio positions
- `data/transactions.json`: All executed trades
- `data/wash_sales.json`: Wash sale tracking
- `data/sp500_metadata.json`: Cached stock metadata

**Monitoring:**
- Check Alpaca paper trading dashboard
- Review transaction logs in `data/transactions.json`
- Monitor positions in `data/positions.json`

### 3.5 Live Trading

**Warning**: Live trading uses real money. Use with extreme caution!

**Setup:**
1. Ensure `ALPACA_BASE_URL` points to live trading: `https://api.alpaca.markets`
2. Verify API keys are for live account (not paper)
3. Start with small capital amounts

**Initial Setup:**
```bash
python strategy/di_tlh_sp500_sampling.py --setup --capital 1000 --mode live
```

**Run Harvest Cycle:**
```bash
python strategy/di_tlh_sp500_sampling.py --harvest --mode live
```

**Safety Features:**
- Trade confirmation prompts (can be disabled)
- Daily trade limit (50 trades/day)
- Enhanced logging with warnings
- Real-time position tracking

**Best Practices:**
1. Always backtest first
2. Test extensively in paper trading
3. Start with minimal capital
4. Monitor closely for first few weeks
5. Review all trades daily
6. Keep detailed records for tax reporting

### 3.6 Tax Reporting

**Generate Form 8949 Report:**
```bash
python strategy/di_tlh_sp500_sampling.py --tax-report --year 2024
```

**Output:**
- Console: Formatted tax report
- File: `data/form_8949_YYYY.txt`

**Report Includes:**
- All sales transactions for the year
- Cost basis and proceeds
- Realized gains/losses
- Total realized losses
- Note: Consult a tax professional for official filing

### 3.7 Configuration

**Modify Strategy Parameters:**

Edit `strategy/config.py`:

```python
class TLHConfig:
    NUM_STOCKS = 100                    # Portfolio size
    HARVEST_THRESHOLD = 0.005          # 0.5% loss threshold
    WASH_SALE_DAYS = 30                # Wash sale window
    CORRELATION_LOOKBACK_DAYS = 90      # Correlation period
    HARVEST_FREQUENCY = 'weekly'        # 'daily', 'weekly', 'monthly'
```

**Backtest-Specific Parameters:**

Edit `backtesting/backtest_engine.py`:

```python
class BacktestParams:
    INITIAL_CAPITAL = 100000.0          # Starting capital
    SHORT_TERM_TAX_RATE = 0.37          # Tax rate for calculations
    SPREAD_BPS = 1                      # Transaction cost
```

### 3.8 Troubleshooting

**Common Issues:**

1. **"No replacement candidates found"**
   - Cause: All candidates in wash sale or already in portfolio
   - Solution: System will skip harvest for that position

2. **"Insufficient data to calculate correlations"**
   - Cause: Not enough historical price data
   - Solution: System falls back to market cap ranking

3. **"Invalid price for symbol"**
   - Cause: Market closed, symbol delisted, or data unavailable
   - Solution: System skips that symbol and continues

4. **API Rate Limits**
   - Cause: Too many API calls
   - Solution: System includes rate limiting and retries

5. **Import Errors**
   - Cause: Missing dependencies or incorrect paths
   - Solution: Ensure all dependencies installed, check Python path

### 3.9 Workflow Recommendations

**Development Workflow:**

1. **Backtest First**
   ```bash
   python backtesting/backtest_engine.py
   ```
   - Review performance metrics
   - Check trade frequency
   - Validate strategy logic

2. **Paper Trade**
   ```bash
   python strategy/di_tlh_sp500_sampling.py --setup --capital 10000 --mode paper
   python strategy/di_tlh_sp500_sampling.py --harvest --mode paper
   ```
   - Monitor for several weeks
   - Compare results to backtest
   - Verify API integration

3. **Live Trade (if confident)**
   ```bash
   python strategy/di_tlh_sp500_sampling.py --setup --capital 1000 --mode live
   python strategy/di_tlh_sp500_sampling.py --harvest --mode live
   ```
   - Start small
   - Monitor closely
   - Scale up gradually

**Production Workflow:**

1. **Daily/Weekly**: Run harvest cycle
   ```bash
   python strategy/di_tlh_sp500_sampling.py --harvest --mode paper
   ```

2. **Monthly**: Review performance and positions

3. **Quarterly**: Review tax implications

4. **Yearly**: Generate tax reports
   ```bash
   python strategy/di_tlh_sp500_sampling.py --tax-report --year 2024
   ```

### 3.10 Command Reference

**Main CLI Commands:**

```bash
# Setup portfolio
python strategy/di_tlh_sp500_sampling.py --setup --capital <amount> [--mode paper|live]

# Run harvest cycle
python strategy/di_tlh_sp500_sampling.py --harvest [--mode paper|live]

# Generate tax report
python strategy/di_tlh_sp500_sampling.py --tax-report [--year YYYY]

# Show help
python strategy/di_tlh_sp500_sampling.py --help
```

**Backtest Commands:**

```bash
# Download historical data
python backtesting/backtest_data_downloader.py

# Run backtest
python backtesting/backtest_engine.py
```

---

## Appendix: Key Metrics Explained

### Tax Alpha
The value of tax savings from harvested losses. Calculated as:
```
tax_alpha = total_losses_harvested × short_term_tax_rate
```

### Tracking Error
Standard deviation of the difference between portfolio returns and benchmark (SPY) returns. Lower is better.

### Sharpe Ratio
Risk-adjusted return metric:
```
sharpe_ratio = (annualized_return - risk_free_rate) / volatility
```
Higher is better. Assumes 4% risk-free rate.

### Maximum Drawdown
Largest peak-to-trough decline in portfolio value. Expressed as negative percentage.

---

## Version History

- **v1.0**: Initial unified architecture implementation
  - Core strategy extraction
  - Three execution engines (backtest, paper, live)
  - Unified configuration
  - CLI interface

---

## Contact & Support

For issues or questions, review the code comments and error messages. The system includes extensive logging to help diagnose issues.

