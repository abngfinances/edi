# Copilot / AI Agent Instructions for EDI (Tax Loss Harvesting)

Purpose: Help AI coding assistants become productive quickly by summarizing architecture, key workflows, conventions, and integration points discovered in the repository.

- **Big picture**: The repo implements a Tax Loss Harvesting (TLH) framework with three clear layers:
  - **Strategy core**: Pure business logic in `strategy/core.py` (class `TLHStrategy`) — stateless, pure functions that make harvest and replacement decisions.
  - **Execution engines**: Pluggable `ExecutionEngine` interface in `strategy/execution.py` with concrete implementations:
    - Backtesting: `backtesting/backtest_execution_engine.py` (uses `BacktestDataLoader` and historical prices)
    - Paper trading: `strategy/paper_trade_engine.py` (Alpaca paper API wrapper)
    - Live trading: `strategy/live_trade_engine.py` (adds safety checks on top of paper engine)
  - **Backtest runner & data**: `backtesting/backtest_engine.py` runs simulations; `backtesting/backtest_data_downloader.py` builds datasets saved under `backtest_data/` (parquet/csv + checkpointed temp files).

- **Why this structure matters**:
  - Strategy core is intentionally side-effect-free so it can be used across backtest, paper, and live flows without code duplication.
  - The `ExecutionEngine` abstraction is the primary integration boundary — any new execution mode must implement that interface.

- **Key files to inspect for behavior examples**:
  - Strategy logic: [strategy/core.py](strategy/core.py)
  - Execution interface: [strategy/execution.py](strategy/execution.py)
  - Backtest runner: [backtesting/backtest_engine.py](backtesting/backtest_engine.py)
  - Backtest data loader: [backtesting/backtest_data_downloader.py](backtesting/backtest_data_downloader.py)
  - Backtest execution engine: [backtesting/backtest_execution_engine.py](backtesting/backtest_execution_engine.py)
  - Paper/live engines: [strategy/paper_trade_engine.py](strategy/paper_trade_engine.py), [strategy/live_trade_engine.py](strategy/live_trade_engine.py)

- **Important patterns / conventions (explicit, discoverable)**:
  - Strategy functions expect plain data structures (dicts, lists) and a `price_data_func(symbol, start, end)` when historical prices are needed (see `TLHStrategy.calculate_correlations`).
  - Execution engines use `set_current_date(date)` before price/execute calls in backtests — call order matters (`set_current_date` → `get_price` / `execute_buy` / `execute_sell`).
  - Historical price range must return a DataFrame with a `close` column (used for returns/correlations).
  - Metadata format: mapping `symbol -> { 'market_cap', 'sector', 'industry', ... }` (see `backtest_data_downloader` outputs and `TLHStrategy.select_stocks`).
  - Backtest code tolerates missing prices by attempting nearby dates (up to 5 days back); tests/changes must keep compatible fallback behavior.

- **Developer workflows and commands (how to run common tasks)**:
  - Build backtest dataset (download + checkpointing): run `python backtesting/backtest_data_downloader.py` (this script writes into `backtest_data/` and uses `BacktestConfig`).
  - Run a backtest: run `python backtesting/backtest_engine.py` (this constructs `TLHBacktester` and uses `BacktestExecutionEngine`).
  - Run paper/live runners: import and instantiate `PaperTradeRunner` or `LiveTradeRunner` from the respective files and call `run_harvest()`; live runners require Alpaca credentials configured in `strategy/config.py`.
  - Data files used by backtests: `backtest_data/prices/all_prices.parquet` (or CSV) and `backtest_data/metadata/stock_metadata.json`.

- **External integrations & dependencies**:
  - Historical data: `yfinance` (primary), with fallbacks noted in `backtest_data_downloader.py`.
  - Trading API: Alpaca (paper and live). `strategy/paper_trade_engine.py` contains `AlpacaClient`.
  - Results/plots: matplotlib / seaborn used inside `backtesting/backtest_engine.py`.

- **Tests / validation notes (what to look for when changing code)**:
  - Keep `TLHStrategy` pure — avoid introducing I/O or engine-specific behavior into `strategy/core.py`.
  - When modifying `ExecutionEngine` interface, update all implementers (`backtest`, `paper`, `live`) and adjust callers in `backtesting/backtest_engine.py` and live/paper runners.
  - Backtest reproducibility relies on `BacktestDataLoader` checkpoint files under `backtest_data/checkpoints/` — do not change paths without updating `BacktestConfig`.

- **Small examples (copy/paste ready)**
  - Call strategy correlation function (used in backtest replacement logic):

    price_data = lambda s, a, b: backtest_execution_engine.get_prices_range(s, a, b)
    corr = TLHStrategy.calculate_correlations('AAPL', ['MSFT','GOOGL'], price_data, 90, current_date=some_date)

  - Initialize backtest and run an initial portfolio:

    from backtesting.backtest_engine import TLHBacktester, BacktestParams
    bt = TLHBacktester(BacktestParams)
    bt.initialize_portfolio(start_date, symbols)

- **When in doubt**:
  - Check `strategy/config.py` for global constants (tax rates, thresholds, commission, file paths).
  - Prefer editing `TLHStrategy` for algorithm changes, and `ExecutionEngine` implementations for environment-specific behavior.

Please review this draft for missing integration details (credentials, exact run flags) or any local scripts you use for CI/test runs; I can merge updates and iterate.
