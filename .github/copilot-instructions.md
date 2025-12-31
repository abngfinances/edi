# Copilot / AI Agent Instructions for EDI (Tax Loss Harvesting)

Purpose: Help AI coding assistants become productive quickly by summarizing architecture, key workflows, conventions, and integration points discovered in the repository.

- **Big picture**: The repo implements a Tax Loss Harvesting (TLH) framework with three clear layers:
  - **Strategy core**: Pure business logic in `strategy/core.py` (class `TLHStrategy`) — stateless, pure functions that make harvest and replacement decisions.
  - **Execution engines**: Pluggable `ExecutionEngine` interface in `strategy/execution.py` with concrete implementations:
    - Backtesting: `backtesting/backtest_execution_engine.py` (uses `BacktestDataLoader` and historical prices)
    - Paper trading: `strategy/paper_trade_engine.py` (Alpaca paper API wrapper)
    - Live trading: `strategy/live_trade_engine.py` (adds safety checks on top of paper engine)
  - **Backtest runner & data**: `backtesting/backtest_engine.py` runs simulations; `backtesting/backtest_data_downloader.py` builds datasets saved under `backtest_data/` (parquet/csv + checkpointed temp files).
  - **Data management tools**: `backtesting/index_downloader.py` downloads S&P 500 constituents; `backtesting/metadata_downloader.py` manages stock metadata with incremental updates, validation, and ignore-symbols mechanism for handling problematic tickers.

- **Why this structure matters**:
  - Strategy core is intentionally side-effect-free so it can be used across backtest, paper, and live flows without code duplication.
  - The `ExecutionEngine` abstraction is the primary integration boundary — any new execution mode must implement that interface.

- **Key files to inspect for behavior examples**:
  - Strategy logic: [strategy/core.py](strategy/core.py)
  - Execution interface: [strategy/execution.py](strategy/execution.py)
  - Backtest runner: [backtesting/backtest_engine.py](backtesting/backtest_engine.py)
  - Backtest data loader: [backtesting/backtest_data_downloader.py](backtesting/backtest_data_downloader.py)
  - Backtest execution engine: [backtesting/backtest_execution_engine.py](backtesting/backtest_execution_engine.py)
  - Data management: [backtesting/index_downloader.py](backtesting/index_downloader.py), [backtesting/metadata_downloader.py](backtesting/metadata_downloader.py)
  - Paper/live engines: [strategy/paper_trade_engine.py](strategy/paper_trade_engine.py), [strategy/live_trade_engine.py](strategy/live_trade_engine.py)

- **Important patterns / conventions (explicit, discoverable)**:
  - Strategy functions expect plain data structures (dicts, lists) and a `price_data_func(symbol, start, end)` when historical prices are needed (see `TLHStrategy.calculate_correlations`).
  - Execution engines use `set_current_date(date)` before price/execute calls in backtests — call order matters (`set_current_date` → `get_price` / `execute_buy` / `execute_sell`).
  - Historical price range must return a DataFrame with a `close` column (used for returns/correlations).
  - Metadata format:
    - Constituents: `{ 'symbols': [...], 'metadata': { 'index_symbol', 'download_timestamp', 'total_holdings' } }`
    - Stock metadata: `symbol -> { 'symbol', 'name', 'sector', 'industry', 'market_cap', 'exchange', 'currency', 'last_updated' }`
    - Validation enforces exact field set (no extra fields allowed)
  - CLI tools use `argparse` with detailed help, `logging.basicConfig` for consistent formatting, `sys.exit(main())` for exit codes (0=success, 1=error).
  - Backtest code tolerates missing prices by attempting nearby dates (up to 5 days back); tests/changes must keep compatible fallback behavior.

- **Developer workflows and commands (how to run common tasks)**:
  - Build backtest dataset (3 steps):
    1. Download S&P 500 constituents: `python backtesting/index_downloader.py SPY --output-dir backtest_data`
    2. Download stock metadata: `python backtesting/metadata_downloader.py SPY --output-dir backtest_data [--ignore-symbols SYMBOL1,SYMBOL2]`
    3. Download historical prices: `python backtesting/backtest_data_downloader.py`
  - Run a backtest: `python backtesting/backtest_engine.py` (constructs `TLHBacktester` and uses `BacktestExecutionEngine`).
  - Run paper/live runners: import and instantiate `PaperTradeRunner` or `LiveTradeRunner` from the respective files and call `run_harvest()`; live runners require Alpaca credentials configured in `strategy/config.py`.
  - Data files used by backtests:
    - Prices: `backtest_data/prices/all_prices.parquet` (or CSV)
    - Constituents: `backtest_data/metadata/spy_constituents.json`
    - Stock metadata: `backtest_data/metadata/spy_metadata.json`
    - Checkpoints: `backtest_data/checkpoints/` (for download progress tracking)

- **External integrations & dependencies**:
  - Historical data: `yfinance` (primary), with fallbacks noted in `backtest_data_downloader.py`.
  - Trading API: Alpaca (paper and live). `strategy/paper_trade_engine.py` contains `AlpacaClient`.
  - Results/plots: matplotlib / seaborn used inside `backtesting/backtest_engine.py`.

- **Tests / validation notes (what to look for when changing code)**:
  - Keep `TLHStrategy` pure — avoid introducing I/O or engine-specific behavior into `strategy/core.py`.
  - When modifying `ExecutionEngine` interface, update all implementers (`backtest`, `paper`, `live`) and adjust callers in `backtesting/backtest_engine.py` and live/paper runners.
  - Backtest reproducibility relies on `BacktestDataLoader` checkpoint files under `backtest_data/checkpoints/` — do not change paths without updating `BacktestConfig`.

- **Testing best practices (TDD approach used)**:
  - Write tests in phases: file loading → validation → business logic → orchestration → CLI → integration
  - Use pytest fixtures: `tmp_path` for file operations, `capsys` for stdout/stderr, `caplog` for logging output
  - Mock external dependencies with `@patch` decorator (e.g., `yf.Ticker`, `time.sleep`, `sys.argv`)
  - CLI tests should verify: exit codes (0=success, 1=error), output messages, specific error details
  - Integration tests use `@pytest.mark.integration` marker (run with `pytest -m integration`, skip with `pytest -m "not integration"`)
  - Register custom markers in `pytest.ini` to avoid warnings: `markers = integration: marks tests as integration tests`

- **Error handling patterns**:
  - CLI tools return exit code 0 on success, 1 on error (enables shell script integration: `if python script.py; then ...`)
  - `metadata_downloader` validates `failed_symbols == ignore_symbols` exactly (no surprises: unexpected failures OR unnecessary ignores both cause errors)
  - Error messages include actionable fix commands (e.g., "To fix: --ignore-symbols A,B,C" with exact symbols)
  - Use `logger.warning()` for failed symbols list (appears in caplog during tests), `logger.error()` for validation failures
  - Pretty-printed summaries separate from logging (use `print()` for user-facing output)

- **Small examples (copy/paste ready)**
  - Call strategy correlation function (used in backtest replacement logic):

    price_data = lambda s, a, b: backtest_execution_engine.get_prices_range(s, a, b)
    corr = TLHStrategy.calculate_correlations('AAPL', ['MSFT','GOOGL'], price_data, 90, current_date=some_date)

  - Initialize backtest and run an initial portfolio:

    from backtesting.backtest_engine import TLHBacktester, BacktestParams
    bt = TLHBacktester(BacktestParams)
    bt.initialize_portfolio(start_date, symbols)

  - Download and update metadata with error handling:

    from backtesting.metadata_downloader import MetadataDownloader
    
    downloader = MetadataDownloader(
        index_symbol='SPY',
        ignore_symbols={'BRK.B', 'BF.B'},  # Known problematic symbols
        output_dir='backtest_data'
    )
    stats = downloader.update_metadata(rate_limit_delay=2.0)
    # Returns: { 'constituents_count', 'to_delete_count', 'to_add_count', 'failed_count', 'metadata_count' }

- **When in doubt**:
  - Check `strategy/config.py` for global constants (tax rates, thresholds, commission, file paths).
  - Prefer editing `TLHStrategy` for algorithm changes, and `ExecutionEngine` implementations for environment-specific behavior.
  - For data pipeline issues, check that all 3 steps completed: constituents → metadata → prices.
  - If metadata download fails, add problematic symbols to `--ignore-symbols` (error message shows exact list).

- **Test suite overview**:
  - `tests/test_index_downloader.py`: 6 tests for constituent downloading (file validation, error handling, CLI)
  - `tests/test_metadata_downloader.py`: 43 tests (42 unit + 1 integration) covering all phases from file loading to CLI
  - Run unit tests only: `pytest -m "not integration"` (default, fast)
  - Run with integration: `pytest -m integration` (slow, requires network, calls real yfinance API)

Please review this draft for missing integration details (credentials, exact run flags) or any local scripts you use for CI/test runs; I can merge updates and iterate.
