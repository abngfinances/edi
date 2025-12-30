# EDI - Tax Loss Harvesting System

A comprehensive Tax Loss Harvesting (TLH) framework with backtesting, paper trading, and live trading capabilities.

## Project Structure

The repository implements a Tax Loss Harvesting framework with three clear layers:

- **Strategy core**: Pure business logic in `strategy/core.py` (class `TLHStrategy`)
- **Execution engines**: Pluggable `ExecutionEngine` interface with concrete implementations for backtesting, paper trading, and live trading
- **Data management**: Tools for downloading and managing market data

## Components

### Index Downloader

Downloads index/ETF constituent lists from Alpha Vantage ETF_PROFILE API.

**Location**: `backtesting/index_downloader.py`

#### Usage

```bash
# Download index constituents
python3 backtesting/index_downloader.py <SYMBOL> <API_KEY> <EXPECTED_COUNT> [--output-dir DIR]

# Example: Download QQQ (Nasdaq-100)
python3 backtesting/index_downloader.py QQQ DUJRZATXXEQL2M9R 102

# Example: Download SPY (S&P 500)
python3 backtesting/index_downloader.py SPY DUJRZATXXEQL2M9R 503 --output-dir backtest_data/metadata
```

**Arguments:**
- `SYMBOL`: Index/ETF symbol (e.g., SPY, QQQ)
- `API_KEY`: Alpha Vantage API key
- `EXPECTED_COUNT`: Expected number of constituents (required for validation)
- `--output-dir`: Output directory (default: `backtest_data/metadata`)

**Output Format:**

Files are saved as `{symbol}_constituents.json`:

```json
{
  "symbols": ["NVDA", "AAPL", "MSFT", ...],
  "metadata": {
    "index_symbol": "SPY",
    "download_timestamp": "2025-12-30T14:13:39.809678Z",
    "total_holdings": 503,
    "data_source": "Alpha Vantage ETF_PROFILE"
  }
}
```

## Testing

### Running Tests

```bash
# During development - fast unit tests only (with mocks)
pytest tests/test_index_downloader.py -m "not integration"

# Before committing - run all tests including integration
pytest tests/test_index_downloader.py

# Run integration test with specific symbol/count
pytest tests/test_index_downloader.py -m integration

# Run tests with ERROR-only logging (fast, minimal output)
pytest tests/test_index_downloader.py --log-level ERROR

# Run tests with DEBUG logging for troubleshooting
pytest tests/test_index_downloader.py --log-level DEBUG

# Use environment variable for logging level
TEST_LOG_LEVEL=ERROR pytest tests/test_index_downloader.py
```

**Test Structure:**
- **Unit tests**: Use mocked API responses, run quickly, no external dependencies
- **Integration tests**: Make real API calls (marked with `@pytest.mark.integration`)

**Logging Levels:**
- `ERROR`: Only show errors (fastest, for CI/production)
- `INFO`: Show progress messages (default, good for development)
- `DEBUG`: Show detailed diagnostic info (for troubleshooting)

### Running Specific Tests

```bash
# Run only unit tests
pytest tests/test_index_downloader.py::TestIndexDownloader -v

# Run only integration tests
pytest tests/test_index_downloader.py::TestIndexDownloaderIntegration -v

# Run specific test
pytest tests/test_index_downloader.py::TestIndexDownloader::test_download_constituents_success -v
```

## Configuration

### Environment Variables

Set up environment variables by sourcing the setup script:

```bash
source setup_env.sh
```

This sets:
- `ALPACA_API_KEY`: Alpaca API key for trading
- `ALPACA_SECRET_KEY`: Alpaca secret key
- `ALPACA_BASE_URL`: API endpoint (paper or live)
- `POLYGON_API_KEY`: Polygon.io API key for market data
- `INDEX_SYMBOL`: Index symbol for backtesting

### Alpha Vantage API Key

Get a free Alpha Vantage API key at: https://www.alphavantage.co/support/#api-key

**Test API Key** (for development):
```
DUJRZATXXEQL2M9R
```

**Rate Limits:**
- Free tier: 5 calls/minute, 100 calls/day
- Premium tiers have higher limits

## Common Index Constituent Counts

| Symbol | Index | Typical Count |
|--------|-------|---------------|
| SPY    | S&P 500 | ~503 |
| QQQ    | Nasdaq-100 | ~102 |
| IWM    | Russell 2000 | ~2000 |
| DIA    | Dow Jones | ~30 |
| VTI    | Total Stock Market | ~4000 |

Note: Counts vary slightly as companies are added/removed from indices.

## Development Guidelines

### Code Structure

- **Strategy core** (`strategy/core.py`): Pure business logic, stateless functions
- **Execution engines** (`strategy/*_engine.py`): Environment-specific implementations
- **Data tools** (`backtesting/`): Data download and management utilities
- **Tests** (`tests/`): Comprehensive unit and integration test coverage

### Testing Best Practices

1. **Unit tests first**: Write tests with mocked dependencies during development
2. **Integration tests before commit**: Run real API tests to verify end-to-end behavior
3. **Mark expensive tests**: Use `@pytest.mark.integration` for tests that hit external APIs
4. **Keep tests focused**: Test business logic, not standard library functions

### Key Design Principles

- **Simple and modular**: Each component has a single, well-defined responsibility
- **No fallbacks**: Fail fast with clear error messages
- **Test-agnostic code**: Production code has no test dependencies
- **Comprehensive logging**: Log at INFO level throughout for observability

## Troubleshooting

### Index Downloader Issues

**"Symbol count mismatch" error:**
- Index constituents change over time - verify the expected count is current
- Some ETFs may have more/fewer holdings than the underlying index

**"Invalid API key" error:**
- Verify your Alpha Vantage API key is correct
- Check if you've exceeded rate limits (wait 1 minute and retry)

**"Note: Thank you for using Alpha Vantage..." error:**
- This is a rate limit message from Alpha Vantage
- Free tier allows 5 calls/minute
- Wait 60 seconds before retrying

## Architecture

See `DESIGN.md` for detailed architecture documentation.

### Key Integration Points

- **Strategy core**: Pure functions for harvest and replacement decisions
- **Execution engines**: Pluggable interface for backtest/paper/live execution
- **Data format**: JSON files with `symbols` array and `metadata` for freshness tracking
- **Backtest engine**: Consumes constituent lists from `backtest_data/metadata/`

## Resources

- **Alpha Vantage Documentation**: https://www.alphavantage.co/documentation/
- **Alpaca Trading API**: https://alpaca.markets/docs/
- **Polygon.io Market Data**: https://polygon.io/docs/

## License

[Add license information]
