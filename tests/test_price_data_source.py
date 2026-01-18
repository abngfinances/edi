"""
Tests for Price Data Source

Comprehensive test suite for price data source abstraction and yfinance implementation.
Tests follow TDD phases: construction, single symbol, batch parallel, error handling.
"""

import json
import logging
import pytest
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from concurrent.futures import ThreadPoolExecutor
from backtesting.price_data_source import PriceDataSource, YFinanceSource, PriceDownloader

# Configure logging for tests
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: Construction Tests
# ============================================================================

class TestYFinanceSourceConstruction:
    """Test YFinanceSource initialization and configuration"""
    
    def test_initialization_defaults(self):
        """Should initialize with default parameters"""
        source = YFinanceSource()
        
        assert source.interval == '1d'
        assert source.auto_adjust is True
        assert source.prepost is False
        assert source.max_workers == 5
    
    def test_initialization_custom_params(self):
        """Should initialize with custom parameters"""
        source = YFinanceSource(
            interval='1wk',
            auto_adjust=False,
            prepost=True,
            max_workers=10
        )
        
        assert source.interval == '1wk'
        assert source.auto_adjust is False
        assert source.prepost is True
        assert source.max_workers == 10


# ============================================================================
# PHASE 2: Single Symbol Download Tests
# ============================================================================

class TestDownloadSymbol:
    """Test single symbol download functionality"""
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_success(self, mock_ticker_class):
        """Should successfully download price data for a symbol"""
        # Setup mock
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2020-01-01', periods=3, freq='D'))
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        source = YFinanceSource()
        result = source.download_symbol('AAPL', '2020-01-01', '2020-01-03')
        
        # Verify
        assert 'prices' in result
        assert 'splits' in result
        assert 'dividends' in result
        assert len(result['prices']) == 3
        assert list(result['prices'].columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Verify splits and dividends are empty
        assert len(result['splits']) == 0
        assert len(result['dividends']) == 0
        
        # Verify yfinance was called correctly
        mock_ticker_class.assert_called_once_with('AAPL')
        mock_ticker.history.assert_called_once_with(
            start='2020-01-01',
            end='2020-01-04',  # Adjusted +1 day for inclusive behavior
            interval='1d',
            auto_adjust=True,
            prepost=False
        )
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_includes_split_data(self, mock_ticker_class):
        """Should include split information in the result"""
        # Setup mock with split data
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        
        mock_splits = pd.Series(
            [4.0],  # 4-for-1 split
            index=pd.to_datetime(['2020-08-31'])
        )
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = mock_splits
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        source = YFinanceSource()
        result = source.download_symbol('AAPL', '2020-01-01', '2020-12-31')
        
        # Verify
        assert 'splits' in result
        assert len(result['splits']) == 1
        assert result['splits'].iloc[0] == 4.0
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_includes_dividend_data(self, mock_ticker_class):
        """Should include dividend information in the result"""
        # Setup mock with dividend data
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        
        mock_dividends = pd.Series(
            [0.77, 0.82],
            index=pd.to_datetime(['2020-02-07', '2020-05-08'])
        )
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = mock_dividends
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        source = YFinanceSource()
        result = source.download_symbol('AAPL', '2020-01-01', '2020-12-31')
        
        # Verify
        assert 'dividends' in result
        assert len(result['dividends']) == 2
        assert result['dividends'].iloc[0] == 0.77
        assert result['dividends'].iloc[1] == 0.82
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_validates_split_ratios(self, mock_ticker_class):
        """Should raise ValueError for non-whole number split ratios"""
        # Setup mock with invalid split ratio (fractional)
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        
        mock_splits = pd.Series(
            [3.5],  # Invalid: fractional split ratio
            index=pd.to_datetime(['2020-08-31'])
        )
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = mock_splits
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute and verify
        source = YFinanceSource()
        with pytest.raises(ValueError) as exc_info:
            source.download_symbol('AAPL', '2020-01-01', '2020-12-31')
        
        assert 'invalid split ratio' in str(exc_info.value).lower()
        assert '3.5' in str(exc_info.value)
        assert 'whole numbers' in str(exc_info.value).lower()
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_empty_data_raises_error(self, mock_ticker_class):
        """Should raise ValueError when no data is returned"""
        # Setup mock with empty DataFrame
        mock_ticker = Mock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker
        
        # Execute and verify
        source = YFinanceSource()
        with pytest.raises(ValueError) as exc_info:
            source.download_symbol('INVALID', '2020-01-01', '2020-01-03')
        
        assert 'no price data' in str(exc_info.value).lower()
        assert 'INVALID' in str(exc_info.value)
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_handles_any_exception(self, mock_ticker_class):
        """Should handle any exception from yfinance and wrap as ValueError"""
        # Setup mock to raise generic exception (could be any type)
        mock_ticker_class.side_effect = Exception("Something went wrong")
        
        # Execute and verify
        source = YFinanceSource()
        with pytest.raises(ValueError) as exc_info:
            source.download_symbol('AAPL', '2020-01-01', '2020-01-03')
        
        # Verify error message provides context
        error_msg = str(exc_info.value)
        assert 'failed to download' in error_msg.lower()
        assert 'AAPL' in error_msg
        assert 'something went wrong' in error_msg.lower()  # Original error included
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_returns_standard_format(self, mock_ticker_class):
        """Should return data in standardized format with required columns"""
        # Setup mock
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [104.0],
            'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        source = YFinanceSource()
        result = source.download_symbol('AAPL', '2020-01-01', '2020-01-01')
        
        # Verify standardized format
        assert isinstance(result, dict)
        assert 'prices' in result
        assert 'splits' in result
        assert 'dividends' in result
        assert isinstance(result['prices'], pd.DataFrame)
        assert isinstance(result['prices'].index, pd.DatetimeIndex)
        
        # Verify OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in result['prices'].columns
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_symbol_single_day_inclusive(self, mock_ticker_class):
        """Should download single day when start_date == end_date (inclusive behavior)"""
        # Setup mock
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [104.0],
            'Volume': [1000000]
        }, index=pd.date_range('2020-01-02', periods=1, freq='D'))
        
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute with start == end
        source = YFinanceSource()
        result = source.download_symbol('AAPL', '2020-01-02', '2020-01-02')
        
        # Verify yfinance was called with adjusted end_date (+1 day)
        mock_ticker.history.assert_called_once_with(
            start='2020-01-02',
            end='2020-01-03',  # Should be adjusted to next day
            interval='1d',
            auto_adjust=True,
            prepost=False
        )
        
        # Verify result has data
        assert len(result['prices']) == 1
        assert result['prices']['Close'].iloc[0] == 104.0


# ============================================================================
# PHASE 3: Parallel Batch Download Tests
# ============================================================================

class TestDownloadBatchParallel:
    """Test parallel batch download functionality"""
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_parallel_all_success(self, mock_ticker_class):
        """Should download all symbols successfully in parallel"""
        # Setup mock to return different data for each symbol
        def create_mock_ticker(symbol):
            mock_ticker = Mock()
            mock_prices = pd.DataFrame({
                'Open': [100.0], 'High': [105.0], 'Low': [99.0],
                'Close': [104.0], 'Volume': [1000000]
            }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
            mock_ticker.history.return_value = mock_prices
            mock_ticker.splits = pd.Series([], dtype=float)
            mock_ticker.dividends = pd.Series([], dtype=float)
            return mock_ticker
        
        mock_ticker_class.side_effect = lambda s: create_mock_ticker(s)
        
        # Execute
        source = YFinanceSource(max_workers=3)
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        failed, results = source.download_batch_parallel(symbols, '2020-01-01', '2020-01-03', rate_limit_delay=0)
        
        # Verify
        assert len(failed) == 0
        assert len(results) == 3
        assert set(results.keys()) == {'AAPL', 'MSFT', 'GOOGL'}
        
        for symbol in symbols:
            assert 'prices' in results[symbol]
            assert 'splits' in results[symbol]
            assert 'dividends' in results[symbol]
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_parallel_partial_failures(self, mock_ticker_class):
        """Should handle partial failures and continue downloading others"""
        # Setup mock to fail for specific symbol
        def create_mock_ticker(symbol):
            if symbol == 'INVALID':
                raise Exception("Symbol not found")
            mock_ticker = Mock()
            mock_prices = pd.DataFrame({
                'Open': [100.0], 'High': [105.0], 'Low': [99.0],
                'Close': [104.0], 'Volume': [1000000]
            }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
            mock_ticker.history.return_value = mock_prices
            mock_ticker.splits = pd.Series([], dtype=float)
            mock_ticker.dividends = pd.Series([], dtype=float)
            return mock_ticker
        
        mock_ticker_class.side_effect = lambda s: create_mock_ticker(s)
        
        # Execute
        source = YFinanceSource(max_workers=3)
        symbols = ['AAPL', 'INVALID', 'MSFT']
        failed, results = source.download_batch_parallel(symbols, '2020-01-01', '2020-01-03', rate_limit_delay=0)
        
        # Verify
        assert len(failed) == 1
        assert 'INVALID' in failed
        assert len(results) == 2
        assert 'AAPL' in results
        assert 'MSFT' in results
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_returns_failed_set_and_results_dict(self, mock_ticker_class):
        """Should return tuple of (failed_set, results_dict)"""
        # Setup mock
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute
        source = YFinanceSource()
        result = source.download_batch_parallel(['AAPL'], '2020-01-01', '2020-01-03', rate_limit_delay=0)
        
        # Verify return type
        assert isinstance(result, tuple)
        assert len(result) == 2
        failed, results = result
        assert isinstance(failed, set)
        assert isinstance(results, dict)
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_respects_max_workers(self, mock_ticker_class):
        """Should respect max_workers configuration"""
        # Setup mock to return different data for each symbol
        def create_mock_ticker(symbol):
            mock_ticker = Mock()
            # Create unique price data per symbol (use symbol hash for uniqueness)
            base_price = 100.0 + len(symbol) * 10  # Different base price per symbol
            mock_prices = pd.DataFrame({
                'Open': [base_price], 'High': [base_price + 5.0], 'Low': [base_price - 1.0],
                'Close': [base_price + 4.0], 'Volume': [1000000]
            }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
            mock_ticker.history.return_value = mock_prices
            mock_ticker.splits = pd.Series([], dtype=float)
            mock_ticker.dividends = pd.Series([], dtype=float)
            return mock_ticker
        
        mock_ticker_class.side_effect = lambda s: create_mock_ticker(s)
        
        # Execute with different max_workers
        source = YFinanceSource(max_workers=2)
        assert source.max_workers == 2
        
        # Verify it completes successfully with correct data for each symbol
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'IBM', 'TSLA']
        failed, results = source.download_batch_parallel(symbols, '2020-01-01', '2020-01-03', rate_limit_delay=0)
        
        # Verify all symbols processed successfully
        assert len(failed) == 0
        assert len(results) == 5
        assert set(results.keys()) == {'AAPL', 'MSFT', 'GOOGL', 'IBM', 'TSLA'}
        
        # Verify each symbol has unique correct data
        for symbol in symbols:
            assert 'prices' in results[symbol]
            assert 'splits' in results[symbol]
            assert 'dividends' in results[symbol]
            # Verify unique prices were returned for this symbol
            expected_base = 100.0 + len(symbol) * 10
            assert results[symbol]['prices']['Open'].iloc[0] == expected_base
            assert results[symbol]['prices']['Close'].iloc[0] == expected_base + 4.0
    
    @patch('backtesting.price_data_source.time.sleep')
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_rate_limiting_between_batches(self, mock_ticker_class, mock_sleep):
        """Should apply rate limiting after batch completes"""
        # Setup mock
        mock_ticker = Mock()
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        mock_ticker.history.return_value = mock_prices
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute with rate limiting
        source = YFinanceSource()
        source.download_batch_parallel(['AAPL'], '2020-01-01', '2020-01-03', rate_limit_delay=2.5)
        
        # Verify rate limiting was applied
        mock_sleep.assert_called_once_with(2.5)
    
    @patch('backtesting.price_data_source.yf.Ticker')
    def test_download_batch_retries_on_429_error(self, mock_ticker_class):
        """Should retry downloads on 429 rate limit errors with exponential backoff"""
        # Setup mock to fail first, then succeed
        mock_ticker = Mock()
        call_count = [0]
        
        def history_side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("429 Rate limit exceeded")
            # Second call succeeds
            return pd.DataFrame({
                'Open': [100.0], 'High': [105.0], 'Low': [99.0],
                'Close': [104.0], 'Volume': [1000000]
            }, index=pd.date_range('2020-01-01', periods=1, freq='D'))
        
        mock_ticker.history.side_effect = history_side_effect
        mock_ticker.splits = pd.Series([], dtype=float)
        mock_ticker.dividends = pd.Series([], dtype=float)
        mock_ticker_class.return_value = mock_ticker
        
        # Execute with mocked sleep to avoid waiting
        with patch('backtesting.price_data_source.time.sleep'):
            source = YFinanceSource()
            failed, results = source.download_batch_parallel(['AAPL'], '2020-01-01', '2020-01-03', rate_limit_delay=0)
        
        # Verify retry succeeded
        assert len(failed) == 0
        assert 'AAPL' in results
        assert call_count[0] == 2  # First failed, second succeeded


# ============================================================================
# PHASE 2: PriceDownloader Initialization Tests
# ============================================================================

class TestPriceDownloaderInitialization:
    """Test PriceDownloader initialization with validation"""
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_with_valid_params(self, mock_file, mock_exists):
        """Should initialize successfully with valid parameters"""
        # Setup mock for constituents file
        mock_exists.return_value = True
        constituents_data = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],
            'metadata': {'index_symbol': 'SPY'}
        }
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='test_data/metadata',
            output_dir='test_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            interval='1d',
            source='yfinance'
        )
        
        # Verify
        assert downloader.index_symbol == 'SPY'
        assert downloader.start_date == '2020-01-01'
        assert downloader.end_date == '2020-12-31'
        assert downloader.interval == '1d'
        assert downloader.source == 'yfinance'
        assert downloader.output_dir == Path('test_data')
        assert downloader.metadata_dir == Path('test_data/metadata')
        assert len(downloader.constituents) == 3
        assert downloader.constituents == ['AAPL', 'MSFT', 'GOOGL']
        assert isinstance(downloader.data_source, YFinanceSource)
    
    def test_initialization_validates_date_format(self):
        """Should raise ValueError for invalid date format"""
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='01-01-2020',  # Invalid format
                end_date='2020-12-31'
            )
        
        assert 'YYYY-MM-DD' in str(exc_info.value)
        assert 'start_date' in str(exc_info.value)
    
    def test_initialization_validates_date_order(self):
        """Should raise ValueError if start_date > end_date"""
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='2020-12-31',
                end_date='2020-01-01'  # Before start_date
            )
        
        assert 'start_date must be <=' in str(exc_info.value)
    
    @patch('backtesting.price_data_source.datetime')
    def test_initialization_validates_future_dates(self, mock_datetime):
        """Should raise ValueError if end_date is today or in the future"""
        # Mock current date to 2020-06-01 (using a Mock that acts like datetime)
        mock_now_result = Mock()
        mock_now_result.replace.return_value = datetime(2020, 6, 1)
        mock_datetime.now.return_value = mock_now_result
        mock_datetime.strptime = datetime.strptime  # Keep real strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='2020-01-01',
                end_date='2020-06-01'  # Today's date (mocked)
            )
        
        assert 'must be before today' in str(exc_info.value)
        assert 'partial trading day' in str(exc_info.value)
    
    def test_initialization_validates_interval(self):
        """Should raise ValueError for invalid interval"""
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='2020-01-01',
                end_date='2020-12-31',
                interval='5m'  # Invalid for this use case
            )
        
        assert 'Invalid interval' in str(exc_info.value)
        assert '5m' in str(exc_info.value)
    
    def test_initialization_validates_source(self):
        """Should raise ValueError for invalid source"""
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='2020-01-01',
                end_date='2020-12-31',
                source='alphavantage'  # Invalid source
            )
        
        assert 'Invalid source' in str(exc_info.value)
        assert 'alphavantage' in str(exc_info.value)
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_loads_and_filters_constituents(self, mock_file, mock_exists):
        """Should load constituents and filter ignore_symbols"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'BRK.B', 'AMZN']
        }
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute with ignore_symbols
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='test_data/metadata',
            output_dir='test_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            ignore_symbols={'BRK.B', 'AMZN'}
        )
        
        # Verify filtering worked
        assert len(downloader.constituents) == 3
        assert 'AAPL' in downloader.constituents
        assert 'MSFT' in downloader.constituents
        assert 'GOOGL' in downloader.constituents
        assert 'BRK.B' not in downloader.constituents
        assert 'AMZN' not in downloader.constituents
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_allows_equal_start_end_dates(self, mock_file, mock_exists):
        """Should allow start_date == end_date (single day download)"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute - should NOT raise ValueError
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='test_data/metadata',
            output_dir='test_data',
            start_date='2020-01-02',
            end_date='2020-01-02'  # Same as start_date
        )
        
        # Verify initialization succeeded
        assert downloader.start_date == '2020-01-02'
        assert downloader.end_date == '2020-01-02'
    
    @patch('backtesting.price_data_source.datetime')
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_initialization_rejects_todays_date(self, mock_file, mock_exists, mock_datetime):
        """Should reject today's date to avoid partial trading day data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Mock current date to 2020-06-15
        mock_now_result = Mock()
        mock_now_result.replace.return_value = datetime(2020, 6, 15)
        mock_datetime.now.return_value = mock_now_result
        mock_datetime.strptime = datetime.strptime
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Try to use today's date as end_date
        with pytest.raises(ValueError) as exc_info:
            PriceDownloader(
                index_symbol='SPY',
                metadata_dir='test_data/metadata',
                output_dir='test_data',
                start_date='2020-06-01',
                end_date='2020-06-15'  # Today's date
            )
        
        assert 'must be before today' in str(exc_info.value)
        assert 'partial trading day' in str(exc_info.value)
        assert '2020-06-14' in str(exc_info.value)  # Should suggest yesterday


# ============================================================================
# PHASE 3: Metadata & Folder Structure Tests
# ============================================================================

class TestMetadataAndFolderStructure:
    """Test metadata file operations and folder structure"""
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_symbol_folder_path(self, mock_file, mock_exists):
        """Should construct correct folder path for symbol"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Verify folder structure
        folder = downloader._get_symbol_folder('AAPL')
        assert folder == Path('backtest_data/AAPL/yfinance_1d')
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_metadata_path(self, mock_file, mock_exists):
        """Should construct correct metadata file path"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['MSFT']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Verify metadata path
        metadata_path = downloader._get_metadata_path('MSFT')
        assert metadata_path == Path('backtest_data/MSFT/yfinance_1d/metadata.json')
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_get_prices_path(self, mock_file, mock_exists):
        """Should construct correct prices file path"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['GOOGL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Verify prices path
        prices_path = downloader._get_prices_path('GOOGL')
        assert prices_path == Path('backtest_data/GOOGL/yfinance_1d/prices.parquet')
    
    @patch('builtins.open', new_callable=mock_open)
    def test_read_metadata_file_not_exists(self, mock_file):
        """Should return None when metadata file doesn't exist"""
        # Setup mock for constituents
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Patch exists after initialization to simulate metadata file not existing
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            metadata = downloader._read_metadata('AAPL')
            assert metadata is None
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_metadata_success(self, mock_file, mock_exists):
        """Should successfully read valid metadata file"""
        # Setup mock
        mock_exists.return_value = True
        
        # Mock will be called twice: once for constituents, once for metadata
        constituents_data = {'symbols': ['AAPL']}
        metadata_data = {
            'symbol': 'AAPL',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {'2020-08-31': 4.0},
            'dividends': {'2020-02-07': 0.77},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(metadata_data)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Verify metadata is read correctly
        metadata = downloader._read_metadata('AAPL')
        assert metadata is not None
        assert metadata['symbol'] == 'AAPL'
        assert metadata['interval'] == '1d'
        assert metadata['source'] == 'yfinance'
        assert metadata['start_date'] == '2020-01-01'
        assert metadata['end_date'] == '2020-12-31'
        assert metadata['total_days'] == 252
        assert metadata['splits'] == {'2020-08-31': 4.0}
        assert metadata['dividends'] == {'2020-02-07': 0.77}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_read_metadata_missing_required_fields(self, mock_file, mock_exists):
        """Should raise ValueError if metadata missing required fields"""
        # Setup mock
        mock_exists.return_value = True
        
        constituents_data = {'symbols': ['AAPL']}
        incomplete_metadata = {
            'start_date': '2020-01-01',
            # Missing: symbol, interval, source, end_date, total_days, splits, dividends, last_updated
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(incomplete_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Verify raises error
        with pytest.raises(ValueError) as exc_info:
            downloader._read_metadata('AAPL')
        
        assert 'missing required fields' in str(exc_info.value).lower()
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('backtesting.price_data_source.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_metadata_includes_all_required_fields(self, mock_file, mock_mkdir, mock_exists):
        """Should write metadata with all required fields validated"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            interval='1d',
            source='yfinance'
        )
        
        # Capture what gets written to the file
        written_content = []
        def write_side_effect(content):
            written_content.append(content)
            return len(content)
        
        mock_file.return_value.write.side_effect = write_side_effect
        
        # Write metadata
        downloader._write_metadata(
            symbol='AAPL',
            start_date='2020-01-01',
            end_date='2020-12-31',
            total_days=252,
            splits={'2020-08-31': 4.0},
            dividends={'2020-02-07': 0.77}
        )
        
        # Verify mkdir was called
        mock_mkdir.assert_called()
        
        # Parse the written JSON
        written_json = ''.join(written_content)
        metadata = json.loads(written_json)
        
        # Verify all required fields are present
        required_fields = {
            'symbol', 'interval', 'source',
            'start_date', 'end_date', 'total_days',
            'splits', 'dividends', 'last_updated'
        }
        assert set(metadata.keys()) == required_fields, f"Field mismatch. Got: {set(metadata.keys())}"
        
        # Verify field values
        assert metadata['symbol'] == 'AAPL'
        assert metadata['interval'] == '1d'
        assert metadata['source'] == 'yfinance'
        assert metadata['start_date'] == '2020-01-01'
        assert metadata['end_date'] == '2020-12-31'
        assert metadata['total_days'] == 252
        assert metadata['splits'] == {'2020-08-31': 4.0}
        assert metadata['dividends'] == {'2020-02-07': 0.77}
        assert metadata['last_updated'].endswith('Z')  # ISO format with UTC
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('backtesting.price_data_source.Path.mkdir')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_metadata_captures_instance_interval(self, mock_file, mock_mkdir, mock_exists):
        """Should capture interval and source from downloader instance"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['MSFT']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute with non-default interval
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            interval='1wk',  # Non-default
            source='yfinance'
        )
        
        # Capture writes
        written_content = []
        mock_file.return_value.write.side_effect = lambda c: written_content.append(c) or len(c)
        
        # Write metadata
        downloader._write_metadata(
            symbol='MSFT',
            start_date='2020-01-01',
            end_date='2020-12-31',
            total_days=52,
            splits={},
            dividends={'2020-05-20': 0.51}
        )
        
        # Parse written JSON
        written_json = ''.join(written_content)
        metadata = json.loads(written_json)
        
        # Verify interval/source from downloader instance
        assert metadata['interval'] == '1wk'
        assert metadata['source'] == 'yfinance'
        assert metadata['symbol'] == 'MSFT'


# ============================================================================
# PHASE 4: Date Range Planning Tests
# ============================================================================

class TestDateRangePlanning:
    """Test date range planning and merge logic"""
    
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_no_existing_metadata(self, mock_file):
        """Should return full range when no existing metadata"""
        # Setup mock
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # No metadata file exists
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            ranges = downloader._plan_date_range('AAPL', '2020-01-01', '2020-12-31')
        
        # Should return full range
        assert ranges == [('2020-01-01', '2020-12-31')]
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_subset_skip(self, mock_file, mock_exists):
        """Should return None when requested range is subset of existing (skip download)"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        existing_metadata = {
            'symbol': 'AAPL',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-06-01',  # Subset of existing
            end_date='2020-09-30'
        )
        
        # Request subset of existing range
        ranges = downloader._plan_date_range('AAPL', '2020-06-01', '2020-09-30')
        
        # Should skip (return None)
        assert ranges is None
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_extend_after(self, mock_file, mock_exists):
        """Should download from existing_end+1 to new_end when extending after"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['MSFT']}
        existing_metadata = {
            'symbol': 'MSFT',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-06-30',  # Ends mid-year
            'total_days': 126,
            'splits': {},
            'dividends': {},
            'last_updated': '2020-07-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'  # Extend to end of year
        )
        
        # Request range that extends beyond existing end
        ranges = downloader._plan_date_range('MSFT', '2020-01-01', '2020-12-31')
        
        # Should download only the new part: 2020-07-01 to 2020-12-31
        assert ranges == [('2020-07-01', '2020-12-31')]
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_extend_before(self, mock_file, mock_exists):
        """Should download from new_start to existing_start-1 when extending before"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['GOOGL']}
        existing_metadata = {
            'symbol': 'GOOGL',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-07-01',  # Starts mid-year
            'end_date': '2020-12-31',
            'total_days': 126,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',  # Extend to start of year
            end_date='2020-12-31'
        )
        
        # Request range that extends before existing start
        ranges = downloader._plan_date_range('GOOGL', '2020-01-01', '2020-12-31')
        
        # Should download only the new part: 2020-01-01 to 2020-06-30
        assert ranges == [('2020-01-01', '2020-06-30')]
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_extend_both_sides(self, mock_file, mock_exists):
        """Should download two ranges when extending both before and after existing data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AMZN']}
        existing_metadata = {
            'symbol': 'AMZN',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-06-01',  # Mid-year data only
            'end_date': '2020-08-31',
            'total_days': 92,
            'splits': {},
            'dividends': {},
            'last_updated': '2020-09-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',  # Extends before existing
            end_date='2020-12-31'      # AND after existing
        )
        
        # Request range that extends both before and after
        ranges = downloader._plan_date_range('AMZN', '2020-01-01', '2020-12-31')
        
        # Should download two ranges: before and after existing
        assert ranges == [('2020-01-01', '2020-05-31'), ('2020-09-01', '2020-12-31')]
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_gap_after(self, mock_file, mock_exists):
        """Should raise ValueError when gap exists after existing data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['TSLA']}
        existing_metadata = {
            'symbol': 'TSLA',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2019-01-01',
            'end_date': '2019-12-31',  # 2019 data
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2020-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2021-01-01',  # Gap: requesting 2021 when we have 2019
            end_date='2021-12-31'
        )
        
        # Request range with gap - should raise error
        with pytest.raises(ValueError) as exc_info:
            downloader._plan_date_range('TSLA', '2021-01-01', '2021-12-31')
        
        # Verify error message is helpful
        assert 'gap' in str(exc_info.value).lower()
        assert '2019-01-01' in str(exc_info.value)
        assert '2019-12-31' in str(exc_info.value)
        assert '2021-01-01' in str(exc_info.value)
        assert '2021-12-31' in str(exc_info.value)
        assert 'contiguous' in str(exc_info.value).lower()
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_gap_before(self, mock_file, mock_exists):
        """Should raise ValueError when gap exists before existing data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['NFLX']}
        existing_metadata = {
            'symbol': 'NFLX',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2021-01-01',  # 2021 data
            'end_date': '2021-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2022-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2019-01-01',  # Gap: requesting 2019 when we have 2021
            end_date='2019-12-31'
        )
        
        # Request range with gap - should raise error
        with pytest.raises(ValueError) as exc_info:
            downloader._plan_date_range('NFLX', '2019-01-01', '2019-12-31')
        
        # Verify error message is helpful
        assert 'gap' in str(exc_info.value).lower()
        assert '2021-01-01' in str(exc_info.value)
        assert '2021-12-31' in str(exc_info.value)
        assert '2019-01-01' in str(exc_info.value)
        assert '2019-12-31' in str(exc_info.value)
        assert 'contiguous' in str(exc_info.value).lower()
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_different_interval_separate_folders(self, mock_file, mock_exists):
        """Should download full range when interval differs (different folders)"""
        # Setup mock - existing 1wk data
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        existing_metadata = {
            'symbol': 'AAPL',
            'interval': '1wk',  # Existing is weekly
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 52,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute with different interval (1d instead of 1wk)
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            interval='1d'  # Requesting daily
        )
        
        # Simulate no metadata in the 1d folder (only 1wk folder has data)
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            ranges = downloader._plan_date_range('AAPL', '2020-01-01', '2020-12-31')
        
        # Should download full range (new folder: yfinance_1d separate from yfinance_1wk)
        assert ranges == [('2020-01-01', '2020-12-31')]
        
        # Verify folder paths are different
        folder_1d = downloader._get_symbol_folder('AAPL')
        assert str(folder_1d).endswith('yfinance_1d')
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_different_source_separate_folders(self, mock_file, mock_exists):
        """Should download full range when source differs (different folders)"""
        # Setup mock - existing alphavantage data
        mock_exists.return_value = True
        constituents_data = {'symbols': ['MSFT']}
        existing_metadata = {
            'symbol': 'MSFT',
            'interval': '1d',
            'source': 'alphavantage',  # Existing is alphavantage
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute with different source (yfinance instead of alphavantage)
        # Note: This will fail validation since 'alphavantage' is not in VALID_SOURCES,
        # but that's a separate concern. For this test, assume it was valid.
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31',
            source='yfinance'  # Requesting yfinance
        )
        
        # Simulate no metadata in the yfinance folder (only alphavantage folder has data)
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            ranges = downloader._plan_date_range('MSFT', '2020-01-01', '2020-12-31')
        
        # Should download full range (new folder: yfinance_1d separate from alphavantage_1d)
        assert ranges == [('2020-01-01', '2020-12-31')]
        
        # Verify folder paths are different
        folder_yfinance = downloader._get_symbol_folder('MSFT')
        assert str(folder_yfinance).endswith('yfinance_1d')
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_plan_date_range_exact_match_skip(self, mock_file, mock_exists):
        """Should return None when requested range exactly matches existing (edge case)"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AMZN']}
        existing_metadata = {
            'symbol': 'AMZN',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Execute
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Request exact same range
        ranges = downloader._plan_date_range('AMZN', '2020-01-01', '2020-12-31')
        
        # Should skip (subset case includes exact match)
        assert ranges is None


# ============================================================================
# PHASE 5: Download & Merge Single Symbol Tests
# ============================================================================

class TestDownloadSymbolPrices:
    """Test downloading and merging price data for a single symbol"""
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_no_existing_data(self, mock_file, mock_exists):
        """Should download full range and write new files when no existing data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Create downloader
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-01-10'
        )
        
        # Mock data source to return price data
        mock_prices = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2020-01-02', periods=3, freq='D'))
        
        downloader.data_source.download_symbol = Mock(return_value={
            'prices': mock_prices,
            'splits': pd.Series([], dtype=float),
            'dividends': pd.Series([], dtype=float)
        })
        
        # Mock file operations
        with patch('backtesting.price_data_source.Path.mkdir'), \
             patch('backtesting.price_data_source.Path.exists', return_value=False), \
             patch.object(downloader, '_write_metadata') as mock_write_metadata, \
             patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            
            # Execute
            result = downloader.download_symbol_prices('AAPL')
        
        # Verify download was called
        downloader.data_source.download_symbol.assert_called_once_with('AAPL', '2020-01-01', '2020-01-10')
        
        # Verify metadata was written
        mock_write_metadata.assert_called_once()
        call_args = mock_write_metadata.call_args[1]
        assert call_args['symbol'] == 'AAPL'
        assert call_args['start_date'] == '2020-01-02'  # First price date
        assert call_args['end_date'] == '2020-01-04'    # Last price date
        assert call_args['total_days'] == 3
        
        # Verify prices were written
        mock_to_parquet.assert_called_once()
        
        # Verify result
        assert result == {'symbol': 'AAPL', 'downloaded': True, 'skipped': False}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_skip_when_exists(self, mock_file, mock_exists):
        """Should skip download when data already exists (subset case)"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['MSFT']}
        existing_metadata = {
            'symbol': 'MSFT',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Create downloader requesting subset of existing range
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-06-01',  # Subset
            end_date='2020-09-30'
        )
        
        # Execute
        result = downloader.download_symbol_prices('MSFT')
        
        # Verify no download occurred
        assert result == {'symbol': 'MSFT', 'downloaded': False, 'skipped': True}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_parquet')
    def test_download_symbol_prices_extend_after(self, mock_read_parquet, mock_file, mock_exists):
        """Should download only new dates and merge with existing data"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['GOOGL']}
        existing_metadata = {
            'symbol': 'GOOGL',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-02',
            'end_date': '2020-01-10',
            'total_days': 7,
            'splits': {},
            'dividends': {'2020-01-05': 0.50},
            'last_updated': '2020-01-11T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata),  # For _plan_date_range
            json.dumps(existing_metadata)   # For download_symbol_prices (loading existing metadata)
        ]
        
        # Mock existing price data
        existing_prices = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [104.0, 105.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-01-02', periods=2, freq='D'))
        mock_read_parquet.return_value = existing_prices
        
        # Create downloader extending after existing range
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-02',  # Same as existing start (not before)
            end_date='2020-01-15'      # Extends after
        )
        
        # Mock new price data (2020-01-11 to 2020-01-15)
        new_prices = pd.DataFrame({
            'Open': [106.0, 107.0],
            'High': [108.0, 109.0],
            'Low': [105.0, 106.0],
            'Close': [107.0, 108.0],
            'Volume': [1200000, 1300000]
        }, index=pd.date_range('2020-01-13', periods=2, freq='D'))
        
        downloader.data_source.download_symbol = Mock(return_value={
            'prices': new_prices,
            'splits': pd.Series([], dtype=float),
            'dividends': pd.Series([0.55], index=pd.to_datetime(['2020-01-14']))
        })
        
        # Mock file operations
        with patch('backtesting.price_data_source.Path.mkdir'), \
             patch.object(downloader, '_write_metadata') as mock_write_metadata, \
             patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            
            # Execute
            result = downloader.download_symbol_prices('GOOGL')
        
        # Verify download called with only new range
        downloader.data_source.download_symbol.assert_called_once_with('GOOGL', '2020-01-11', '2020-01-15')
        
        # Verify metadata updated with extended range
        mock_write_metadata.assert_called_once()
        call_args = mock_write_metadata.call_args[1]
        assert call_args['symbol'] == 'GOOGL'
        assert call_args['start_date'] == '2020-01-02'  # Unchanged from existing
        assert call_args['end_date'] == '2020-01-14'    # Extended to last new price
        assert call_args['total_days'] == 4  # 2 existing + 2 new
        # Verify dividends merged
        assert call_args['dividends'] == {'2020-01-05': 0.50, '2020-01-14': 0.55}
        
        # Verify merged prices written
        mock_to_parquet.assert_called_once()
        # to_parquet is called as df.to_parquet(path), so mock is on the DataFrame method
        # The path is the first argument
        written_path = mock_to_parquet.call_args[0][0]
        assert 'prices.parquet' in str(written_path)
        
        # Verify result
        assert result == {'symbol': 'GOOGL', 'downloaded': True, 'skipped': False}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_parquet')
    def test_download_symbol_prices_extend_both_sides(self, mock_read_parquet, mock_file, mock_exists):
        """Should download two ranges and merge when extending both before and after"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AMZN']}
        existing_metadata = {
            'symbol': 'AMZN',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-06-01',
            'end_date': '2020-08-31',
            'total_days': 92,
            'splits': {},
            'dividends': {},
            'last_updated': '2020-09-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata),  # For _plan_date_range
            json.dumps(existing_metadata)   # For download_symbol_prices (loading existing metadata)
        ]
        
        # Mock existing price data
        existing_prices = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0],
            'Low': [99.0, 100.0],
            'Close': [104.0, 105.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2020-06-01', periods=2, freq='D'))
        mock_read_parquet.return_value = existing_prices
        
        # Create downloader extending both sides
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Mock data source to return data for both ranges
        call_count = [0]
        def download_side_effect(symbol, start, end):
            call_count[0] += 1
            if call_count[0] == 1:  # Before range
                return {
                    'prices': pd.DataFrame({
                        'Open': [90.0], 'High': [95.0], 'Low': [89.0],
                        'Close': [94.0], 'Volume': [900000]
                    }, index=pd.date_range('2020-05-29', periods=1, freq='D')),
                    'splits': pd.Series([], dtype=float),
                    'dividends': pd.Series([], dtype=float)
                }
            else:  # After range
                return {
                    'prices': pd.DataFrame({
                        'Open': [110.0], 'High': [115.0], 'Low': [109.0],
                        'Close': [114.0], 'Volume': [1300000]
                    }, index=pd.date_range('2020-09-01', periods=1, freq='D')),
                    'splits': pd.Series([], dtype=float),
                    'dividends': pd.Series([], dtype=float)
                }
        
        downloader.data_source.download_symbol = Mock(side_effect=download_side_effect)
        
        # Mock file operations
        with patch('backtesting.price_data_source.Path.mkdir'), \
             patch.object(downloader, '_write_metadata') as mock_write_metadata, \
             patch('pandas.DataFrame.to_parquet') as mock_to_parquet:
            
            # Execute
            result = downloader.download_symbol_prices('AMZN')
        
        # Verify two downloads occurred
        assert downloader.data_source.download_symbol.call_count == 2
        
        # Verify metadata updated with full range
        mock_write_metadata.assert_called_once()
        call_args = mock_write_metadata.call_args[1]
        assert call_args['start_date'] == '2020-05-29'
        assert call_args['end_date'] == '2020-09-01'
        assert call_args['total_days'] == 4  # 1 before + 2 existing + 1 after
        
        # Verify result
        assert result == {'symbol': 'AMZN', 'downloaded': True, 'skipped': False}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_handles_download_failure(self, mock_file, mock_exists):
        """Should raise ValueError when download fails"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['INVALID']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Create downloader
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-01-10'
        )
        
        # Mock data source to raise error
        downloader.data_source.download_symbol = Mock(side_effect=ValueError("Symbol not found"))
        
        # Execute and verify error
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            with pytest.raises(ValueError) as exc_info:
                downloader.download_symbol_prices('INVALID')
        
        assert 'symbol not found' in str(exc_info.value).lower()
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.read_parquet')
    def test_download_symbol_prices_merges_splits_data(self, mock_read_parquet, mock_file, mock_exists):
        """Should correctly merge splits data from existing and new downloads"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['TSLA']}
        existing_metadata = {
            'symbol': 'TSLA',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-06-30',
            'total_days': 126,
            'splits': {'2020-03-15': 2.0},  # Existing split
            'dividends': {},
            'last_updated': '2020-07-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata),  # For _plan_date_range
            json.dumps(existing_metadata)   # For download_symbol_prices (loading existing metadata)
        ]
        
        # Mock existing prices
        existing_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-02', periods=1, freq='D'))
        mock_read_parquet.return_value = existing_prices
        
        # Create downloader
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Mock new data with new split
        new_prices = pd.DataFrame({
            'Open': [200.0], 'High': [205.0], 'Low': [199.0],
            'Close': [204.0], 'Volume': [2000000]
        }, index=pd.date_range('2020-07-01', periods=1, freq='D'))
        
        downloader.data_source.download_symbol = Mock(return_value={
            'prices': new_prices,
            'splits': pd.Series([4.0], index=pd.to_datetime(['2020-08-31'])),  # New split
            'dividends': pd.Series([], dtype=float)
        })
        
        # Mock file operations
        with patch('backtesting.price_data_source.Path.mkdir'), \
             patch.object(downloader, '_write_metadata') as mock_write_metadata, \
             patch('pandas.DataFrame.to_parquet'):
            
            # Execute
            downloader.download_symbol_prices('TSLA')
        
        # Verify splits were merged correctly
        call_args = mock_write_metadata.call_args[1]
        assert call_args['splits'] == {'2020-03-15': 2.0, '2020-08-31': 4.0}
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_validates_symbol_in_constituents(self, mock_file, mock_exists):
        """Should raise ValueError if symbol not in constituents"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL', 'MSFT']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Create downloader
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-12-31'
        )
        
        # Execute with symbol not in constituents
        with pytest.raises(ValueError) as exc_info:
            downloader.download_symbol_prices('GOOGL')  # Not in constituents
        
        assert 'not in constituents' in str(exc_info.value).lower()
        assert 'GOOGL' in str(exc_info.value)
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_empty_dataframe_raises_error(self, mock_file, mock_exists):
        """Should raise ValueError if download returns empty DataFrame"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        mock_file.return_value.read.return_value = json.dumps(constituents_data)
        
        # Create downloader
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-01-01',
            end_date='2020-01-10'
        )
        
        # Mock data source to return empty DataFrame
        downloader.data_source.download_symbol = Mock(return_value={
            'prices': pd.DataFrame(),  # Empty
            'splits': pd.Series([], dtype=float),
            'dividends': pd.Series([], dtype=float)
        })
        
        # Execute and verify error
        with patch('backtesting.price_data_source.Path.exists', return_value=False):
            with pytest.raises(ValueError) as exc_info:
                downloader.download_symbol_prices('AAPL')
        
        assert 'no price data' in str(exc_info.value).lower()
        assert 'AAPL' in str(exc_info.value)
    
    @patch('backtesting.price_data_source.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_download_symbol_prices_returns_dict_with_status(self, mock_file, mock_exists):
        """Should return dict with symbol, downloaded, and skipped status"""
        # Setup mock
        mock_exists.return_value = True
        constituents_data = {'symbols': ['AAPL']}
        existing_metadata = {
            'symbol': 'AAPL',
            'interval': '1d',
            'source': 'yfinance',
            'start_date': '2020-01-01',
            'end_date': '2020-12-31',
            'total_days': 252,
            'splits': {},
            'dividends': {},
            'last_updated': '2021-01-01T00:00:00Z'
        }
        
        mock_file.return_value.read.side_effect = [
            json.dumps(constituents_data),
            json.dumps(existing_metadata)
        ]
        
        # Create downloader (subset case - will skip)
        downloader = PriceDownloader(
            index_symbol='SPY',
            metadata_dir='backtest_data/metadata',
            output_dir='backtest_data',
            start_date='2020-06-01',
            end_date='2020-09-30'
        )
        
        # Execute
        result = downloader.download_symbol_prices('AAPL')
        
        # Verify return format
        assert isinstance(result, dict)
        assert 'symbol' in result
        assert 'downloaded' in result
        assert 'skipped' in result
        assert result['symbol'] == 'AAPL'
        assert result['downloaded'] is False
        assert result['skipped'] is True

