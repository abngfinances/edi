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
