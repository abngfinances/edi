"""
Tests for metadata_downloader.py

Comprehensive test suite with mocked and integration tests.
Tests follow TDD approach with incremental implementation.
"""

import json
import logging
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from backtesting.metadata_downloader import MetadataDownloader


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--log-level",
        action="store",
        default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for tests (default: ERROR)"
    )


def pytest_configure(config):
    """Configure pytest with custom options"""
    log_level = config.getoption("--log-level")
    logging.getLogger().setLevel(getattr(logging, log_level))


# ============================================================================
# PHASE 1, STEP 1.1: Constituents File Loading Tests
# ============================================================================

class TestLoadConstituents:
    """Test constituents file loading and validation"""
    
    def test_load_constituents_file_not_found(self, tmp_path):
        """Should raise ValueError with helpful message when file doesn't exist"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.load_constituents()
        
        assert 'not found' in str(exc_info.value).lower()
        assert 'spy_constituents.json' in str(exc_info.value).lower()
    
    def test_load_constituents_invalid_format_no_symbols_key(self, tmp_path):
        """Should raise ValueError when JSON missing 'symbols' key"""
        # Create invalid constituents file (missing 'symbols' key)
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({'metadata': {'index_symbol': 'SPY'}}, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.load_constituents()
        
        assert 'symbols' in str(exc_info.value).lower()
    
    def test_load_constituents_invalid_format_symbols_not_array(self, tmp_path):
        """Should raise ValueError when 'symbols' is not a list"""
        # Create invalid constituents file ('symbols' is not a list)
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({'symbols': 'AAPL,MSFT'}, f)  # String instead of list
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.load_constituents()
        
        assert 'symbols' in str(exc_info.value).lower()
        assert 'list' in str(exc_info.value).lower() or 'array' in str(exc_info.value).lower()
    
    def test_load_constituents_success(self, tmp_path):
        """Should successfully load symbols from valid constituents file"""
        # Create valid constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        expected_symbols = ['AAPL', 'MSFT', 'GOOGL']
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': expected_symbols,
                'metadata': {
                    'index_symbol': 'SPY',
                    'download_timestamp': '2025-12-30T10:00:00Z',
                    'total_holdings': 3
                }
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        symbols = downloader.load_constituents()
        
        assert symbols == expected_symbols
        assert isinstance(symbols, list)


# ============================================================================
# PHASE 1, STEP 1.2: Metadata File Loading Tests
# ============================================================================

class TestLoadMetadata:
    """Test metadata file loading and creation"""
    
    def test_load_metadata_file_not_found_creates_empty(self, tmp_path):
        """Should return empty dict when metadata file doesn't exist"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Metadata file doesn't exist yet
        assert not downloader.metadata_file.exists()
        
        metadata = downloader.load_metadata()
        
        assert metadata == {}
        assert isinstance(metadata, dict)
    
    def test_load_metadata_invalid_json(self, tmp_path):
        """Should raise ValueError when file contains invalid JSON"""
        # Create invalid JSON file
        metadata_file = tmp_path / 'spy_metadata.json'
        with open(metadata_file, 'w') as f:
            f.write('{ invalid json }')
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.load_metadata()
        
        assert 'parse' in str(exc_info.value).lower() or 'json' in str(exc_info.value).lower()
    
    def test_load_metadata_invalid_format_not_dict(self, tmp_path):
        """Should raise ValueError when JSON is not a dictionary"""
        # Create file with array instead of object
        metadata_file = tmp_path / 'spy_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(['AAPL', 'MSFT'], f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.load_metadata()
        
        assert 'dict' in str(exc_info.value).lower() or 'object' in str(exc_info.value).lower()
    
    def test_load_metadata_success(self, tmp_path):
        """Should successfully load metadata from valid file"""
        # Create valid metadata file
        metadata_file = tmp_path / 'spy_metadata.json'
        expected_metadata = {
            'AAPL': {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 3000000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'last_updated': '2025-12-30T10:00:00Z'
            },
            'MSFT': {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 2800000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'last_updated': '2025-12-30T10:00:00Z'
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(expected_metadata, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        metadata = downloader.load_metadata()
        
        assert metadata == expected_metadata
        assert isinstance(metadata, dict)
        assert 'AAPL' in metadata
        assert 'MSFT' in metadata


# ============================================================================
# PHASE 1, STEP 1.3: Metadata Entry Validation Tests
# ============================================================================

class TestValidateMetadataEntry:
    """Test metadata entry validation"""
    
    def test_validate_metadata_entry_all_fields_present(self, tmp_path):
        """Should return True when all required fields exist"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        valid_entry = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD'
        }
        
        assert downloader._validate_metadata_entry(valid_entry) is True
    
    def test_validate_metadata_entry_missing_field(self, tmp_path):
        """Should return False when any required field missing"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Missing 'sector' field
        incomplete_entry = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD'
        }
        
        assert downloader._validate_metadata_entry(incomplete_entry) is False
    
    def test_validate_metadata_entry_null_field(self, tmp_path):
        """Should return False when field is None"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # 'sector' is None
        null_entry = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': None,
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD'
        }
        
        assert downloader._validate_metadata_entry(null_entry) is False
    
    def test_validate_metadata_entry_unexpected_fields(self, tmp_path):
        """Should return False when entry has unexpected fields"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Has all required fields but also unexpected ones
        entry_with_extra = {
            'symbol': 'AAPL',
            'name': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'market_cap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD',
            'extra_field': 'unexpected',
            'another_unexpected': 123
        }
        
        assert downloader._validate_metadata_entry(entry_with_extra) is False


# ============================================================================
# PHASE 2, STEP 2.1: Planning Updates (Diffing Logic) Tests
# ============================================================================

class TestPlanUpdates:
    """Test planning updates between constituents and metadata"""
    
    def test_plan_updates_empty_metadata(self, tmp_path):
        """Should add all constituents when metadata is empty"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Load constituents and empty metadata
        constituents = downloader.load_constituents()
        metadata = {}  # Empty metadata
        
        to_delete, to_add = downloader.plan_updates(constituents, metadata)
        
        assert to_delete == set()
        assert to_add == {'A', 'B'}
    
    def test_plan_updates_no_changes_needed(self, tmp_path):
        """Should have no changes when metadata matches constituents"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Load constituents
        constituents = downloader.load_constituents()
        
        # Metadata matches constituents
        metadata = {
            'A': {'symbol': 'A', 'name': 'Company A', 'sector': 'Tech', 
                  'industry': 'Software', 'market_cap': 1000000, 
                  'exchange': 'NYSE', 'currency': 'USD'},
            'B': {'symbol': 'B', 'name': 'Company B', 'sector': 'Finance', 
                  'industry': 'Banking', 'market_cap': 2000000, 
                  'exchange': 'NYSE', 'currency': 'USD'}
        }
        
        to_delete, to_add = downloader.plan_updates(constituents, metadata)
        
        assert to_delete == set()
        assert to_add == set()
    
    def test_plan_updates_add_new_symbols(self, tmp_path):
        """Should add new symbols from constituents"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B', 'C'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Load constituents
        constituents = downloader.load_constituents()
        
        # Metadata only has A and B
        metadata = {
            'A': {'symbol': 'A', 'name': 'Company A', 'sector': 'Tech', 
                  'industry': 'Software', 'market_cap': 1000000, 
                  'exchange': 'NYSE', 'currency': 'USD'},
            'B': {'symbol': 'B', 'name': 'Company B', 'sector': 'Finance', 
                  'industry': 'Banking', 'market_cap': 2000000, 
                  'exchange': 'NYSE', 'currency': 'USD'}
        }
        
        to_delete, to_add = downloader.plan_updates(constituents, metadata)
        
        assert to_delete == set()
        assert to_add == {'C'}
    
    def test_plan_updates_delete_old_symbols(self, tmp_path):
        """Should delete symbols not in constituents"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Load constituents
        constituents = downloader.load_constituents()
        
        # Metadata has A and B, but B is no longer in constituents
        metadata = {
            'A': {'symbol': 'A', 'name': 'Company A', 'sector': 'Tech', 
                  'industry': 'Software', 'market_cap': 1000000, 
                  'exchange': 'NYSE', 'currency': 'USD'},
            'B': {'symbol': 'B', 'name': 'Company B', 'sector': 'Finance', 
                  'industry': 'Banking', 'market_cap': 2000000, 
                  'exchange': 'NYSE', 'currency': 'USD'}
        }
        
        to_delete, to_add = downloader.plan_updates(constituents, metadata)
        
        assert to_delete == {'B'}
        assert to_add == set()
    
    def test_plan_updates_both_add_and_delete(self, tmp_path):
        """Should both add and delete symbols"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'C'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Load constituents
        constituents = downloader.load_constituents()
        
        # Metadata has A and B, constituents have A and C
        metadata = {
            'A': {'symbol': 'A', 'name': 'Company A', 'sector': 'Tech', 
                  'industry': 'Software', 'market_cap': 1000000, 
                  'exchange': 'NYSE', 'currency': 'USD'},
            'B': {'symbol': 'B', 'name': 'Company B', 'sector': 'Finance', 
                  'industry': 'Banking', 'market_cap': 2000000, 
                  'exchange': 'NYSE', 'currency': 'USD'}
        }
        
        to_delete, to_add = downloader.plan_updates(constituents, metadata)
        
        assert to_delete == {'B'}
        assert to_add == {'C'}


# ============================================================================
# PHASE 3, STEP 3.1: Single Symbol Download Tests
# ============================================================================

class TestDownloadSymbolMetadata:
    """Test downloading metadata for a single symbol"""
    
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_symbol_metadata_success(self, mock_ticker, tmp_path):
        """Should successfully download and format metadata for a symbol"""
        # Mock yfinance Ticker response
        mock_info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'sector': 'Technology',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD'
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        metadata = downloader.download_symbol_metadata('AAPL')
        
        # Verify correct fields returned
        assert metadata['symbol'] == 'AAPL'
        assert metadata['name'] == 'Apple Inc.'
        assert metadata['sector'] == 'Technology'
        assert metadata['industry'] == 'Consumer Electronics'
        assert metadata['market_cap'] == 3000000000000
        assert metadata['exchange'] == 'NASDAQ'
        assert metadata['currency'] == 'USD'
        
        # Verify yfinance was called correctly
        mock_ticker.assert_called_once_with('AAPL')
    
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_symbol_metadata_missing_field(self, mock_ticker, tmp_path):
        """Should raise ValueError when required field is missing"""
        # Mock yfinance with incomplete data (missing 'sector')
        mock_info = {
            'symbol': 'AAPL',
            'shortName': 'Apple Inc.',
            'industry': 'Consumer Electronics',
            'marketCap': 3000000000000,
            'exchange': 'NASDAQ',
            'currency': 'USD'
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.download_symbol_metadata('AAPL')
        
        assert 'missing required field' in str(exc_info.value).lower()
        assert 'sector' in str(exc_info.value).lower()
    
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_symbol_metadata_yfinance_exception(self, mock_ticker, tmp_path):
        """Should raise ValueError when yfinance raises exception"""
        # Mock yfinance to raise exception
        mock_ticker.side_effect = Exception("Network error")
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        with pytest.raises(ValueError) as exc_info:
            downloader.download_symbol_metadata('AAPL')
        
        assert 'failed' in str(exc_info.value).lower()
        assert 'aapl' in str(exc_info.value).lower()


# ============================================================================
# PHASE 3, STEP 3.2: Batch Download with Progress Tests
# ============================================================================

class TestDownloadMetadataBatch:
    """Test batch downloading metadata with progress tracking"""
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_metadata_all_success(self, mock_ticker, mock_sleep, tmp_path):
        """Should download all symbols successfully"""
        # Mock yfinance for 3 symbols
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        symbols_to_add = ['A', 'B', 'C']
        failed_symbols, new_metadata = downloader.download_metadata_batch(symbols_to_add, rate_limit_delay=0.5)
        
        assert failed_symbols == set()
        assert len(new_metadata) == 3
        assert 'A' in new_metadata
        assert 'B' in new_metadata
        assert 'C' in new_metadata
        
        # Verify rate limiting was called (2 times for 3 symbols: after 1st and 2nd)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.5)
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_metadata_some_failures(self, mock_ticker, mock_sleep, tmp_path):
        """Should track failed symbols and continue with others"""
        # Mock yfinance - symbol 'B' will fail
        def mock_ticker_side_effect(symbol):
            if symbol == 'B':
                raise Exception("Download failed for B")
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        symbols_to_add = ['A', 'B', 'C']
        failed_symbols, new_metadata = downloader.download_metadata_batch(symbols_to_add, rate_limit_delay=0.5)
        
        assert failed_symbols == {'B'}
        assert len(new_metadata) == 2
        assert 'A' in new_metadata
        assert 'C' in new_metadata
        assert 'B' not in new_metadata
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_metadata_respects_rate_limit(self, mock_ticker, mock_sleep, tmp_path):
        """Should enforce rate limiting between downloads"""
        # Mock yfinance
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        symbols_to_add = ['A', 'B', 'C', 'D']
        failed_symbols, new_metadata = downloader.download_metadata_batch(symbols_to_add, rate_limit_delay=1.5)
        
        # Should sleep 3 times for 4 symbols (after 1st, 2nd, 3rd)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_called_with(1.5)
    
    @patch('backtesting.metadata_downloader.tqdm')
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_download_metadata_shows_progress(self, mock_ticker, mock_sleep, mock_tqdm, tmp_path):
        """Should display progress bar with correct total"""
        # Mock yfinance
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Mock tqdm to track calls
        mock_progress = Mock()
        mock_tqdm.return_value = mock_progress
        mock_progress.__iter__ = Mock(return_value=iter(['A', 'B', 'C']))
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        symbols_to_add = ['A', 'B', 'C']
        failed_symbols, new_metadata = downloader.download_metadata_batch(symbols_to_add, rate_limit_delay=0.5)
        
        # Verify tqdm was called with correct parameters
        mock_tqdm.assert_called_once()
        call_args = mock_tqdm.call_args
        assert call_args[0][0] == symbols_to_add  # First positional argument
        assert 'desc' in call_args[1]  # Keyword argument
        assert 'Downloading metadata' in call_args[1]['desc']
