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


# ============================================================================
# PHASE 4, STEP 4.1: Metadata Persistence Tests
# ============================================================================

class TestSaveMetadata:
    """Test metadata persistence to file"""
    
    def test_save_metadata_creates_new_file(self, tmp_path):
        """Should create new metadata file with correct content"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Metadata file doesn't exist yet
        assert not downloader.metadata_file.exists()
        
        # Create metadata to save
        metadata = {
            'AAPL': {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 3000000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
        }
        
        # Save metadata
        downloader.save_metadata(metadata)
        
        # Verify file was created
        assert downloader.metadata_file.exists()
        
        # Verify content
        with open(downloader.metadata_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'AAPL' in saved_data
        
        # Verify all required fields are present
        aapl_data = saved_data['AAPL']
        assert aapl_data['symbol'] == 'AAPL'
        assert aapl_data['name'] == 'Apple Inc.'
        assert aapl_data['sector'] == 'Technology'
        assert aapl_data['industry'] == 'Consumer Electronics'
        assert aapl_data['market_cap'] == 3000000000000
        assert aapl_data['exchange'] == 'NASDAQ'
        assert aapl_data['currency'] == 'USD'
        
        # Verify timestamp was added
        assert 'last_updated' in aapl_data
    
    def test_save_metadata_updates_existing(self, tmp_path):
        """Should correctly update existing metadata file (add/delete)"""
        # Create existing metadata file
        metadata_file = tmp_path / 'spy_metadata.json'
        existing_metadata = {
            'AAPL': {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 3000000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD',
                'last_updated': '2025-12-29T10:00:00Z'
            },
            'OLD': {
                'symbol': 'OLD',
                'name': 'Old Company',
                'sector': 'Finance',
                'industry': 'Banking',
                'market_cap': 1000000000,
                'exchange': 'NYSE',
                'currency': 'USD',
                'last_updated': '2025-12-29T10:00:00Z'
            }
        }
        with open(metadata_file, 'w') as f:
            json.dump(existing_metadata, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # New metadata: keep AAPL, remove OLD, add MSFT
        new_metadata = {
            'AAPL': {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 3000000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            },
            'MSFT': {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 2800000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
        }
        
        # Save updated metadata
        downloader.save_metadata(new_metadata)
        
        # Verify content
        with open(downloader.metadata_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'AAPL' in saved_data
        assert 'MSFT' in saved_data
        assert 'OLD' not in saved_data  # Should be removed
        assert len(saved_data) == 2
    
    def test_save_metadata_includes_timestamps(self, tmp_path):
        """Should add last_updated timestamp to each entry"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Create metadata without timestamps
        metadata = {
            'AAPL': {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'sector': 'Technology',
                'industry': 'Consumer Electronics',
                'market_cap': 3000000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            },
            'MSFT': {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'sector': 'Technology',
                'industry': 'Software',
                'market_cap': 2800000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
        }
        
        # Save metadata
        downloader.save_metadata(metadata)
        
        # Verify timestamps were added
        with open(downloader.metadata_file, 'r') as f:
            saved_data = json.load(f)
        
        assert 'last_updated' in saved_data['AAPL']
        assert 'last_updated' in saved_data['MSFT']
        
        # Verify timestamp format (ISO 8601 with Z)
        timestamp = saved_data['AAPL']['last_updated']
        assert timestamp.endswith('Z')
        assert 'T' in timestamp
        
        # Verify it's a valid ISO 8601 timestamp
        from datetime import datetime
        datetime.fromisoformat(timestamp.replace('Z', '+00:00'))


# ============================================================================
# PHASE 4, STEP 4.2: Ignore Symbols Validation Tests
# ============================================================================

class TestValidateIgnoreSymbols:
    """Test ignore symbols validation logic"""
    
    def test_validate_ignore_symbols_exact_match(self, tmp_path):
        """Should not raise error when failed_symbols exactly match ignore_symbols"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols={'A', 'B'},
            output_dir=str(tmp_path)
        )
        
        failed_symbols = {'A', 'B'}
        
        # Should not raise any exception
        downloader.validate_ignore_symbols(failed_symbols)
    
    def test_validate_ignore_symbols_empty_sets(self, tmp_path):
        """Should not raise error when both sets are empty"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        failed_symbols = set()
        
        # Should not raise any exception
        downloader.validate_ignore_symbols(failed_symbols)
    
    def test_validate_ignore_symbols_unexpected_failures(self, tmp_path):
        """Should raise ValueError when there are unexpected failures"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols={'A'},
            output_dir=str(tmp_path)
        )
        
        failed_symbols = {'A', 'B', 'C'}
        
        with pytest.raises(ValueError) as exc_info:
            downloader.validate_ignore_symbols(failed_symbols)
        
        error_msg = str(exc_info.value)
        
        # Verify error message format
        assert 'Error: failed_symbols != ignore_symbols' in error_msg
        
        # Verify ignore_symbols provided is shown
        assert "ignore_symbols provided: {'A'}" in error_msg
        
        # Verify unexpected failures section with complete set (order may vary)
        assert 'Unexpected failures in metadata download (failed symbols that were NOT in ignore_symbols):' in error_msg
        assert ("{'B', 'C'}" in error_msg or "{'C', 'B'}" in error_msg)
        
        # Verify fix command includes both A, B, C
        assert '--ignore-symbols' in error_msg
        # Fix command should have A,B,C in some order
        assert ('A,B,C' in error_msg or 'A,C,B' in error_msg or 'B,A,C' in error_msg or 
                'B,C,A' in error_msg or 'C,A,B' in error_msg or 'C,B,A' in error_msg)
    
    def test_validate_ignore_symbols_unnecessary_ignores(self, tmp_path):
        """Should raise ValueError when there are unnecessary ignore_symbols"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols={'A', 'B'},
            output_dir=str(tmp_path)
        )
        
        failed_symbols = {'A'}
        
        with pytest.raises(ValueError) as exc_info:
            downloader.validate_ignore_symbols(failed_symbols)
        
        error_msg = str(exc_info.value)
        
        # Verify error message format
        assert 'Error: failed_symbols != ignore_symbols' in error_msg
        
        # Verify the complete ignore_symbols set is shown (order may vary)
        assert "ignore_symbols provided: {'A', 'B'}" in error_msg or "ignore_symbols provided: {'B', 'A'}" in error_msg
        
        # Verify unnecessary ignores section shows B as a set
        assert 'Ignore symbols that did NOT fail (unnecessary):' in error_msg
        assert "{'B'}" in error_msg
        
        # Verify fix command shows only A (the symbol that actually failed)
        assert 'To fix: --ignore-symbols A' in error_msg
    
    def test_validate_ignore_symbols_both_deltas(self, tmp_path):
        """Should show both unexpected failures and unnecessary ignores"""
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols={'A', 'B'},
            output_dir=str(tmp_path)
        )
        
        failed_symbols = {'A', 'C'}
        
        with pytest.raises(ValueError) as exc_info:
            downloader.validate_ignore_symbols(failed_symbols)
        
        error_msg = str(exc_info.value)
        
        # Verify error message format
        assert 'Error: failed_symbols != ignore_symbols' in error_msg
        
        # Verify ignore_symbols provided is shown (order may vary)
        assert "ignore_symbols provided: {'A', 'B'}" in error_msg or "ignore_symbols provided: {'B', 'A'}" in error_msg
        
        # Should show unexpected failures (C) as a set
        assert 'Unexpected failures in metadata download (failed symbols that were NOT in ignore_symbols):' in error_msg
        assert "{'C'}" in error_msg
        
        # Should show unnecessary ignores (B) as a set
        assert 'Ignore symbols that did NOT fail (unnecessary):' in error_msg
        assert "{'B'}" in error_msg
        
        # Should show the fix command with both A and C (order may vary)
        assert '--ignore-symbols' in error_msg
        # Fix command should contain both A and C
        assert ('A,C' in error_msg or 'C,A' in error_msg)
