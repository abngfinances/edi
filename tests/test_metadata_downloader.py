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


# ============================================================================
# PHASE 5, STEP 5.1: Orchestration Method Tests
# ============================================================================

class TestUpdateMetadata:
    """Test complete metadata update orchestration"""
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_update_metadata_initial_download(self, mock_ticker, mock_sleep, tmp_path):
        """Should successfully download all metadata when starting from empty"""
        # Create constituents file with 3 symbols
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['AAPL', 'MSFT', 'GOOGL'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance for all symbols
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000000,
                'exchange': 'NASDAQ',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Run update
        stats = downloader.update_metadata(rate_limit_delay=0.1)
        
        # Verify statistics
        assert stats['constituents_count'] == 3
        assert stats['to_delete_count'] == 0
        assert stats['to_add_count'] == 3
        assert stats['failed_count'] == 0
        assert stats['metadata_count'] == 3
        
        # Verify metadata file was created
        assert downloader.metadata_file.exists()
        
        # Verify content
        with open(downloader.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert len(metadata) == 3
        assert 'AAPL' in metadata
        assert 'MSFT' in metadata
        assert 'GOOGL' in metadata
        
        # Verify all entries have timestamps
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            assert 'last_updated' in metadata[symbol]
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_update_metadata_add_and_delete(self, mock_ticker, mock_sleep, tmp_path):
        """Should handle both adding new symbols and deleting old ones"""
        # Create constituents file with A and C
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'C'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Create existing metadata with A and B
        metadata_file = tmp_path / 'spy_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'A': {
                    'symbol': 'A',
                    'name': 'Company A',
                    'sector': 'Technology',
                    'industry': 'Software',
                    'market_cap': 1000000,
                    'exchange': 'NYSE',
                    'currency': 'USD',
                    'last_updated': '2025-12-29T10:00:00Z'
                },
                'B': {
                    'symbol': 'B',
                    'name': 'Company B',
                    'sector': 'Finance',
                    'industry': 'Banking',
                    'market_cap': 2000000,
                    'exchange': 'NYSE',
                    'currency': 'USD',
                    'last_updated': '2025-12-29T10:00:00Z'
                }
            }, f)
        
        # Mock yfinance for symbol C
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 3000000,
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
        
        # Run update
        stats = downloader.update_metadata(rate_limit_delay=0.1)
        
        # Verify statistics
        assert stats['constituents_count'] == 2
        assert stats['to_delete_count'] == 1  # B removed
        assert stats['to_add_count'] == 1     # C added
        assert stats['failed_count'] == 0
        assert stats['metadata_count'] == 2   # A and C
        
        # Verify content
        with open(downloader.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert len(metadata) == 2
        assert 'A' in metadata
        assert 'C' in metadata
        assert 'B' not in metadata  # Should be deleted
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_update_metadata_no_changes(self, mock_ticker, mock_sleep, tmp_path):
        """Should handle case when metadata is already up-to-date"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Create matching metadata
        metadata_file = tmp_path / 'spy_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump({
                'A': {
                    'symbol': 'A',
                    'name': 'Company A',
                    'sector': 'Technology',
                    'industry': 'Software',
                    'market_cap': 1000000,
                    'exchange': 'NYSE',
                    'currency': 'USD',
                    'last_updated': '2025-12-29T10:00:00Z'
                },
                'B': {
                    'symbol': 'B',
                    'name': 'Company B',
                    'sector': 'Finance',
                    'industry': 'Banking',
                    'market_cap': 2000000,
                    'exchange': 'NYSE',
                    'currency': 'USD',
                    'last_updated': '2025-12-29T10:00:00Z'
                }
            }, f)
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Run update
        stats = downloader.update_metadata(rate_limit_delay=0.1)
        
        # Verify statistics - no changes
        assert stats['constituents_count'] == 2
        assert stats['to_delete_count'] == 0
        assert stats['to_add_count'] == 0
        assert stats['failed_count'] == 0
        assert stats['metadata_count'] == 2
        
        # yfinance should not have been called
        mock_ticker.assert_not_called()
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_update_metadata_with_failures_and_ignores(self, mock_ticker, mock_sleep, tmp_path):
        """Should handle download failures correctly when they match ignore_symbols"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B', 'C'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
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
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols={'B'},  # B is expected to fail
            output_dir=str(tmp_path)
        )
        
        # Run update - should succeed because B is in ignore_symbols
        stats = downloader.update_metadata(rate_limit_delay=0.1)
        
        # Verify statistics
        assert stats['constituents_count'] == 3
        assert stats['to_delete_count'] == 0
        assert stats['to_add_count'] == 3
        assert stats['failed_count'] == 1
        assert stats['metadata_count'] == 2  # Only A and C
        
        # Verify content
        with open(downloader.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert len(metadata) == 2
        assert 'A' in metadata
        assert 'C' in metadata
        assert 'B' not in metadata
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_update_metadata_with_unexpected_failures(self, mock_ticker, mock_sleep, tmp_path):
        """Should raise ValueError when download fails for symbol not in ignore_symbols"""
        # Create constituents file
        constituents_file = tmp_path / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
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
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),  # No ignores - B failure is unexpected
            output_dir=str(tmp_path)
        )
        
        # Should raise ValueError due to validation failure
        with pytest.raises(ValueError) as exc_info:
            downloader.update_metadata(rate_limit_delay=0.1)
        
        error_msg = str(exc_info.value)
        assert 'failed_symbols != ignore_symbols' in error_msg.lower()
        assert "{'B'}" in error_msg


# ============================================================================
# PHASE 5, STEP 5.2: CLI Tests
# ============================================================================

class TestMain:
    """Test CLI main function"""
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    @patch('sys.argv', ['metadata_downloader.py', 'SPY', '--output-dir', '/tmp/test'])
    def test_main_success(self, mock_ticker, mock_sleep, tmp_path, capsys):
        """Should successfully run main with minimal arguments"""
        # Create constituents file
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir()
        constituents_file = output_dir / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Patch sys.argv with correct output directory
        with patch('sys.argv', ['metadata_downloader.py', 'SPY', '--output-dir', str(output_dir)]):
            from backtesting.metadata_downloader import main
            exit_code = main()
        
        # Verify success
        assert exit_code == 0
        
        # Verify output contains summary
        captured = capsys.readouterr()
        assert 'Metadata Update Summary' in captured.out
        assert 'Index Symbol:' in captured.out
        assert 'SPY' in captured.out
        assert '✓' in captured.out
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_main_with_ignore_symbols(self, mock_ticker, mock_sleep, tmp_path, capsys, caplog):
        """Should parse ignore-symbols argument correctly"""
        # Create constituents file
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir()
        constituents_file = output_dir / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B', 'C'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance - B will fail
        def mock_ticker_side_effect(symbol):
            if symbol == 'B':
                raise Exception("Download failed")
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Run with ignore-symbols
        with patch('sys.argv', ['metadata_downloader.py', 'SPY', 
                               '--output-dir', str(output_dir),
                               '--ignore-symbols', 'B']):
            from backtesting.metadata_downloader import main
            exit_code = main()
        
        # Should succeed because B is ignored
        assert exit_code == 0
        
        captured = capsys.readouterr()
        assert 'Failed Downloads:       1' in captured.out
        # Verify the specific failed symbol is mentioned in logs
        assert 'B' in caplog.text
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_main_validation_error(self, mock_ticker, mock_sleep, tmp_path, capsys):
        """Should return error code 1 on validation failure"""
        # Create constituents file
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir()
        constituents_file = output_dir / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance - B will fail
        def mock_ticker_side_effect(symbol):
            if symbol == 'B':
                raise Exception("Download failed")
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Run without ignoring B - should fail validation
        with patch('sys.argv', ['metadata_downloader.py', 'SPY', 
                               '--output-dir', str(output_dir)]):
            from backtesting.metadata_downloader import main
            exit_code = main()
        
        # Should fail with exit code 1
        assert exit_code == 1
        
        # Verify error message includes the specific failed symbol
        captured = capsys.readouterr()
        assert '❌' in captured.err
        # Should mention the specific symbol that failed
        assert 'B' in captured.err or 'B' in captured.out
        # Should show it was an unexpected failure
        assert 'unexpected' in captured.err.lower() or 'unexpected' in captured.out.lower()
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_main_multiple_ignore_symbols(self, mock_ticker, mock_sleep, tmp_path, capsys, caplog):
        """Should parse comma-separated ignore-symbols correctly"""
        # Create constituents file
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir()
        constituents_file = output_dir / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B', 'C', 'D'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance - B and D will fail
        def mock_ticker_side_effect(symbol):
            if symbol in ('B', 'D'):
                raise Exception("Download failed")
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Run with multiple ignore-symbols
        with patch('sys.argv', ['metadata_downloader.py', 'SPY', 
                               '--output-dir', str(output_dir),
                               '--ignore-symbols', 'B,D']):
            from backtesting.metadata_downloader import main
            exit_code = main()
        
        # Should succeed
        assert exit_code == 0
        
        captured = capsys.readouterr()
        assert 'Failed Downloads:       2' in captured.out
        # Verify both failed symbols are mentioned in logs
        assert 'B' in caplog.text
        assert 'D' in caplog.text
    
    @patch('backtesting.metadata_downloader.time.sleep')
    @patch('backtesting.metadata_downloader.yf.Ticker')
    def test_main_custom_rate_limit(self, mock_ticker, mock_sleep, tmp_path, capsys):
        """Should use custom rate-limit-delay argument"""
        # Create constituents file
        output_dir = tmp_path / 'test_output'
        output_dir.mkdir()
        constituents_file = output_dir / 'spy_constituents.json'
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': ['A', 'B'],
                'metadata': {'index_symbol': 'SPY'}
            }, f)
        
        # Mock yfinance
        def mock_ticker_side_effect(symbol):
            mock_instance = Mock()
            mock_instance.info = {
                'symbol': symbol,
                'shortName': f'{symbol} Company',
                'sector': 'Technology',
                'industry': 'Software',
                'marketCap': 1000000,
                'exchange': 'NYSE',
                'currency': 'USD'
            }
            return mock_instance
        
        mock_ticker.side_effect = mock_ticker_side_effect
        
        # Run with custom rate limit
        with patch('sys.argv', ['metadata_downloader.py', 'SPY', 
                               '--output-dir', str(output_dir),
                               '--rate-limit-delay', '0.5']):
            from backtesting.metadata_downloader import main
            exit_code = main()
        
        assert exit_code == 0
        
        # Verify rate limit was used (1 sleep for 2 symbols)
        mock_sleep.assert_called_with(0.5)


# ============================================================================
# PHASE 6: Integration Test with Real yfinance Calls
# ============================================================================

class TestIntegration:
    """Integration tests that call real yfinance API (slow, requires network)"""
    
    @pytest.mark.integration
    def test_download_real_metadata(self, tmp_path):
        """
        Integration test: Download real metadata from yfinance
        
        This test makes actual API calls to yfinance and verifies the complete
        end-to-end workflow. Run with: pytest -m integration
        
        Skip by default with: pytest -m "not integration"
        """
        # Create constituents file with 5 real symbols
        constituents_file = tmp_path / 'spy_constituents.json'
        test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'IBM', 'TSLA']
        
        with open(constituents_file, 'w') as f:
            json.dump({
                'symbols': test_symbols,
                'metadata': {
                    'index_symbol': 'SPY',
                    'download_timestamp': '2025-12-30T10:00:00Z',
                    'total_holdings': 5
                }
            }, f)
        
        # Create downloader and run update
        downloader = MetadataDownloader(
            index_symbol='SPY',
            ignore_symbols=set(),
            output_dir=str(tmp_path)
        )
        
        # Run update with real yfinance calls (rate limit to be nice to API)
        stats = downloader.update_metadata(rate_limit_delay=0.5)
        
        # Verify statistics
        assert stats['constituents_count'] == 5
        assert stats['to_delete_count'] == 0
        assert stats['to_add_count'] == 5
        assert stats['failed_count'] == 0  # All symbols should succeed
        assert stats['metadata_count'] == 5
        
        # Verify metadata file was created
        metadata_file = tmp_path / 'spy_metadata.json'
        assert metadata_file.exists()
        
        # Load and verify metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        assert len(metadata) == 5
        
        # Verify each symbol has all required fields
        for symbol in test_symbols:
            assert symbol in metadata, f"Symbol {symbol} missing from metadata"
            
            entry = metadata[symbol]
            
            # Verify all required fields exist
            assert entry['symbol'] == symbol
            assert 'name' in entry
            assert 'sector' in entry
            assert 'industry' in entry
            assert 'market_cap' in entry
            assert 'exchange' in entry
            assert 'currency' in entry
            assert 'last_updated' in entry
            
            # Verify field types
            assert isinstance(entry['name'], str)
            assert isinstance(entry['sector'], str)
            assert isinstance(entry['industry'], str)
            assert isinstance(entry['market_cap'], (int, float))
            assert isinstance(entry['exchange'], str)
            assert isinstance(entry['currency'], str)
            assert isinstance(entry['last_updated'], str)
            
            # Verify non-empty strings
            assert len(entry['name']) > 0
            assert len(entry['sector']) > 0
            assert len(entry['industry']) > 0
            assert entry['market_cap'] > 0
            assert len(entry['exchange']) > 0
            assert entry['currency'] in ['USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
            
            # Verify timestamp format
            assert 'T' in entry['last_updated']
            assert entry['last_updated'].endswith('Z')
        
        # Verify metadata validation passes
        for symbol, entry in metadata.items():
            assert downloader._validate_metadata_entry(entry), \
                f"Validation failed for {symbol}: {entry}"
