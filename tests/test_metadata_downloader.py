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
