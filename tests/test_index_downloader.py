"""
Tests for Index Downloader

Verifies that index composition is downloaded correctly from Alpha Vantage ETF_PROFILE.
"""

import json
import pytest
import logging
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from backtesting.index_downloader import IndexDownloader

# Configure logging for tests - allow override via environment variable
log_level = os.getenv('TEST_LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@pytest.fixture
def mock_alphavantage_etf_response():
    """Mock JSON response from Alpha Vantage ETF_PROFILE endpoint"""
    return {
        "net_assets": "403000000000",
        "net_expense_ratio": "0.0018",
        "inception_date": "1999-03-10",
        "holdings": [
            {"symbol": "NVDA", "description": "NVIDIA CORP", "weight": "0.0899"},
            {"symbol": "AAPL", "description": "APPLE INC", "weight": "0.0794"},
            {"symbol": "MSFT", "description": "MICROSOFT CORP", "weight": "0.0712"},
            {"symbol": "AMZN", "description": "AMAZON.COM INC", "weight": "0.0487"},
            {"symbol": "TSLA", "description": "TESLA INC", "weight": "0.0421"}
        ]
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary directory for test output"""
    output_dir = tmp_path / "backtest_data" / "metadata"
    output_dir.mkdir(parents=True)
    return output_dir


def pytest_addoption(parser):
    """Add custom command-line options for tests"""
    parser.addoption(
        "--log-level",
        action="store",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level for tests"
    )


def pytest_configure(config):
    """Configure logging level from command-line argument"""
    log_level = config.getoption("--log-level")
    logging.getLogger().setLevel(getattr(logging, log_level))


class TestIndexDownloader:
    """Test suite for IndexDownloader"""
    
    def test_initialization_with_params(self):
        """Test initialization with explicit parameters"""
        logger.info("Testing IndexDownloader initialization with parameters")
        
        downloader = IndexDownloader(
            api_key='test_api_key',
            index_symbol='SPY'
        )
        
        assert downloader.api_key == 'test_api_key'
        assert downloader.index_symbol == 'SPY'
        logger.info("✓ Initialization with parameters successful")
    
    def test_missing_api_key_raises_error(self):
        """Test that missing API key raises ValueError"""
        logger.info("Testing error handling for missing API key")
        
        with pytest.raises(ValueError, match="Alpha Vantage API key is required"):
            IndexDownloader(api_key='', index_symbol='SPY')
        
        logger.info("✓ Missing API key error handled correctly")
    
    def test_missing_index_symbol_raises_error(self):
        """Test that missing index symbol raises ValueError"""
        logger.info("Testing error handling for missing index symbol")
        
        with pytest.raises(ValueError, match="Index symbol is required"):
            IndexDownloader(api_key='test_key', index_symbol='')
        
        logger.info("✓ Missing index symbol error handled correctly")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_constituents_success(self, mock_get, mock_alphavantage_etf_response):
        """Test successful download of constituents"""
        logger.info("Testing successful constituent download")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alphavantage_etf_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        data = downloader.download_constituents()
        
        assert 'symbols' in data
        assert 'metadata' in data
        assert len(data['symbols']) == 5
        assert 'NVDA' in data['symbols']
        assert 'AAPL' in data['symbols']
        assert 'MSFT' in data['symbols']
        assert 'AMZN' in data['symbols']
        assert 'TSLA' in data['symbols']
        
        # Check metadata
        assert data['metadata']['index_symbol'] == 'SPY'
        assert data['metadata']['total_holdings'] == 5
        assert 'download_timestamp' in data['metadata']
        assert data['metadata']['data_source'] == 'Alpha Vantage ETF_PROFILE'
        
        logger.info(f"✓ Successfully downloaded {len(data['symbols'])} symbols with metadata")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_constituents_network_error(self, mock_get):
        """Test handling of network errors"""
        logger.info("Testing network error handling")
        
        import requests
        mock_get.side_effect = requests.RequestException("Network error")
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        with pytest.raises(ValueError, match="Failed to fetch data from Alpha Vantage"):
            downloader.download_constituents()
        
        logger.info("✓ Network error handled correctly")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_constituents_api_error(self, mock_get):
        """Test handling of API error messages"""
        logger.info("Testing API error message handling")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"Error Message": "Invalid API key"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        with pytest.raises(ValueError, match="Invalid API key"):
            downloader.download_constituents()
        
        logger.info("✓ API error message handled correctly")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_constituents_missing_holdings(self, mock_get):
        """Test handling of response without holdings"""
        logger.info("Testing missing holdings handling")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"net_assets": "1000000"}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        with pytest.raises(ValueError, match="No 'holdings' field"):
            downloader.download_constituents()
        
        logger.info("✓ Missing holdings error handled correctly")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_constituents_missing_symbol_in_holding(self, mock_get, mock_alphavantage_etf_response):
        """Test that missing 'symbol' field in a holding raises error"""
        logger.info("Testing missing symbol field in holding")
        
        # Modify response to have a holding without 'symbol'
        malformed_response = mock_alphavantage_etf_response.copy()
        malformed_response['holdings'] = [
            {"symbol": "NVDA", "description": "NVIDIA CORP", "weight": "0.0899"},
            {"description": "APPLE INC", "weight": "0.0794"},  # Missing 'symbol'
            {"symbol": "MSFT", "description": "MICROSOFT CORP", "weight": "0.0712"}
        ]
        
        mock_response = MagicMock()
        mock_response.json.return_value = malformed_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        with pytest.raises(ValueError, match="Holding at index 1 missing required 'symbol' field"):
            downloader.download_constituents()
        
        logger.info("✓ Missing symbol field error handled correctly")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_save_constituents(self, mock_get, mock_alphavantage_etf_response, temp_output_dir):
        """Test saving constituents to JSON file"""
        logger.info("Testing constituent file save")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alphavantage_etf_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        output_path = temp_output_dir / "test_constituents.json"
        data = downloader.download_constituents()
        downloader.save_constituents(str(output_path), data)
        
        # Verify file exists
        assert output_path.exists()
        
        # Verify file content
        with open(output_path, 'r') as f:
            loaded_data = json.load(f)
        
        assert 'symbols' in loaded_data
        assert 'metadata' in loaded_data
        assert len(loaded_data['symbols']) == 5
        assert 'NVDA' in loaded_data['symbols']
        assert 'download_timestamp' in loaded_data['metadata']
        
        logger.info(f"✓ File saved successfully with {len(loaded_data['symbols'])} symbols and metadata")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_and_save(self, mock_get, mock_alphavantage_etf_response, temp_output_dir):
        """Test complete download and save workflow"""
        logger.info("Testing complete download and save workflow")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alphavantage_etf_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        result = downloader.download_and_save(
            output_dir=str(temp_output_dir),
            expected_count=5
        )
        
        # Verify return value
        assert result['index_symbol'] == 'SPY'
        assert result['total_symbols'] == 5
        assert result['expected_symbols'] == 5
        assert 'download_timestamp' in result
        
        # Verify file path uses lowercase symbol
        expected_filename = 'spy_constituents.json'
        assert expected_filename in result['output_path']
        
        # Verify file
        output_path = Path(result['output_path'])
        assert output_path.exists()
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert len(data['symbols']) == 5
        assert 'metadata' in data
        
        logger.info("✓ Complete workflow successful with correct filename")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_and_save_count_match(self, mock_get, mock_alphavantage_etf_response, 
                                          temp_output_dir):
        """Test successful download when expected count matches actual"""
        logger.info("Testing expected count validation - matching counts")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alphavantage_etf_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        # Should complete successfully when count matches
        result = downloader.download_and_save(
            output_dir=str(temp_output_dir),
            expected_count=5  # Expecting 5 and will get 5
        )
        
        assert result['total_symbols'] == 5
        assert result['expected_symbols'] == 5
        
        # Verify file was created
        output_path = Path(result['output_path'])
        assert output_path.exists()
        
        logger.info("✓ Count match validation passed - no error raised")
    
    @patch('backtesting.index_downloader.requests.get')
    def test_download_and_save_count_mismatch(self, mock_get, mock_alphavantage_etf_response, 
                                             temp_output_dir):
        """Test error raised when expected count doesn't match actual"""
        logger.info("Testing expected count validation - mismatched counts")
        
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_alphavantage_etf_response
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        downloader = IndexDownloader(
            api_key='test_key',
            index_symbol='SPY'
        )
        
        # Should raise ValueError when count doesn't match
        with pytest.raises(ValueError, match="Symbol count mismatch for SPY: expected 500 but got 5"):
            downloader.download_and_save(
                output_dir=str(temp_output_dir),
                expected_count=500  # Expecting 500 but will get 5
            )
        
        logger.info("✓ Count mismatch error raised correctly")


@pytest.mark.integration
class TestIndexDownloaderIntegration:
    """Integration tests that hit the real Alpha Vantage API"""
    
    def test_real_api_download(self, temp_output_dir):
        """
        Integration test with real Alpha Vantage API.
        Verifies that downloaded file exists and contains expected number of symbols.
        
        Usage:
            pytest tests/test_index_downloader.py::TestIndexDownloaderIntegration::test_real_api_download \
                -k integration --symbol SPY --expected-count 503 --api-key YOUR_KEY
        """
        logger.info("Running integration test with real Alpha Vantage ETF_PROFILE API")
        
        # Test parameters
        api_key = 'DUJRZATXXEQL2M9R'  # Default test API key
        symbol = 'QQQ'  # Default to QQQ
        expected_count = 103  # Default expectation (updated 2025-01)
        
        logger.info(f"Testing with symbol={symbol}, expected_count={expected_count}")
        
        downloader = IndexDownloader(
            api_key=api_key,
            index_symbol=symbol
        )
        
        result = downloader.download_and_save(
            output_dir=str(temp_output_dir),
            expected_count=expected_count
        )
        
        output_path = Path(result['output_path'])
        
        # Verify file exists
        assert output_path.exists(), "Downloaded file does not exist"
        logger.info(f"✓ File exists at {output_path}")
        
        # Verify file content
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        assert 'symbols' in data, "JSON file missing 'symbols' key"
        assert 'metadata' in data, "JSON file missing 'metadata' key"
        
        symbol_count = len(data['symbols'])
        
        # Verify we have the expected count (or close to it)
        logger.info(f"Downloaded {symbol_count} symbols (expected {expected_count})")
        
        # Allow 5% variance in count
        min_count = int(expected_count * 0.95)
        max_count = int(expected_count * 1.05)
        assert min_count <= symbol_count <= max_count, \
            f"Expected between {min_count}-{max_count} symbols, got {symbol_count}"
        
        # Verify metadata fields
        assert 'download_timestamp' in data['metadata']
        assert 'index_symbol' in data['metadata']
        assert data['metadata']['index_symbol'] == symbol
        assert data['metadata']['total_holdings'] == symbol_count
        
        logger.info(f"✓ Downloaded {symbol_count} symbols with complete metadata")
        logger.info(f"✓ Timestamp: {data['metadata']['download_timestamp']}")
        logger.info(f"✓ Integration test passed for {symbol}")
