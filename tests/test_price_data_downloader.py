"""
Tests for Price Data Downloader CLI

Tests follow TDD phases: argument parsing, validation, download orchestration, CLI integration.
"""

import json
import pytest
import logging
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from backtesting.price_data_downloader import main

# Configure logging for tests
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Phase 1: Argument Parsing Tests
# ============================================================================

class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation"""
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_required_arguments_parsed(self, mock_downloader_class):
        """Should parse required arguments: index_symbol, start_date, end_date"""
        # Mock downloader instance
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 5, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        # Execute
        exit_code = main()
        
        # Verify PriceDownloader was initialized with correct args
        assert mock_downloader_class.called
        call_kwargs = mock_downloader_class.call_args[1]
        assert call_kwargs['index_symbol'] == 'SPY'
        assert call_kwargs['start_date'] == '2020-01-01'
        assert call_kwargs['end_date'] == '2020-12-31'
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'QQQ', '--start-date', '2020-01-01', 
                        '--end-date', '2020-12-31', '--interval', '1wk', '--batch-size', '5'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_optional_arguments_parsed(self, mock_downloader_class):
        """Should parse optional arguments with custom values"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 10, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        call_kwargs = mock_downloader_class.call_args[1]
        assert call_kwargs['interval'] == '1wk'
        assert call_kwargs['batch_size'] == 5
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', 
                        '--end-date', '2020-12-31', '--ignore-symbols', 'BRK.B,BF.B'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_ignore_symbols_parsed(self, mock_downloader_class):
        """Should parse comma-separated ignore symbols"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 3, 'completed': 3, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        call_kwargs = mock_downloader_class.call_args[1]
        assert call_kwargs['ignore_symbols'] == {'BRK.B', 'BF.B'}
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01',
                        '--end-date', '2020-12-31', '--max-workers', '10', '--rate-limit-delay', '5.0'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_performance_parameters_parsed(self, mock_downloader_class):
        """Should parse max-workers and rate-limit-delay"""
        mock_instance = MagicMock()
        mock_instance.data_source = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 5, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        call_kwargs = mock_downloader_class.call_args[1]
        assert call_kwargs['max_workers'] == 10
        # rate_limit_delay is set on data_source after initialization
        assert mock_instance.data_source.rate_limit_delay == 5.0
        assert exit_code == 0


# ============================================================================
# Phase 2: Validation & Error Handling Tests
# ============================================================================

class TestCLIValidation:
    """Test CLI validation and error handling"""
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', 'invalid-date', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_invalid_date_format_returns_error_code(self, mock_downloader_class):
        """Should return exit code 1 when date format is invalid"""
        mock_downloader_class.side_effect = ValueError("Invalid date format: invalid-date")
        
        exit_code = main()
        
        assert exit_code == 1
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-12-31', '--end-date', '2020-01-01'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_end_before_start_returns_error_code(self, mock_downloader_class):
        """Should return exit code 1 when end_date is before start_date"""
        mock_downloader_class.side_effect = ValueError("end_date must be >= start_date")
        
        exit_code = main()
        
        assert exit_code == 1
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_missing_constituents_file_returns_error_code(self, mock_downloader_class, caplog):
        """Should return error code when constituents file is missing"""
        mock_downloader_class.side_effect = FileNotFoundError("spy_constituents.json not found")
        
        exit_code = main()
        
        assert exit_code == 1
        log_output = caplog.text
        # Should show helpful message about running prerequisite steps
        assert 'index_downloader' in log_output
        assert 'metadata_downloader' in log_output
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_download_failure_returns_success_if_some_complete(self, mock_downloader_class):
        """Should return exit code 0 if at least some downloads succeed"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 7, 'failed': 3, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        # Should succeed if some downloads completed
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_all_downloads_fail_returns_error_code(self, mock_downloader_class):
        """Should return exit code 1 if all downloads fail"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 0, 'failed': 5, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        assert exit_code == 1


# ============================================================================
# Phase 3: Download Orchestration Tests
# ============================================================================

class TestCLIDownloadOrchestration:
    """Test CLI download orchestration and checkpoint resume"""
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_calls_download_all(self, mock_downloader_class):
        """Should call download_all method on downloader instance"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 5, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        # Verify download_all was called
        mock_instance.download_all.assert_called_once()
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_checkpoint_resume_workflow(self, mock_downloader_class):
        """Should support checkpoint resume when rerunning with same parameters"""
        mock_instance = MagicMock()
        # Simulate checkpoint resume: 2 completed before, 3 new downloads
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 3, 'failed': 0, 'skipped': 2
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        assert exit_code == 0
        # download_all handles checkpoint internally
        mock_instance.download_all.assert_called_once()
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', 
                        '--end-date', '2020-12-31', '--batch-size', '3', '--max-workers', '2'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_respects_batch_and_worker_settings(self, mock_downloader_class):
        """Should pass batch_size and max_workers to downloader"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 10, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        call_kwargs = mock_downloader_class.call_args[1]
        assert call_kwargs['batch_size'] == 3
        assert call_kwargs['max_workers'] == 2
        assert exit_code == 0


# ============================================================================
# Phase 4: Output & Logging Tests
# ============================================================================

class TestCLIOutput:
    """Test CLI output formatting and summary display"""
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_success_summary_displayed(self, mock_downloader_class, caplog):
        """Should display formatted summary on successful completion"""
        caplog.set_level(logging.INFO)
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 8, 'failed': 2, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        log_output = caplog.text
        # Verify summary contains key information
        assert 'DOWNLOAD COMPLETE' in log_output
        assert 'Total symbols: 10' in log_output
        assert 'Successfully downloaded: 8' in log_output
        assert 'Failed: 2' in log_output
        assert 'Skipped (from checkpoint): 0' in log_output
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_checkpoint_resume_summary(self, mock_downloader_class, caplog):
        """Should show skipped count in summary when resuming from checkpoint"""
        caplog.set_level(logging.INFO)
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 3, 'failed': 0, 'skipped': 7
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        log_output = caplog.text
        assert 'Successfully downloaded: 3' in log_output
        assert 'Skipped (from checkpoint): 7' in log_output
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-12-31'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_warning_displayed_on_failures(self, mock_downloader_class, caplog):
        """Should display warning when some downloads fail"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 10, 'completed': 7, 'failed': 3, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        exit_code = main()
        
        log_output = caplog.text
        assert 'WARNING' in log_output
        assert '3 symbols failed' in log_output
        assert exit_code == 0
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', 
                        '--end-date', '2020-12-31', '--log-level', 'DEBUG'])
    @patch('backtesting.price_data_downloader.PriceDownloader')
    def test_log_level_configured(self, mock_downloader_class, caplog):
        """Should configure logging level from CLI argument"""
        mock_instance = MagicMock()
        mock_instance.download_all.return_value = {
            'total': 5, 'completed': 5, 'failed': 0, 'skipped': 0
        }
        mock_downloader_class.return_value = mock_instance
        
        with caplog.at_level(logging.DEBUG):
            exit_code = main()
        
        # Should have debug level logs
        assert any('Initializing price downloader' in record.message for record in caplog.records)
        assert exit_code == 0


# ============================================================================
# Phase 5: File System Integration Test (mocked data source)
# ============================================================================

class TestCLIFileSystemIntegration:
    """Test CLI with real file system but mocked data source"""
    
    @patch('sys.argv', ['price_data_downloader.py', 'SPY', '--start-date', '2020-01-01', '--end-date', '2020-01-10'])
    @patch('backtesting.price_data_source.YFinanceSource.download_symbol')
    def test_end_to_end_download_flow(self, mock_download_symbol, tmp_path):
        """Should perform end-to-end download with file creation (mocked data source)"""
        # Setup temporary directories
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        output_dir = tmp_path / "backtest_data"
        
        # Create constituents file
        constituents_file = metadata_dir / "spy_constituents.json"
        constituents_file.write_text(json.dumps({
            'symbols': ['AAPL', 'MSFT', 'GOOGL']
        }))
        
        # Mock download_symbol to return test data
        import pandas as pd
        mock_prices = pd.DataFrame({
            'Open': [100.0], 'High': [105.0], 'Low': [99.0],
            'Close': [104.0], 'Volume': [1000000]
        }, index=pd.date_range('2020-01-02', periods=1, freq='D'))
        
        mock_download_symbol.return_value = {
            'prices': mock_prices,
            'splits': pd.Series([], dtype=float),
            'dividends': pd.Series([], dtype=float)
        }
        
        # Patch sys.argv to include temp paths
        with patch('sys.argv', [
            'price_data_downloader.py', 'SPY',
            '--start-date', '2020-01-01',
            '--end-date', '2020-01-10',
            '--metadata-dir', str(metadata_dir),
            '--output-dir', str(output_dir)
        ]):
            exit_code = main()
        
        # Verify success
        assert exit_code == 0
        
        # Verify checkpoint was created
        checkpoint_dir = output_dir / "checkpoints"
        assert checkpoint_dir.exists()
        
        # Verify price files were created (at least one symbol folder)
        symbol_folders = list(output_dir.glob('*/'))
        assert len(symbol_folders) > 0


# ============================================================================
# Phase 6: Real Integration Test (marked for optional execution)
# ============================================================================

@pytest.mark.integration
class TestRealYFinanceIntegration:
    """Real integration test with yfinance API (slow, requires network)"""
    
    def test_download_real_price_data(self, tmp_path):
        """Should download real historical prices from yfinance"""
        logger.info("Running integration test with real yfinance API")
        
        # Setup temporary directories
        metadata_dir = tmp_path / "metadata"
        metadata_dir.mkdir()
        output_dir = tmp_path / "backtest_data"
        
        # Create constituents file with real symbols
        # Using AAPL, MSFT, GOOGL to test the fix for historical split filtering
        constituents_file = metadata_dir / "spy_constituents.json"
        constituents_file.write_text(json.dumps({
            'symbols': ['AAPL', 'MSFT', 'GOOGL'],  # Test symbols (MSFT/GOOGL have fractional splits)
            'metadata': {
                'index_symbol': 'SPY',
                'download_timestamp': '2020-01-01T00:00:00',
                'total_holdings': 3
            }
        }))
        
        # Run CLI with real API (no mocking) - only 2 days for speed
        # Date range: 2020-01-02 to 2020-01-03 (should NOT include splits from 1991/2014)
        with patch('sys.argv', [
            'price_data_downloader.py', 'SPY',
            '--start-date', '2020-01-02',
            '--end-date', '2020-01-03',  # Only 2 days in 2020
            '--metadata-dir', str(metadata_dir),
            '--output-dir', str(output_dir),
            '--batch-size', '3',
            '--rate-limit-delay', '2.0'  # Be nice to yfinance
        ]):
            exit_code = main()
        
        # The test should pass - verify we got the files
        # If it fails due to split validation, the bug is that we're getting
        # historical splits from outside our requested date range
        assert exit_code == 0, f"Download failed with exit code {exit_code}"
        
        # Check which symbols succeeded (skip checkpoints directory)
        symbol_dirs = [d for d in output_dir.glob('*/') if d.name != 'checkpoints']
        logger.info(f"Symbol directories created: {[d.name for d in symbol_dirs]}")
        
        # Verify we got all 3 symbols
        assert len(symbol_dirs) == 3, f"Expected 3 symbols, got {len(symbol_dirs)}"
        
        # If we have any successful downloads, verify the structure
        for symbol_dir in symbol_dirs:
            symbol = symbol_dir.name
            data_dir = symbol_dir / 'yfinance_1d'
            
            logger.info(f"Verifying {symbol} data...")
            assert data_dir.exists(), f"{symbol} data directory not found"
            assert (data_dir / 'prices.parquet').exists(), f"{symbol} prices.parquet not found"
            assert (data_dir / 'metadata.json').exists(), f"{symbol} metadata.json not found"
            
            # Check the actual data
            import pandas as pd
            prices = pd.read_parquet(data_dir / 'prices.parquet')
            logger.info(f"{symbol}: Downloaded {len(prices)} price records")
            logger.info(f"{symbol}: Price date range: {prices.index.min()} to {prices.index.max()}")
            
            # Verify all OHLCV columns are present
            expected_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
            assert expected_columns.issubset(prices.columns), \
                f"{symbol}: Missing columns. Expected {expected_columns}, got {set(prices.columns)}"
            
            # Verify metadata
            metadata = json.loads((data_dir / 'metadata.json').read_text())
            assert metadata['symbol'] == symbol
            assert metadata['start_date'] == '2020-01-02'
            assert metadata['end_date'] == '2020-01-03'
            logger.info(f"{symbol}: Metadata verified")
        
        logger.info("Real integration test completed - AAPL, MSFT, and GOOGL downloaded successfully!")
