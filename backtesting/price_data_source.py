"""
Price Data Source Abstraction

Provides abstract interface and implementations for downloading historical price data.
Supports parallel downloads with proper rate limiting and error handling.
"""

import json
import logging
import time
import yfinance as yf
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class PriceDataSource(ABC):
    """Abstract base class for price data sources"""
    
    @abstractmethod
    def download_symbol(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Download price data for a single symbol.
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Dictionary with:
                - prices: DataFrame with DatetimeIndex and OHLCV columns
                - splits: Series with split ratios (date -> ratio)
                - dividends: Series with dividend amounts (date -> amount)
                
        Raises:
            ValueError: If download fails or data is invalid
        """
        pass
    
    @abstractmethod
    def download_batch_parallel(self, symbols: list, start_date: str, end_date: str, 
                                rate_limit_delay: float = 2.0) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Download price data for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            rate_limit_delay: Delay in seconds between batches (default: 2.0)
            
        Returns:
            Tuple of (failed_symbols, results_dict)
            - failed_symbols: Set of symbols that failed to download
            - results_dict: Dict mapping symbol -> result dict (with prices, splits, dividends)
        """
        pass


class YFinanceSource(PriceDataSource):
    """yfinance implementation with parallel downloads and retry logic"""
    
    def __init__(self, interval: str = '1d', auto_adjust: bool = True, 
                 prepost: bool = False, max_workers: int = 5):
        """
        Initialize yfinance data source.
        
        Args:
            interval: Data interval ('1d', '1wk', '1mo', etc.)
            auto_adjust: Whether to auto-adjust for splits/dividends (default: True)
            prepost: Include pre/post market data (default: False)
            max_workers: Maximum number of parallel download threads (default: 5)
        """
        self.interval = interval
        self.auto_adjust = auto_adjust
        self.prepost = prepost
        self.max_workers = max_workers
        
        logger.info(f"Initialized YFinanceSource: interval={interval}, "
                   f"auto_adjust={auto_adjust}, max_workers={max_workers}")
    
    def download_symbol(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """
        Download price data for a single symbol from yfinance.
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date in 'YYYY-MM-DD' format (INCLUSIVE)
            end_date: End date in 'YYYY-MM-DD' format (INCLUSIVE)
            
        Returns:
            Dictionary with:
                - prices: DataFrame with DatetimeIndex and OHLCV columns
                - splits: Series with split ratios (date -> ratio)
                - dividends: Series with dividend amounts (date -> amount)
                
        Raises:
            ValueError: If download fails or data is invalid
        """
        try:
            # Adjust end_date +1 day to make it inclusive (yfinance uses exclusive end)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            end_dt_adjusted = end_dt + timedelta(days=1)
            end_date_yf = end_dt_adjusted.strftime('%Y-%m-%d')
            
            logger.debug(f"Downloading {symbol} from {start_date} to {end_date} (inclusive)")
            
            ticker = yf.Ticker(symbol)
            
            # Download price history
            prices = ticker.history(
                start=start_date,
                end=end_date_yf,
                interval=self.interval,
                auto_adjust=self.auto_adjust,
                prepost=self.prepost
            )
            
            # Check if we got any data
            if prices.empty or len(prices) == 0:
                error_msg = f"No price data returned for {symbol}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in prices.columns]
            if missing_columns:
                error_msg = f"{symbol} missing required columns: {missing_columns}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Get splits and dividends
            splits = ticker.splits
            dividends = ticker.dividends
            
            # Validate split ratios are whole numbers
            if len(splits) > 0:
                for date, ratio in splits.items():
                    if not abs(ratio - round(ratio)) < 0.001:  # Allow small floating point error
                        error_msg = (f"{symbol} has invalid split ratio {ratio} on {date}. "
                                   f"Split ratios must be whole numbers (e.g., 2.0, 4.0)")
                        logger.error(error_msg)
                        raise ValueError(error_msg)
            
            logger.debug(f"Successfully downloaded {symbol}: {len(prices)} rows, "
                        f"{len(splits)} splits, {len(dividends)} dividends")
            
            return {
                'prices': prices,
                'splits': splits,
                'dividends': dividends
            }
            
        except Exception as e:
            error_msg = f"Failed to download {symbol}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def _download_symbol_with_retry(self, symbol: str, start_date: str, end_date: str,
                                    max_retries: int = 3) -> Dict:
        """
        Download symbol with exponential backoff retry on 429 errors.
        
        Args:
            symbol: Stock symbol to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            Dictionary with prices, splits, dividends
            
        Raises:
            ValueError: If all retries fail
        """
        for attempt in range(max_retries):
            try:
                return self.download_symbol(symbol, start_date, end_date)
            except Exception as e:
                # Check if it's a rate limit error (429)
                if '429' in str(e) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1.0  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Rate limit hit for {symbol}, retrying in {wait_time}s "
                                 f"(attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise
    
    def download_batch_parallel(self, symbols: list, start_date: str, end_date: str,
                                rate_limit_delay: float = 2.0) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Download price data for multiple symbols in parallel using ThreadPoolExecutor.
        
        Args:
            symbols: List of stock symbols to download
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            rate_limit_delay: Delay in seconds after batch completes (default: 2.0)
            
        Returns:
            Tuple of (failed_symbols, results_dict)
            - failed_symbols: Set of symbols that failed to download
            - results_dict: Dict mapping symbol -> result dict (with prices, splits, dividends)
        """
        logger.info(f"Starting parallel download of {len(symbols)} symbols "
                   f"(max_workers={self.max_workers})")
        
        failed_symbols = set()
        results_dict = {}
        
        # Download in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(self._download_symbol_with_retry, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Process completed downloads
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results_dict[symbol] = result
                    logger.debug(f"✓ Downloaded {symbol}")
                except Exception as e:
                    logger.warning(f"✗ Failed to download {symbol}: {e}")
                    failed_symbols.add(symbol)
        
        logger.info(f"Batch download complete: {len(results_dict)} succeeded, "
                   f"{len(failed_symbols)} failed")
        
        # Rate limiting between batches
        if rate_limit_delay > 0:
            logger.debug(f"Rate limiting: sleeping {rate_limit_delay}s")
            time.sleep(rate_limit_delay)
        
        return failed_symbols, results_dict


class PriceDownloader:
    """
    High-level price downloader that orchestrates data downloads for an index.
    
    Responsibilities:
    - Validate all inputs (dates, intervals, etc.)
    - Load constituents and filter ignore symbols
    - Manage data source (yfinance)
    - Plan date ranges and merge logic (Phase 4-5)
    - Checkpoint progress (Phase 6)
    - Orchestrate batch downloads (Phase 7)
    """
    
    VALID_INTERVALS = {'1d', '1wk', '1mo'}
    VALID_SOURCES = {'yfinance'}
    
    def __init__(self,
                 index_symbol: str,
                 metadata_dir: str,
                 output_dir: str,
                 start_date: str,
                 end_date: str,
                 interval: str = '1d',
                 source: str = 'yfinance',
                 ignore_symbols: Optional[Set[str]] = None,
                 batch_size: int = 10,
                 max_workers: int = 5):
        """
        Initialize PriceDownloader with validation.
        
        Args:
            index_symbol: Index to download (e.g., 'SPY', 'IWM')
            metadata_dir: Directory containing constituent files (REQUIRED)
            output_dir: Root directory for price data (REQUIRED)
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            interval: Price interval ('1d', '1wk', '1mo')
            source: Data source ('yfinance')
            ignore_symbols: Symbols to exclude from download
            batch_size: Number of symbols to download per batch (for checkpointing)
            max_workers: Maximum concurrent downloads within a batch
            
        Raises:
            ValueError: If any validation fails
        """
        # Validate dates
        self.start_date = self._validate_date_format(start_date, 'start_date')
        self.end_date = self._validate_date_format(end_date, 'end_date')
        self._validate_date_order(self.start_date, self.end_date)
        self._validate_not_future(self.end_date)
        
        # Validate interval
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Must be one of {self.VALID_INTERVALS}")
        self.interval = interval
        
        # Validate source
        if source not in self.VALID_SOURCES:
            raise ValueError(f"Invalid source '{source}'. Must be one of {self.VALID_SOURCES}")
        self.source = source
        
        # Store config
        self.index_symbol = index_symbol
        self.metadata_dir = Path(metadata_dir)
        self.output_dir = Path(output_dir)
        self.ignore_symbols = ignore_symbols or set()
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # Load constituents
        self.constituents = self._load_constituents()
        
        # Create data source
        self.data_source = YFinanceSource(
            interval=self.interval,
            max_workers=self.max_workers
        )
        
        logger.info(f"Initialized PriceDownloader: {self.index_symbol}, "
                   f"{len(self.constituents)} constituents, "
                   f"{self.start_date} to {self.end_date} (inclusive)")
    
    def _validate_date_format(self, date_str: str, param_name: str) -> str:
        """Validate date is in YYYY-MM-DD format."""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return date_str
        except ValueError:
            raise ValueError(f"{param_name} must be in 'YYYY-MM-DD' format, got '{date_str}'")
    
    def _validate_date_order(self, start_date: str, end_date: str):
        """Validate start_date is before or equal to end_date."""
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        if start > end:
            raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
    
    def _validate_not_future(self, date_str: str):
        """Validate date is not today or in the future (at most yesterday)."""
        date = datetime.strptime(date_str, '%Y-%m-%d')
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if date >= today:
            yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
            raise ValueError(
                f"end_date must be before today (at most {yesterday}) to avoid "
                f"partial trading day data, got {date_str}"
            )
    
    def _load_constituents(self) -> list:
        """
        Load constituents from metadata file and filter ignore_symbols.
        
        Returns:
            List of symbol strings after filtering
            
        Raises:
            ValueError: If constituent file not found or invalid format
        """
        constituents_file = self.metadata_dir / f'{self.index_symbol.lower()}_constituents.json'
        
        if not constituents_file.exists():
            raise ValueError(f"Constituents file not found: {constituents_file}")
        
        try:
            with open(constituents_file, 'r') as f:
                data = json.load(f)
            
            if 'symbols' not in data:
                raise ValueError(f"Invalid constituents file: missing 'symbols' key in {constituents_file}")
            
            symbols = data['symbols']
            if not isinstance(symbols, list):
                raise ValueError(f"Invalid constituents file: 'symbols' must be a list in {constituents_file}")
            
            # Filter ignore_symbols
            filtered = [s for s in symbols if s not in self.ignore_symbols]
            
            if self.ignore_symbols:
                logger.info(f"Filtered {len(symbols) - len(filtered)} ignored symbols, "
                           f"{len(filtered)} remaining")
            
            return filtered
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in constituents file {constituents_file}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to load constituents from {constituents_file}: {e}")
    
    def _get_symbol_folder(self, symbol: str) -> Path:
        """
        Get folder path for a symbol's data.
        
        Structure: {output_dir}/{symbol}/{source}_{interval}/
        Example: backtest_data/AAPL/yfinance_1d/
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Path to symbol's data folder
        """
        return self.output_dir / symbol / f"{self.source}_{self.interval}"
    
    def _get_metadata_path(self, symbol: str) -> Path:
        """
        Get metadata file path for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Path to metadata.json file
        """
        return self._get_symbol_folder(symbol) / 'metadata.json'
    
    def _get_prices_path(self, symbol: str) -> Path:
        """
        Get prices file path for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Path to prices.parquet file
        """
        return self._get_symbol_folder(symbol) / 'prices.parquet'
    
    def _get_checkpoint_path(self) -> Path:
        """
        Get checkpoint file path for tracking download progress.
        
        Checkpoint files are namespaced by source and interval to avoid conflicts
        when downloading multiple configurations (e.g., yfinance_1d vs yfinance_1wk).
        
        Returns:
            Path to {source}_{interval}_progress.json checkpoint file
        """
        checkpoint_filename = f"{self.source}_{self.interval}_progress.json"
        return self.output_dir / 'checkpoints' / checkpoint_filename
    
    def _read_checkpoint(self) -> Set[str]:
        """
        Read checkpoint file to get list of completed symbols.
        
        Validates that checkpoint metadata matches current downloader configuration
        (source, interval, date range) to avoid using stale or mismatched checkpoints.
        
        Returns:
            Set of symbols that have been successfully downloaded
        """
        checkpoint_path = self._get_checkpoint_path()
        
        if not checkpoint_path.exists():
            logger.debug("No checkpoint file found, starting fresh")
            return set()
        
        try:
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            
            # Validate date range matches (source/interval already enforced by filename namespace)
            if data.get('start_date') != self.start_date or data.get('end_date') != self.end_date:
                raise ValueError(
                    f"Checkpoint exists for different date range. "
                    f"Existing download in progress: [{data.get('start_date')}, {data.get('end_date')}]. "
                    f"Current request: [{self.start_date}, {self.end_date}]. "
                    f"Please complete the existing download first by running with dates "
                    f"[{data.get('start_date')}, {data.get('end_date')}], or delete the checkpoint file: {checkpoint_path}"
                )
            
            completed_symbols = data.get('completed_symbols', [])
            logger.info(f"Loaded checkpoint: {len(completed_symbols)} symbols already completed")
            return set(completed_symbols)
            
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read checkpoint file: {e}. Starting fresh.")
            return set()
    
    def _write_checkpoint(self, completed_symbols: Set[str]) -> None:
        """
        Write checkpoint file with list of completed symbols.
        
        Merges with existing checkpoint data to preserve progress across multiple calls.
        
        Args:
            completed_symbols: Set of symbols that have been successfully downloaded
        """
        checkpoint_path = self._get_checkpoint_path()
        
        # Read existing checkpoint and merge with new symbols
        existing_completed = self._read_checkpoint()
        merged_symbols = existing_completed | completed_symbols
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build checkpoint data
        checkpoint_data = {
            'index_symbol': self.index_symbol,
            'interval': self.interval,
            'source': self.source,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'completed_symbols': sorted(list(merged_symbols)),  # Sort for deterministic output
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Write checkpoint file
        try:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            new_count = len(completed_symbols)
            total_count = len(merged_symbols)
            logger.debug(f"Checkpoint saved: {new_count} new symbols, {total_count} total completed")
            
        except IOError as e:
            logger.error(f"Failed to write checkpoint file: {e}")
            raise ValueError(f"Failed to write checkpoint: {e}") from e
    
    def _read_metadata(self, symbol: str) -> Optional[Dict]:
        """
        Read existing metadata for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Metadata dictionary if file exists, None otherwise
            
        Raises:
            ValueError: If metadata file is corrupted or missing required fields
        """
        metadata_path = self._get_metadata_path(symbol)
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Validate required fields (all fields needed for Phase 4 date range planning)
            required_fields = {
                'symbol', 'interval', 'source',
                'start_date', 'end_date', 'total_days',
                'splits', 'dividends', 'last_updated'
            }
            missing_fields = required_fields - set(metadata.keys())
            if missing_fields:
                raise ValueError(f"Metadata missing required fields: {missing_fields}")
            
            return metadata
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted metadata for {symbol}: {e}")
        except Exception as e:
            raise ValueError(f"Failed to read metadata for {symbol}: {e}")
    
    def _write_metadata(self, symbol: str, start_date: str, end_date: str,
                       total_days: int, splits: Dict[str, float], 
                       dividends: Dict[str, float]) -> None:
        """
        Write metadata for a symbol.
        
        Args:
            symbol: Stock symbol
            start_date: Start date of data (YYYY-MM-DD)
            end_date: End date of data (YYYY-MM-DD)
            total_days: Number of trading days
            splits: Dict mapping date string -> split ratio (e.g., {'2020-08-31': 4.0})
            dividends: Dict mapping date string -> dividend amount (e.g., {'2020-02-07': 0.77})
                
        Raises:
            ValueError: If metadata format is invalid
        """
        # Build metadata dictionary with all required fields
        metadata = {
            'symbol': symbol,
            'interval': self.interval,
            'source': self.source,
            'start_date': start_date,
            'end_date': end_date,
            'total_days': total_days,
            'splits': splits,
            'dividends': dividends,
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Validate all required fields present
        required_fields = {
            'symbol', 'interval', 'source',
            'start_date', 'end_date', 'total_days',
            'splits', 'dividends', 'last_updated'
        }
        missing_fields = required_fields - set(metadata.keys())
        if missing_fields:
            raise ValueError(f"Metadata missing required fields: {missing_fields}")
        
        # Verify no None values in required fields
        none_fields = [k for k, v in metadata.items() if v is None]
        if none_fields:
            raise ValueError(f"Metadata has None values for fields: {none_fields}")
        
        # Create folder if it doesn't exist
        metadata_path = self._get_metadata_path(symbol)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Wrote metadata for {symbol}: {start_date} to {end_date}")
            
        except Exception as e:
            raise ValueError(f"Failed to write metadata for {symbol}: {e}")
    
    def _plan_date_range(self, symbol: str, start: str, end: str) -> Optional[list]:
        """
        Plan date ranges to download based on existing metadata.
        
        Note: Different interval/source combinations use separate folders (e.g., yfinance_1d vs yfinance_1wk),
        so there's no conflict when downloading multiple intervals or sources for the same symbol.
        
        Implements 6 merge cases:
        1. No existing metadata → [(start, end)] (full download, includes new interval/source combos)
        2. Subset: requested range within existing → None (skip, already have data)
        3. Extend both: extends before AND after → [(start, existing_start-1), (existing_end+1, end)]
        4. Extend after: contiguous extension beyond end → [(existing_end+1, end)]
        5. Extend before: contiguous extension before start → [(start, existing_start-1)]
        6. Gap: non-contiguous ranges → ERROR (raise ValueError)
        
        Args:
            symbol: Stock symbol
            start: Requested start date (YYYY-MM-DD, pre-validated)
            end: Requested end date (YYYY-MM-DD, pre-validated)
            
        Returns:
            None if data already exists (skip download)
            List of (start, end) date tuples to download
            
        Raises:
            ValueError: If date range has gaps (non-contiguous)
        """
        from datetime import datetime, timedelta
        
        # Load existing metadata
        metadata = self._read_metadata(symbol)
        
        # Case 1: No existing data - download full range
        # (This includes different interval/source combinations, which get separate folders)
        if metadata is None:
            logger.debug(
                f"{symbol}: No existing metadata, downloading full range "
                f"[{start}, {end}]"
            )
            return [(start, end)]
        
        # Parse dates
        existing_start = datetime.strptime(metadata['start_date'], '%Y-%m-%d').date()
        existing_end = datetime.strptime(metadata['end_date'], '%Y-%m-%d').date()
        start_date = datetime.strptime(start, '%Y-%m-%d').date()
        end_date = datetime.strptime(end, '%Y-%m-%d').date()
        
        # Case 2: Subset - new range completely within existing (skip)
        if start_date >= existing_start and end_date <= existing_end:
            logger.debug(
                f"{symbol}: Requested [{start}, {end}] is subset of "
                f"existing [{metadata['start_date']}, {metadata['end_date']}], skipping"
            )
            return None
        
        # Case 3: Extend both sides - new range extends before AND after existing
        if start_date < existing_start and end_date > existing_end:
            # Download two ranges: before and after
            download_before_end = (existing_start - timedelta(days=1)).strftime('%Y-%m-%d')
            download_after_start = (existing_end + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.debug(
                f"{symbol}: Extending both sides. Existing: [{metadata['start_date']}, {metadata['end_date']}], "
                f"downloading [{start}, {download_before_end}] and [{download_after_start}, {end}]"
            )
            return [(start, download_before_end), (download_after_start, end)]
        
        # Case 4: Extend after - new range extends beyond existing end
        if start_date <= existing_end and end_date > existing_end:
            # Download from day after existing_end to new_end
            download_start = (existing_end + timedelta(days=1)).strftime('%Y-%m-%d')
            logger.debug(
                f"{symbol}: Extending after existing end {metadata['end_date']}, "
                f"downloading [{download_start}, {end}]"
            )
            return [(download_start, end)]
        
        # Case 5: Extend before - new range extends before existing start
        if end_date >= existing_start and start_date < existing_start:
            # Download from new_start to day before existing_start
            download_end = (existing_start - timedelta(days=1)).strftime('%Y-%m-%d')
            logger.debug(
                f"{symbol}: Extending before existing start {metadata['start_date']}, "
                f"downloading [{start}, {download_end}]"
            )
            return [(start, download_end)]
        
        # Case 6: Gap - no overlap, new range is completely separate
        # This is an error - user should fix their date ranges
        if end_date < existing_start or start_date > existing_end:
            raise ValueError(
                f"Date range gap for {symbol}: "
                f"Existing data covers [{metadata['start_date']}, {metadata['end_date']}], "
                f"but you requested [{start}, {end}]. "
                f"There is a gap between these ranges. "
                f"Please request a contiguous date range that extends the existing data, "
                f"or delete the existing data folder to start fresh."
            )
        
        # Case 7: Complex overlap (shouldn't reach here with above logic, but defensive)
        raise ValueError(
            f"Unexpected date range scenario for {symbol}. "
            f"Existing: [{metadata['start_date']}, {metadata['end_date']}], "
            f"Requested: [{start}, {end}]. "
            f"Please report this as a bug."
        )
    
    def _verify_no_overlap_and_merge(self, symbol: str, 
                                     new_data: Dict[str, float],
                                     existing_data: Dict[str, float],
                                     data_type: str) -> Dict[str, float]:
        """
        Verify no overlap between new and existing data, then merge.
        
        Following EDI pattern: Error out rather than auto-compensate.
        If overlap exists, it indicates a bug in date range planning.
        
        Args:
            symbol: Stock symbol (for error messages)
            new_data: Newly downloaded data {date_str: value}
            existing_data: Existing data from metadata {date_str: value}
            data_type: 'split' or 'dividend' (for error messages)
            
        Returns:
            Merged dictionary (simple dict merge since no overlap verified)
            
        Raises:
            ValueError: If any overlap detected between new and existing data
        """
        # Check for overlap
        overlapping_dates = set(new_data.keys()) & set(existing_data.keys())
        if overlapping_dates:
            raise ValueError(
                f"{symbol}: {data_type.capitalize()} overlap detected on {len(overlapping_dates)} date(s). "
                f"First overlap: {sorted(overlapping_dates)[0]}. "
                f"Bug in date range planning - new downloads should not overlap with existing data."
            )
        
        # No overlap - safe to merge
        return {**existing_data, **new_data}
    
    def download_symbol_prices(self, symbol: str) -> Dict:
        """
        Download and merge price data for a single symbol.
        
        This method:
        1. Validates symbol is in constituents
        2. Plans date ranges to download using _plan_date_range()
        3. Downloads data for each range
        4. Merges with existing data (prices, splits, dividends)
        5. Writes updated files (prices.parquet, metadata.json)
        
        Args:
            symbol: Stock symbol to download
            
        Returns:
            Dict with keys:
                - symbol: The symbol processed
                - downloaded: True if data was downloaded, False if skipped
                - skipped: True if download was skipped (data already exists)
                
        Raises:
            ValueError: If symbol not in constituents, download fails, or data is empty
        """
        # Validate symbol in constituents
        if symbol not in self.constituents:
            raise ValueError(
                f"Symbol {symbol} not in constituents. "
                f"Available symbols: {len(self.constituents)} total"
            )
        
        # Plan date ranges to download
        ranges_to_download = self._plan_date_range(symbol, self.start_date, self.end_date)
        
        # Skip if no download needed
        if ranges_to_download is None:
            logger.debug(f"{symbol}: Skipping download (data already exists)")
            return {'symbol': symbol, 'downloaded': False, 'skipped': True}
        
        logger.info(f"{symbol}: Downloading {len(ranges_to_download)} range(s)")
        
        # Download and collect each range
        all_prices = []
        all_splits = {}
        all_dividends = {}
        
        for start, end in ranges_to_download:
            logger.debug(f"{symbol}: Downloading range [{start}, {end}]")
            
            # Download data
            try:
                result = self.data_source.download_symbol(symbol, start, end)
            except Exception as e:
                raise ValueError(f"Failed to download {symbol}: {e}") from e
            
            # Validate data
            if result['prices'].empty or len(result['prices']) == 0:
                raise ValueError(f"No price data returned for {symbol} in range [{start}, {end}]")
            
            # Collect prices
            all_prices.append(result['prices'])
            
            # Collect splits - use helper to verify no overlap and merge
            if len(result['splits']) > 0:
                range_splits = {date.strftime('%Y-%m-%d'): ratio 
                               for date, ratio in result['splits'].items()}
                all_splits = self._verify_no_overlap_and_merge(
                    symbol=symbol,
                    new_data=range_splits,
                    existing_data=all_splits,
                    data_type='split'
                )
            
            # Collect dividends - use helper to verify no overlap and merge
            if len(result['dividends']) > 0:
                range_dividends = {date.strftime('%Y-%m-%d'): amount 
                                  for date, amount in result['dividends'].items()}
                all_dividends = self._verify_no_overlap_and_merge(
                    symbol=symbol,
                    new_data=range_dividends,
                    existing_data=all_dividends,
                    data_type='dividend'
                )
        
        # === Process downloaded data ===
        
        # Combine new prices and check for duplicates within downloaded data
        if len(all_prices) > 1:
            # Multiple ranges downloaded - verify no overlap
            combined_new_prices = pd.concat(all_prices, axis=0).sort_index()
            if combined_new_prices.index.duplicated().any():
                duplicate_dates = combined_new_prices.index[combined_new_prices.index.duplicated()].unique()
                raise ValueError(
                    f"{symbol}: Downloaded ranges contain overlapping dates. "
                    f"First duplicate: {duplicate_dates[0]}. "
                    f"Bug in date range planning - ranges should not overlap."
                )
        else:
            combined_new_prices = all_prices[0]
        
        # === Merge with existing data ===
        
        # Load existing data if it exists and verify no overlap
        prices_path = self._get_prices_path(symbol)
        if prices_path.exists():
            logger.debug(f"{symbol}: Loading existing data from {prices_path}")
            existing_prices = pd.read_parquet(prices_path)
            existing_metadata = self._read_metadata(symbol)
            
            # Verify no overlap between new and existing prices
            overlapping_dates = combined_new_prices.index.intersection(existing_prices.index)
            if len(overlapping_dates) > 0:
                raise ValueError(
                    f"{symbol}: Downloaded data overlaps with existing data on {len(overlapping_dates)} date(s). "
                    f"First overlap: {overlapping_dates[0]}. "
                    f"Bug in date range planning - new downloads should not overlap with existing data."
                )
            
            # Combine prices (no overlap, so simple concatenation)
            combined_prices = pd.concat([existing_prices, combined_new_prices], axis=0).sort_index()
            
            # Merge splits/dividends using helper (verifies no overlap)
            if existing_metadata:
                all_splits = self._verify_no_overlap_and_merge(
                    symbol=symbol,
                    new_data=all_splits,
                    existing_data=existing_metadata.get('splits', {}),
                    data_type='split'
                )
                
                all_dividends = self._verify_no_overlap_and_merge(
                    symbol=symbol,
                    new_data=all_dividends,
                    existing_data=existing_metadata.get('dividends', {}),
                    data_type='dividend'
                )
        else:
            # No existing data - use new data directly
            combined_prices = combined_new_prices
        
        # === Write final data ===
        
        # Calculate metadata
        start_date = combined_prices.index.min().strftime('%Y-%m-%d')
        end_date = combined_prices.index.max().strftime('%Y-%m-%d')
        total_days = len(combined_prices)
        
        # Write prices to parquet
        prices_path.parent.mkdir(parents=True, exist_ok=True)
        combined_prices.to_parquet(prices_path)
        logger.debug(f"{symbol}: Wrote {total_days} days of prices to {prices_path}")
        
        # Write metadata
        self._write_metadata(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_days=total_days,
            splits=all_splits,
            dividends=all_dividends
        )
        
        logger.info(f"{symbol}: Successfully downloaded and saved {total_days} days "
                   f"[{start_date}, {end_date}]")
        
        return {'symbol': symbol, 'downloaded': True, 'skipped': False}

