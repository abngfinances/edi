"""
Price Data Source Abstraction

Provides abstract interface and implementations for downloading historical price data.
Supports parallel downloads with proper rate limiting and error handling.
"""

import logging
import time
import yfinance as yf
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Set, Tuple, Optional
from datetime import datetime
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
        logger.debug(f"Downloading {symbol} from {start_date} to {end_date}")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Download price history
            prices = ticker.history(
                start=start_date,
                end=end_date,
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
