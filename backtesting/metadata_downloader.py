"""
Stock Metadata Downloader

Downloads and maintains stock metadata from yfinance.
Incrementally updates metadata for index constituents.
"""

import json
import logging
import time
import yfinance as yf
from typing import List, Dict, Set, Tuple
from pathlib import Path
from tqdm import tqdm

# Logger will be configured in main() or by caller
logger = logging.getLogger(__name__)


class MetadataDownloader:
    """Downloads and maintains stock metadata using yfinance"""
    
    def __init__(self, index_symbol: str, ignore_symbols: Set[str], output_dir: str):
        """
        Initialize the metadata downloader.
        
        Args:
            index_symbol: Index symbol (e.g., 'SPY', 'QQQ')
            ignore_symbols: Set of symbols to ignore failures for
            output_dir: Directory containing input/output files
        """
        if not index_symbol:
            raise ValueError("Index symbol is required")
        
        self.index_symbol = index_symbol
        self.ignore_symbols = ignore_symbols
        self.output_dir = Path(output_dir)
        
        # File paths
        self.constituents_file = self.output_dir / f"{index_symbol.lower()}_constituents.json"
        self.metadata_file = self.output_dir / f"{index_symbol.lower()}_metadata.json"
        
        logger.info(f"Initialized MetadataDownloader for {self.index_symbol}")
    
    def load_constituents(self) -> List[str]:
        """
        Load constituent symbols from JSON file.
        
        Returns:
            List of symbol strings
            
        Raises:
            ValueError: If file not found or has invalid format
        """
        logger.info(f"Loading constituents from {self.constituents_file}")
        
        # Check if file exists
        if not self.constituents_file.exists():
            error_msg = f"Constituents file not found: {self.constituents_file}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Load and parse JSON
        try:
            with open(self.constituents_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse constituents file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Validate format
        if 'symbols' not in data:
            error_msg = "Constituents file missing 'symbols' key"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verify index_symbol matches expected value
        if 'index_symbol' in data:
            file_index = data['index_symbol']
            if file_index != self.index_symbol:
                error_msg = (f"Index symbol mismatch: expected '{self.index_symbol}', "
                            f"but file contains '{file_index}'")
                logger.error(error_msg)
                raise ValueError(error_msg)
            logger.info(f"Verified index_symbol matches: {self.index_symbol}")
        else:
            logger.warning(f"Constituents file missing 'index_symbol' field")
        
        symbols = data['symbols']
        
        if not isinstance(symbols, list):
            error_msg = "Constituents 'symbols' must be a list/array"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully loaded {len(symbols)} symbols")
        return symbols
    
    def load_metadata(self) -> Dict[str, Dict]:
        """
        Load existing metadata from JSON file or return empty dict if not found.
        
        Returns:
            Dictionary mapping symbol -> metadata dict
            
        Raises:
            ValueError: If file exists but has invalid format
        """
        logger.info(f"Loading metadata from {self.metadata_file}")
        
        # If file doesn't exist, return empty dict (will be created later)
        if not self.metadata_file.exists():
            logger.info("Metadata file not found, returning empty dict")
            return {}
        
        # Load and parse JSON
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse metadata file: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Validate format - must be a dictionary
        if not isinstance(data, dict):
            error_msg = "Metadata file must contain a dictionary/object"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully loaded metadata for {len(data)} symbols")
        return data
    
    def _validate_metadata_entry(self, entry: Dict) -> bool:
        """
        Validate that a metadata entry has all required fields and no unexpected fields.
        
        Args:
            entry: Metadata dictionary for a single symbol
            
        Returns:
            True if all required fields present, non-null, and no unexpected fields
        """
        required_fields = {
            'symbol',
            'name',
            'sector',
            'industry',
            'market_cap',
            'exchange',
            'currency'
        }
        
        entry_fields = set(entry.keys())
        
        # Check all required fields are present and non-null
        for field in required_fields:
            if field not in entry or entry[field] is None:
                return False
        
        # Check no unexpected fields (allow 'last_updated' as it's added during persistence)
        allowed_fields = required_fields | {'last_updated'}
        if not entry_fields.issubset(allowed_fields):
            return False
        
        return True
    
    def plan_updates(self, constituents: List[str], metadata: Dict[str, Dict]) -> tuple[Set[str], Set[str]]:
        """
        Plan updates by computing diff between constituents and metadata.
        
        Args:
            constituents: List of symbol strings from constituents file
            metadata: Dictionary mapping symbol -> metadata dict
            
        Returns:
            Tuple of (to_delete, to_add) sets
            - to_delete: Symbols in metadata but not in constituents
            - to_add: Symbols in constituents but not in metadata
        """
        logger.info("Planning metadata updates...")
        
        constituents_set = set(constituents)
        metadata_set = set(metadata.keys())
        
        # Symbols to delete: in metadata but not in constituents
        to_delete = metadata_set - constituents_set
        
        # Symbols to add: in constituents but not in metadata
        to_add = constituents_set - metadata_set
        
        logger.info(f"Plan: delete {len(to_delete)} symbols, add {len(to_add)} symbols")
        if to_delete:
            logger.debug(f"To delete: {sorted(to_delete)}")
        if to_add:
            logger.debug(f"To add: {sorted(to_add)}")
        
        return to_delete, to_add
    
    def download_symbol_metadata(self, symbol: str) -> Dict:
        """
        Download metadata for a single symbol from yfinance.
        
        Args:
            symbol: Stock symbol to download
            
        Returns:
            Dictionary with metadata fields
            
        Raises:
            ValueError: If download fails or required fields are missing
        """
        logger.debug(f"Downloading metadata for {symbol}")
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
        except Exception as e:
            error_msg = f"Failed to download metadata for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Extract required fields
        required_mapping = {
            'symbol': 'symbol',
            'name': 'shortName',
            'sector': 'sector',
            'industry': 'industry',
            'market_cap': 'marketCap',
            'exchange': 'exchange',
            'currency': 'currency'
        }
        
        metadata = {}
        missing_fields = []
        
        for our_field, yf_field in required_mapping.items():
            value = info.get(yf_field)
            if value is None:
                missing_fields.append(yf_field)
            else:
                metadata[our_field] = value
        
        # Check if any required fields are missing
        if missing_fields:
            error_msg = f"Symbol {symbol} missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Successfully downloaded metadata for {symbol}")
        return metadata
    
    def download_metadata_batch(self, symbols_to_add: List[str], rate_limit_delay: float = 1.0) -> Tuple[Set[str], Dict[str, Dict]]:
        """
        Download metadata for multiple symbols with progress tracking and rate limiting.
        
        Args:
            symbols_to_add: List of symbols to download
            rate_limit_delay: Delay in seconds between downloads (default: 1.0)
            
        Returns:
            Tuple of (failed_symbols, new_metadata)
            - failed_symbols: Set of symbols that failed to download
            - new_metadata: Dict mapping symbol -> metadata dict for successful downloads
        """
        logger.info(f"Downloading metadata for {len(symbols_to_add)} symbols...")
        
        failed_symbols = set()
        new_metadata = {}
        
        # Download with progress bar
        for i, symbol in enumerate(tqdm(symbols_to_add, desc="Downloading metadata")):
            try:
                metadata = self.download_symbol_metadata(symbol)
                new_metadata[symbol] = metadata
            except ValueError as e:
                logger.warning(f"Failed to download {symbol}: {e}")
                failed_symbols.add(symbol)
            
            # Rate limiting (don't sleep after last symbol)
            if i < len(symbols_to_add) - 1:
                time.sleep(rate_limit_delay)
        
        logger.info(f"Downloaded {len(new_metadata)} symbols, {len(failed_symbols)} failed")
        return failed_symbols, new_metadata
