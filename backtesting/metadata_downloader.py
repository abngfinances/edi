"""
Stock Metadata Downloader

Downloads and maintains stock metadata from yfinance.
Incrementally updates metadata for index constituents.
"""

import json
import logging
from typing import List, Dict, Set
from pathlib import Path

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
