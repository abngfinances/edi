"""
Stock Metadata Downloader

Downloads and maintains stock metadata from yfinance.
Incrementally updates metadata for index constituents.
"""

import argparse
import json
import logging
import sys
import time
import yfinance as yf
from datetime import datetime
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
    
    def save_metadata(self, metadata: Dict[str, Dict]) -> None:
        """
        Save metadata to JSON file with timestamps.
        
        Args:
            metadata: Dictionary mapping symbol -> metadata dict
        """
        logger.info(f"Saving metadata for {len(metadata)} symbols to {self.metadata_file}")
        
        # Add last_updated timestamp to each entry
        timestamped_metadata = {}
        current_time = datetime.utcnow().isoformat() + 'Z'
        
        for symbol, entry in metadata.items():
            timestamped_entry = entry.copy()
            timestamped_entry['last_updated'] = current_time
            timestamped_metadata[symbol] = timestamped_entry
        
        # Ensure output directory exists
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to file
        with open(self.metadata_file, 'w') as f:
            json.dump(timestamped_metadata, f, indent=2)
        
        logger.info(f"Successfully saved metadata to {self.metadata_file}")
    
    def validate_ignore_symbols(self, failed_symbols: Set[str]) -> None:
        """
        Validate that failed symbols match ignore_symbols exactly.
        
        Raises ValueError if there's a mismatch, with detailed error message
        showing which symbols to add/remove from ignore_symbols.
        
        Args:
            failed_symbols: Set of symbols that failed to download
            
        Raises:
            ValueError: If failed_symbols != ignore_symbols
        """
        if failed_symbols == self.ignore_symbols:
            logger.info("Failed symbols match ignore_symbols exactly")
            return
        
        # Calculate deltas
        unexpected_failures = failed_symbols - self.ignore_symbols
        unnecessary_ignores = self.ignore_symbols - failed_symbols
        
        # Build detailed error message
        error_lines = [
            "Error: failed_symbols != ignore_symbols",
            f"ignore_symbols provided: {self.ignore_symbols}",
        ]
        
        if unexpected_failures:
            error_lines.append(
                f"Unexpected failures in metadata download (failed symbols that were NOT in ignore_symbols): {unexpected_failures}"
            )
        
        if unnecessary_ignores:
            error_lines.append(
                f"Ignore symbols that did NOT fail (unnecessary): {unnecessary_ignores}"
            )
        
        # Generate fix command
        correct_ignore_symbols = failed_symbols
        if correct_ignore_symbols:
            symbols_str = ','.join(sorted(correct_ignore_symbols))
            error_lines.append(f"To fix: --ignore-symbols {symbols_str}")
        else:
            error_lines.append("To fix: remove --ignore-symbols argument (no failures expected)")
        
        error_msg = '\n'.join(error_lines)
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    def update_metadata(self, rate_limit_delay: float = 1.0) -> Dict[str, any]:
        """
        Orchestrate the full metadata update process.
        
        This method combines all phases:
        1. Load constituents
        2. Load existing metadata
        3. Plan updates (diff)
        4. Download new metadata
        5. Delete removed symbols and merge new metadata
        6. Save updated metadata
        7. Validate ignore symbols
        
        Args:
            rate_limit_delay: Delay in seconds between yfinance requests (default: 1.0)
            
        Returns:
            Dictionary with statistics about the update:
            - constituents_count: Number of constituent symbols
            - to_delete_count: Number of symbols removed
            - to_add_count: Number of symbols to add
            - failed_count: Number of failed downloads
            - metadata_count: Final metadata count
            
        Raises:
            ValueError: If validation fails or required files missing
        """
        logger.info(f"Starting metadata update for {self.index_symbol}")
        
        # Phase 1: Load data
        logger.info("Phase 1: Loading constituents and existing metadata")
        constituents = self.load_constituents()
        metadata = self.load_metadata()
        
        logger.info(f"Loaded {len(constituents)} constituents, {len(metadata)} existing metadata entries")
        
        # Phase 2: Plan updates
        logger.info("Phase 2: Planning updates")
        to_delete, to_add = self.plan_updates(constituents, metadata)
        
        logger.info(f"Plan: delete {len(to_delete)} symbols, add {len(to_add)} symbols")
        if to_delete:
            logger.info(f"Symbols to delete: {sorted(to_delete)}")
        if to_add:
            logger.info(f"Symbols to add: {sorted(to_add)}")
        
        # Phase 3: Download new metadata
        failed_symbols = set()
        if to_add:
            logger.info(f"Phase 3: Downloading metadata for {len(to_add)} symbols")
            failed_symbols, new_metadata = self.download_metadata_batch(
                to_add, 
                rate_limit_delay=rate_limit_delay
            )
            
            if failed_symbols:
                logger.warning(f"Failed to download {len(failed_symbols)} symbols: {sorted(failed_symbols)}")
            
            logger.info(f"Successfully downloaded {len(new_metadata)} symbols")
        else:
            logger.info("Phase 3: No new symbols to download")
            new_metadata = {}
        
        # Phase 4: Delete removed symbols and merge new metadata
        logger.info("Phase 4: Updating metadata")
        for symbol in to_delete:
            del metadata[symbol]
            logger.debug(f"Deleted {symbol} from metadata")
        
        metadata.update(new_metadata)
        logger.info(f"Updated metadata: {len(metadata)} total entries")
        
        # Phase 5: Save metadata
        logger.info("Phase 5: Saving metadata to file")
        self.save_metadata(metadata)
        logger.info(f"Metadata saved to {self.metadata_file}")
        
        # Phase 6: Validate ignore symbols
        logger.info("Phase 6: Validating ignore symbols")
        self.validate_ignore_symbols(failed_symbols)
        logger.info("Ignore symbols validation passed")
        
        # Summary
        stats = {
            'constituents_count': len(constituents),
            'to_delete_count': len(to_delete),
            'to_add_count': len(to_add),
            'failed_count': len(failed_symbols),
            'metadata_count': len(metadata)
        }
        
        logger.info(f"Metadata update complete: {stats}")
        return stats


def main():
    """
    Main CLI entry point for metadata downloader.
    
    Parses command-line arguments and runs the metadata update process.
    """
    parser = argparse.ArgumentParser(
        description='Download and maintain stock metadata for index constituents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download SPY metadata to current directory
  python metadata_downloader.py SPY
  
  # Download with specific output directory
  python metadata_downloader.py SPY --output-dir backtest_data/metadata
  
  # Ignore known failing symbols
  python metadata_downloader.py SPY --ignore-symbols BRK.B,BF.B
  
  # Enable debug logging
  python metadata_downloader.py SPY --log-level DEBUG
        """
    )
    
    # Positional arguments
    parser.add_argument(
        'index_symbol',
        type=str,
        help='Index symbol (e.g., SPY, QQQ)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--ignore-symbols',
        type=str,
        default='',
        help='Comma-separated list of symbols to ignore failures for (e.g., BRK.B,BF.B)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory containing input/output files (default: current directory)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=1.0,
        help='Delay in seconds between yfinance requests (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Parse ignore_symbols
    ignore_symbols = set()
    if args.ignore_symbols:
        ignore_symbols = set(s.strip() for s in args.ignore_symbols.split(',') if s.strip())
        logger.info(f"Ignoring failures for symbols: {sorted(ignore_symbols)}")
    
    try:
        # Create downloader
        downloader = MetadataDownloader(
            index_symbol=args.index_symbol,
            ignore_symbols=ignore_symbols,
            output_dir=args.output_dir
        )
        
        # Run update
        logger.info(f"Starting metadata update for {args.index_symbol}")
        stats = downloader.update_metadata(rate_limit_delay=args.rate_limit_delay)
        
        # Print summary
        print("\n" + "="*60)
        print("Metadata Update Summary")
        print("="*60)
        print(f"Index Symbol:           {args.index_symbol}")
        print(f"Constituents:           {stats['constituents_count']}")
        print(f"Symbols to Delete:      {stats['to_delete_count']}")
        print(f"Symbols to Add:         {stats['to_add_count']}")
        print(f"Failed Downloads:       {stats['failed_count']}")
        print(f"Final Metadata Count:   {stats['metadata_count']}")
        print("="*60)
        print(f"✓ Metadata saved to {downloader.metadata_file}")
        print("="*60 + "\n")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        print(f"\n❌ Error: {e}\n", file=sys.stderr)
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"\n❌ Unexpected error: {e}\n", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
