"""
Index Composition Downloader

Downloads index constituent lists from Alpha Vantage ETF_PROFILE API.
Simple, modular downloader with no fallbacks or checkpointing.
"""

import sys
import json
import logging
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

# Logger will be configured in main() or by caller
logger = logging.getLogger(__name__)


class IndexDownloader:
    """Downloads index composition from Alpha Vantage ETF_PROFILE endpoint"""
    
    def __init__(self, api_key: str, index_symbol: str):
        """
        Initialize the index downloader.
        
        Args:
            api_key: Alpha Vantage API key
            index_symbol: Index/ETF symbol to download (e.g., 'SPY', 'QQQ')
        """
        if not api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        if not index_symbol:
            raise ValueError("Index symbol is required")
        
        self.api_key = api_key
        self.index_symbol = index_symbol
        self.base_url = "https://www.alphavantage.co/query"
        logger.info(f"Initialized IndexDownloader for {self.index_symbol}")
    
    def download_constituents(self) -> Dict:
        """
        Download index constituents from Alpha Vantage ETF_PROFILE endpoint.
        
        Returns:
            Dictionary with symbols list and metadata including download timestamp
            
        Raises:
            ValueError: If API request fails or returns invalid data
        """
        logger.info(f"Downloading constituents for {self.index_symbol} from Alpha Vantage ETF_PROFILE...")
        
        params = {
            'function': 'ETF_PROFILE',
            'symbol': self.index_symbol,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as e:
            error_msg = f"Failed to fetch data from Alpha Vantage: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        # Parse JSON response
        try:
            data = response.json()
            
            # Check for API error messages
            if 'Error Message' in data:
                raise ValueError(data['Error Message'])
            
            if 'Note' in data:
                raise ValueError(data['Note'])
            
            if 'Information' in data:
                raise ValueError(data['Information'])
            
            # Extract holdings
            if 'holdings' not in data:
                raise ValueError(f"No 'holdings' field in API response")
            
            holdings = data['holdings']
            if not holdings:
                raise ValueError("Empty holdings list in API response")
            
            # Extract symbols from holdings - error if any holding lacks 'symbol'
            symbols = []
            for i, holding in enumerate(holdings):
                if 'symbol' not in holding:
                    raise ValueError(f"Holding at index {i} missing required 'symbol' field")
                symbols.append(holding['symbol'])
            
            if not symbols:
                raise ValueError("No symbols found in holdings")
            
            logger.info(f"Successfully downloaded {len(symbols)} symbols")
            
            # Build result with freshness metadata
            result = {
                'symbols': symbols,
                'metadata': {
                    'index_symbol': self.index_symbol,
                    'download_timestamp': datetime.utcnow().isoformat() + 'Z',
                    'total_holdings': len(symbols),
                    'data_source': 'Alpha Vantage ETF_PROFILE'
                }
            }
            
            return result
            
        except (KeyError, json.JSONDecodeError) as e:
            error_msg = f"Failed to parse Alpha Vantage response: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
    
    def save_constituents(self, output_path: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Save index constituents to JSON file.
        
        Args:
            output_path: Path to output JSON file
            data: Data dictionary (if None, will download)
        """
        if data is None:
            data = self.download_constituents()
        
        # Ensure output directory exists
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save with metadata
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        symbol_count = len(data.get('symbols', []))
        logger.info(f"Saved {symbol_count} symbols to {output_path}")
    
    def download_and_save(self, output_dir: str, expected_count: int) -> Dict:
        """
        Download constituents and save to file.
        
        Args:
            output_dir: Directory to save the output file
            expected_count: Expected number of constituents (required for validation)
            
        Returns:
            Dictionary with download statistics
            
        Raises:
            ValueError: If expected_count doesn't match actual count
        """
        # Download data (single API call)
        data = self.download_constituents()
        
        # Validate count (required)
        actual_count = len(data['symbols'])
        if actual_count != expected_count:
            error_msg = (
                f"Symbol count mismatch for {self.index_symbol}: "
                f"expected {expected_count} but got {actual_count}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Generate output filename based on index symbol
        output_filename = f"{self.index_symbol.lower()}_constituents.json"
        output_path = Path(output_dir) / output_filename
        
        # Save to file
        self.save_constituents(str(output_path), data)
        
        return {
            'index_symbol': self.index_symbol,
            'total_symbols': actual_count,
            'expected_symbols': expected_count,
            'output_path': str(output_path),
            'download_timestamp': data['metadata']['download_timestamp']
        }


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download index/ETF constituents from Alpha Vantage ETF_PROFILE'
    )
    parser.add_argument(
        'symbol',
        help='Index/ETF symbol (e.g., SPY, QQQ)'
    )
    parser.add_argument(
        'api_key',
        help='Alpha Vantage API key'
    )
    parser.add_argument(
        'expected_count',
        type=int,
        help='Expected number of constituents (required for validation)'
    )
    parser.add_argument(
        '--output-dir',
        default='backtest_data/metadata',
        help='Output directory (default: backtest_data/metadata)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Configure logging based on argument
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        downloader = IndexDownloader(
            api_key=args.api_key,
            index_symbol=args.symbol
        )
        result = downloader.download_and_save(
            output_dir=args.output_dir,
            expected_count=args.expected_count
        )
        
        logger.info("Download completed successfully!")
        logger.info(f"Index: {result['index_symbol']}")
        logger.info(f"Total symbols: {result['total_symbols']}")
        logger.info(f"Expected symbols: {result['expected_symbols']}")
        logger.info(f"Output file: {result['output_path']}")
        logger.info(f"Download timestamp: {result['download_timestamp']}")
        
        return 0
        
    except ValueError as e:
        logger.error(f"Download failed: {str(e)}")
        return 1
    
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
