"""
Price Data Downloader CLI

Downloads historical price data for index constituents with checkpoint resume support.
Orchestrates batch downloads with parallel processing and progress tracking.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

from backtesting.price_data_source import PriceDownloader

# Logger will be configured in main() or by caller
logger = logging.getLogger(__name__)


def main():
    """Main entry point for standalone execution"""
    parser = argparse.ArgumentParser(
        description='Download historical price data for index constituents',
        epilog='''
Examples:
  # Download daily prices for 2020
  python backtesting/price_data_downloader.py SPY --start-date 2020-01-01 --end-date 2020-12-31
  
  # Resume from checkpoint with custom batch settings
  python backtesting/price_data_downloader.py SPY --start-date 2020-01-01 --end-date 2020-12-31 \\
    --batch-size 5 --max-workers 3 --rate-limit-delay 3.0
  
  # Download weekly data with explicit source
  python backtesting/price_data_downloader.py SPY --start-date 2020-01-01 --end-date 2020-12-31 \\
    --interval 1wk --source yfinance
''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'index_symbol',
        help='Index symbol (e.g., SPY, QQQ)'
    )
    parser.add_argument(
        '--start-date',
        required=True,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date',
        required=True,
        help='End date in YYYY-MM-DD format'
    )
    
    # Optional arguments
    parser.add_argument(
        '--source',
        default='yfinance',
        choices=['yfinance'],
        help='Data source for price downloads (default: yfinance)'
    )
    parser.add_argument(
        '--interval',
        default='1d',
        choices=['1d', '1wk', '1mo'],
        help='Data interval (default: 1d)'
    )
    parser.add_argument(
        '--output-dir',
        default='backtest_data',
        help='Output directory for price data (default: backtest_data)'
    )
    parser.add_argument(
        '--metadata-dir',
        default='backtest_data/metadata',
        help='Directory containing constituent metadata (default: backtest_data/metadata)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=10,
        help='Number of symbols to download per batch (default: 10)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Maximum parallel downloads within each batch (default: 5)'
    )
    parser.add_argument(
        '--rate-limit-delay',
        type=float,
        default=2.0,
        help='Delay in seconds between batches to avoid rate limiting (default: 2.0)'
    )
    parser.add_argument(
        '--ignore-symbols',
        help='Comma-separated list of symbols to ignore (e.g., BRK.B,BF.B)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Parse ignore symbols
        ignore_symbols = None
        if args.ignore_symbols:
            ignore_symbols = set(s.strip() for s in args.ignore_symbols.split(','))
            logger.info(f"Ignoring symbols: {ignore_symbols}")
        
        # Initialize downloader
        logger.info(f"Initializing price downloader for {args.index_symbol}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info(f"Interval: {args.interval}")
        logger.info(f"Source: {args.source}")
        logger.info(f"Batch size: {args.batch_size}, Max workers: {args.max_workers}")
        logger.info("=" * 70)
        
        downloader = PriceDownloader(
            index_symbol=args.index_symbol,
            metadata_dir=args.metadata_dir,
            output_dir=args.output_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            source=args.source,
            ignore_symbols=ignore_symbols,
            batch_size=args.batch_size,
            max_workers=args.max_workers
        )
        
        # Set rate limit delay on the data source
        downloader.data_source.rate_limit_delay = args.rate_limit_delay
        
        # Execute batch download with checkpoint resume
        logger.info("Starting batch download...")
        result = downloader.download_all()
        
        # Log summary
        logger.info("=" * 70)
        logger.info("DOWNLOAD COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Index: {args.index_symbol}")
        logger.info(f"Date range: {args.start_date} to {args.end_date}")
        logger.info(f"Interval: {args.interval}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Total symbols: {result['total']}")
        logger.info(f"Successfully downloaded: {result['completed']}")
        logger.info(f"Failed: {result['failed']}")
        logger.info(f"Skipped (from checkpoint): {result['skipped']}")
        
        if result['failed'] > 0:
            logger.warning(f"{result['failed']} symbols failed to download")
            logger.warning("Check logs above for details")
        
        if result['completed'] > 0:
            logger.info(f"Price data saved to: {args.output_dir}")
            logger.info(f"Checkpoint saved to: {args.output_dir}/checkpoints/")
            logger.info(f"Successful symbols list: {args.output_dir}/checkpoints/price_progress.json")
        
        logger.info("=" * 70)
        
        # Return exit code based on results
        if result['failed'] > 0 and result['completed'] == 0:
            # All downloads failed
            logger.error("All downloads failed")
            return 1
        
        return 0
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        logger.error("Please check your arguments and try again.")
        return 1
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {str(e)}")
        logger.error(f"Required file not found: {str(e)}")
        logger.error("Make sure you have run the constituent and metadata downloaders first:")
        logger.error(f"  1. python backtesting/index_downloader.py {args.index_symbol} <api_key> <count> --output-dir {args.metadata_dir}")
        logger.error(f"  2. python backtesting/metadata_downloader.py {args.index_symbol} --output-dir {Path(args.metadata_dir).parent}")
        return 1
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
