"""
Historical Data Downloader for Tax Loss Harvesting Backtesting

Downloads historical price data from free sources:
1. Yahoo Finance (via yfinance) - Most reliable, free, 5+ years of data
2. Alpha Vantage - Backup source
3. Financial Modeling Prep - Already have API key

This script downloads and prepares data for backtesting.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import time
from tqdm import tqdm
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

class BacktestConfig:
    """Configuration for backtesting data"""
    DATA_DIR = 'backtest_data'
    PRICE_DATA_DIR = f'{DATA_DIR}/prices'
    METADATA_DIR = f'{DATA_DIR}/metadata'
    CHECKPOINT_DIR = f'{DATA_DIR}/checkpoints'
    
    # Date range for backtesting (5 years recommended)
    START_DATE = '2019-01-01'
    END_DATE = '2024-12-31'
    
    # Number of stocks to download
    NUM_STOCKS = 100
    
    # File paths
    SP500_LIST_FILE = f'{METADATA_DIR}/sp500_constituents.json'
    PRICE_DATA_FILE = f'{PRICE_DATA_DIR}/all_prices.parquet'
    METADATA_FILE = f'{METADATA_DIR}/stock_metadata.json'
    
    # Checkpoint files
    METADATA_CHECKPOINT = f'{CHECKPOINT_DIR}/metadata_progress.json'
    PRICE_CHECKPOINT = f'{CHECKPOINT_DIR}/price_progress.json'
    TEMP_PRICE_DIR = f'{CHECKPOINT_DIR}/temp_prices'
    
    LOG_FILE = f'{DATA_DIR}/data_download.log'

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    os.makedirs(BacktestConfig.DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(BacktestConfig.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# DATA SOURCES
# ============================================================================

class DataDownloader:
    """Download historical data from multiple sources"""
    
    def __init__(self):
        self.sp500_symbols: List[str] = []
        self.metadata: Dict = {}
        os.makedirs(BacktestConfig.PRICE_DATA_DIR, exist_ok=True)
        os.makedirs(BacktestConfig.METADATA_DIR, exist_ok=True)
        os.makedirs(BacktestConfig.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BacktestConfig.TEMP_PRICE_DIR, exist_ok=True)
    
    def get_sp500_list(self) -> List[str]:
        """Get current S&P 500 constituents with multiple fallback methods"""
        
        # Method 1: Try Wikipedia with proper headers
        try:
            logger.info("Fetching S&P 500 constituents from Wikipedia...")
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            
            # Use requests with headers first
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse with pandas
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            
            symbols = sp500_table['Symbol'].tolist()
            symbols = [s.replace('.', '-') for s in symbols]
            
            sp500_data = {
                'symbols': symbols,
                'download_date': datetime.now().isoformat(),
                'source': 'wikipedia',
                'count': len(symbols)
            }
            
            with open(BacktestConfig.SP500_LIST_FILE, 'w') as f:
                json.dump(sp500_data, f, indent=2)
            
            logger.info(f"✓ Found {len(symbols)} S&P 500 stocks from Wikipedia")
            self.sp500_symbols = symbols
            return symbols
            
        except Exception as e:
            logger.warning(f"Wikipedia fetch failed: {e}")
        
        # Method 2: Try slickcharts.com (alternative source)
        try:
            logger.info("Trying alternate source: slickcharts.com...")
            url = 'https://www.slickcharts.com/sp500'
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            tables = pd.read_html(response.text)
            sp500_table = tables[0]
            
            # Different column name on slickcharts
            if 'Symbol' in sp500_table.columns:
                symbols = sp500_table['Symbol'].tolist()
            elif 'Ticker' in sp500_table.columns:
                symbols = sp500_table['Ticker'].tolist()
            else:
                raise ValueError("Could not find symbol column")
            
            symbols = [str(s).replace('.', '-') for s in symbols]
            
            sp500_data = {
                'symbols': symbols,
                'download_date': datetime.now().isoformat(),
                'source': 'slickcharts',
                'count': len(symbols)
            }
            
            with open(BacktestConfig.SP500_LIST_FILE, 'w') as f:
                json.dump(sp500_data, f, indent=2)
            
            logger.info(f"✓ Found {len(symbols)} S&P 500 stocks from slickcharts")
            self.sp500_symbols = symbols
            return symbols
            
        except Exception as e:
            logger.warning(f"Slickcharts fetch failed: {e}")
        
        # Method 3: Use hardcoded list of major S&P 500 stocks
        logger.warning("All online sources failed. Using hardcoded fallback list...")
        
        # Top ~200 S&P 500 stocks by market cap (as of Dec 2024)
        hardcoded_symbols = [
            # Mega cap tech (Top 10)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'GOOG', 'AVGO',
            
            # Large cap tech (11-40)
            'ORCL', 'ADBE', 'CRM', 'CSCO', 'ACN', 'INTC', 'AMD', 'QCOM', 'IBM', 'TXN',
            'INTU', 'AMAT', 'MU', 'ADI', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT',
            'PANW', 'CRWD', 'ADSK', 'WDAY', 'ABNB', 'DDOG', 'SNOW', 'ZS', 'NET', 'NOW',
            
            # Financials (41-70)
            'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'SPGI', 'BLK', 'C', 'AXP',
            'SCHW', 'CB', 'MMC', 'PGR', 'AON', 'ICE', 'CME', 'MCO', 'TRV', 'ALL',
            'USB', 'PNC', 'TFC', 'COF', 'AIG', 'MET', 'PRU', 'AFL', 'HIG',
            
            # Healthcare (71-100)
            'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'AMGN',
            'CVS', 'BMY', 'ELV', 'GILD', 'MDT', 'CI', 'REGN', 'VRTX', 'HUM', 'ISRG',
            'ZTS', 'BSX', 'SYK', 'EW', 'IDXX', 'A', 'IQV', 'RMD', 'DXCM', 'ALGN',
            
            # Consumer Discretionary (101-130)
            'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'TJX', 'BKNG',
            'CMG', 'MAR', 'DHI', 'LEN', 'YUM', 'GM', 'F', 'ORLY', 'AZO', 'RCL',
            'CCL', 'HLT', 'MGM', 'WYNN', 'LVS', 'POOL', 'ULTA', 'DRI', 'ROST', 'BBY',
            
            # Consumer Staples (131-150)
            'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'CL', 'MDLZ', 'GIS',
            'KMB', 'KHC', 'HSY', 'K', 'CAG', 'SJM', 'CPB', 'CHD', 'CLX', 'TSN',
            
            # Communications (151-165)
            'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'NXST', 'TTWO',
            'MTCH', 'FOXA', 'FOX', 'PARA', 'OMC',
            
            # Industrials (166-195)
            'CAT', 'BA', 'UNP', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'GE', 'MMM',
            'NOC', 'FDX', 'EMR', 'ETN', 'ITW', 'PH', 'CSX', 'NSC', 'GD', 'TDG',
            'CARR', 'OTIS', 'WM', 'RSG', 'FAST', 'PCAR', 'VRSK', 'IEX', 'ROK', 'DOV',
            
            # Energy (196-210)
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',
            'HAL', 'BKR', 'KMI', 'WMB', 'OKE',
            
            # Materials (211-225)
            'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'CTVA', 'DD', 'PPG', 'NUE',
            'VMC', 'MLM', 'STLD', 'CF', 'MOS',
            
            # Utilities (226-240)
            'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'ES', 'PEG',
            'ED', 'WEC', 'AWK', 'DTE', 'ETR',
            
            # Real Estate (241-255)
            'PLD', 'AMT', 'EQIX', 'CCI', 'PSA', 'WELL', 'SPG', 'O', 'DLR', 'VICI',
            'AVB', 'EQR', 'INVH', 'MAA', 'ESS',
            
            # Additional quality stocks to reach ~200
            'GRMN', 'MKTX', 'TYL', 'PAYC', 'ADP', 'PAYX', 'BR', 'FIS', 'FISV', 'GPN',
            'TROW', 'BEN', 'IVZ', 'STT', 'NTRS', 'BK', 'BBT', 'RF', 'KEY', 'CFG',
            'FITB', 'HBAN', 'MTB', 'ZION', 'CMA', 'SIVB', 'ALLY', 'NAVI', 'DFS', 'SYF'
        ]
        
        # Remove duplicates and clean
        symbols = list(set(hardcoded_symbols))
        symbols.sort()
        
        sp500_data = {
            'symbols': symbols,
            'download_date': datetime.now().isoformat(),
            'source': 'hardcoded_fallback',
            'count': len(symbols),
            'note': 'Using hardcoded list of ~200 major S&P 500 stocks. Not complete list but sufficient for backtesting.'
        }
        
        with open(BacktestConfig.SP500_LIST_FILE, 'w') as f:
            json.dump(sp500_data, f, indent=2)
        
        logger.info(f"✓ Using {len(symbols)} hardcoded S&P 500 stocks")
        logger.warning("⚠️  Using fallback list - covers major stocks but not all 500")
        self.sp500_symbols = symbols
        return symbols
    
    def download_metadata_yfinance(self, symbols: List[str]) -> Dict:
        """Download stock metadata using yfinance"""
        logger.info(f"Downloading metadata for {len(symbols)} stocks...")
        
        metadata = {}
        
        for i, symbol in enumerate(tqdm(symbols, desc="Downloading metadata")):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                metadata[symbol] = {
                    'symbol': symbol,
                    'name': info.get('longName', info.get('shortName', '')),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'exchange': info.get('exchange', ''),
                    'currency': info.get('currency', 'USD')
                }
                
                # Rate limiting
                if (i + 1) % 50 == 0:
                    logger.info(f"Progress: {i+1}/{len(symbols)} completed")
                    time.sleep(1)  # Avoid rate limits
                
            except Exception as e:
                logger.warning(f"Could not fetch metadata for {symbol}: {e}")
                metadata[symbol] = {
                    'symbol': symbol,
                    'name': symbol,
                    'sector': 'Unknown',
                    'industry': 'Unknown',
                    'market_cap': 0,
                    'exchange': '',
                    'currency': 'USD'
                }
        
        # Save metadata
        with open(BacktestConfig.METADATA_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved for {len(metadata)} stocks")
        self.metadata = metadata
        return metadata
    
    def download_prices_yfinance(self, symbols: List[str], 
                                 start_date: str, end_date: str) -> pd.DataFrame:
        """Download historical prices using yfinance with checkpoint support"""
        logger.info(f"Downloading price data for {len(symbols)} stocks...")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        # Load checkpoint
        checkpoint = self._load_checkpoint(BacktestConfig.PRICE_CHECKPOINT)
        completed = set(checkpoint.get('completed', []))
        failed = set(checkpoint.get('failed', []))
        
        # Load existing temp files
        existing_data = []
        for symbol in completed:
            temp_file = f"{BacktestConfig.TEMP_PRICE_DIR}/{symbol}.parquet"
            if os.path.exists(temp_file):
                try:
                    df = pd.read_parquet(temp_file)
                    existing_data.append(df)
                except Exception as e:
                    logger.warning(f"Could not load temp file for {symbol}: {e}")
                    completed.discard(symbol)
        
        remaining = [s for s in symbols if s not in completed and s not in failed]
        
        if not remaining:
            logger.info("All price data already downloaded! Combining files...")
            if existing_data:
                combined_df = pd.concat(existing_data, ignore_index=True)
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                combined_df = combined_df.sort_values(['date', 'symbol']).reset_index(drop=True)
                
                # Save final files
                combined_df.to_parquet(BacktestConfig.PRICE_DATA_FILE, index=False)
                csv_file = BacktestConfig.PRICE_DATA_FILE.replace('.parquet', '.csv')
                combined_df.to_csv(csv_file, index=False)
                
                logger.info(f"Combined {len(existing_data)} files into final dataset")
                return combined_df
            return pd.DataFrame()
        
        logger.info(f"Resuming: {len(completed)} completed, {len(remaining)} remaining, {len(failed)} failed")
        
        # Download remaining symbols in batches
        batch_size = 50
        
        for batch_idx in range(0, len(remaining), batch_size):
            batch = remaining[batch_idx:batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(remaining) - 1) // batch_size + 1
            
            logger.info(f"Downloading batch {batch_num}/{total_batches} ({len(batch)} symbols)")
            
            try:
                # Download batch
                data = yf.download(
                    batch,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    group_by='ticker',
                    auto_adjust=True,
                    threads=True  # Enable multi-threading
                )
                
                # Process each symbol in batch
                for symbol in batch:
                    try:
                        if len(batch) == 1:
                            symbol_data = data
                        else:
                            symbol_data = data[symbol]
                        
                        if not symbol_data.empty and len(symbol_data) > 0:
                            # Create DataFrame with consistent columns
                            df = pd.DataFrame({
                                'date': symbol_data.index,
                                'open': symbol_data['Open'].values,
                                'high': symbol_data['High'].values,
                                'low': symbol_data['Low'].values,
                                'close': symbol_data['Close'].values,
                                'volume': symbol_data['Volume'].values,
                                'symbol': symbol
                            })
                            
                            # Save to temp file
                            temp_file = f"{BacktestConfig.TEMP_PRICE_DIR}/{symbol}.parquet"
                            df.to_parquet(temp_file, index=False)
                            
                            completed.add(symbol)
                            existing_data.append(df)
                        else:
                            failed.add(symbol)
                            logger.warning(f"No data for {symbol}")
                    
                    except Exception as e:
                        failed.add(symbol)
                        logger.warning(f"Error processing {symbol}: {e}")
                
                # Save checkpoint after each batch
                self._save_checkpoint(BacktestConfig.PRICE_CHECKPOINT, {
                    'completed': list(completed),
                    'failed': list(failed),
                    'last_update': datetime.now().isoformat(),
                    'batch_completed': batch_num
                })
                
                logger.info(f"Batch {batch_num} complete. Total: {len(completed)}/{len(symbols)}")
                
                time.sleep(1)  # Rate limiting between batches
                
            except Exception as e:
                logger.error(f"Error downloading batch {batch_num}: {e}")
                # Mark all symbols in failed batch as failed for this attempt
                for symbol in batch:
                    if symbol not in completed:
                        logger.warning(f"Marking {symbol} for retry")
                
                # Save checkpoint even on error
                self._save_checkpoint(BacktestConfig.PRICE_CHECKPOINT, {
                    'completed': list(completed),
                    'failed': list(failed),
                    'last_update': datetime.now().isoformat(),
                    'last_error': str(e),
                    'batch_failed': batch_num
                })
        
        if failed:
            logger.warning(f"Failed to download {len(failed)} symbols: {list(failed)[:10]}...")
        
        # Combine all data into single DataFrame
        if existing_data:
            combined_df = pd.concat(existing_data, ignore_index=True)
            combined_df['date'] = pd.to_datetime(combined_df['date'])
            combined_df = combined_df.sort_values(['date', 'symbol']).reset_index(drop=True)
            
            # Save final files
            combined_df.to_parquet(BacktestConfig.PRICE_DATA_FILE, index=False)
            csv_file = BacktestConfig.PRICE_DATA_FILE.replace('.parquet', '.csv')
            combined_df.to_csv(csv_file, index=False)
            
            # Mark as finished
            self._save_checkpoint(BacktestConfig.PRICE_CHECKPOINT, {
                'completed': list(completed),
                'failed': list(failed),
                'last_update': datetime.now().isoformat(),
                'finished': True
            })
            
            logger.info(f"Downloaded {len(completed)} stocks successfully")
            logger.info(f"Data saved to {BacktestConfig.PRICE_DATA_FILE}")
            logger.info(f"Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
            logger.info(f"Total rows: {len(combined_df):,}")
            
            return combined_df
        else:
            logger.error("No data downloaded!")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """Validate downloaded data quality"""
        logger.info("Validating data quality...")
        
        validation_report = {
            'total_symbols': df['symbol'].nunique(),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max()),
                'trading_days': df['date'].nunique()
            },
            'data_quality': {},
            'issues': []
        }
        
        # Get reference trading days (use SPY as benchmark if available)
        reference_symbol = None
        for sym in ['SPY', 'AAPL', 'MSFT', 'GOOGL']:
            if sym in df['symbol'].unique():
                reference_symbol = sym
                break
        
        if reference_symbol:
            reference_data = df[df['symbol'] == reference_symbol]
            expected_trading_days = len(reference_data)
            logger.info(f"Using {reference_symbol} as reference: {expected_trading_days} trading days")
        else:
            # Fallback: estimate ~252 trading days per year
            date_range = (df['date'].max() - df['date'].min()).days
            years = date_range / 365.25
            expected_trading_days = int(years * 252)
            logger.info(f"No reference symbol found. Estimating {expected_trading_days} trading days")
        
        # Check each symbol
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol]
            
            # Check for missing data (compare to expected trading days)
            actual_days = len(symbol_data)
            completeness = (actual_days / expected_trading_days) * 100
            
            # Cap at 100% (some stocks might have more data than reference)
            completeness = min(completeness, 100.0)
            
            # Check for zero/negative prices
            invalid_prices = (symbol_data['close'] <= 0).sum()
            
            # Check for missing values
            missing_values = symbol_data.isnull().sum().sum()
            
            # Check for data gaps (missing trading days)
            if len(symbol_data) > 1:
                symbol_dates = set(symbol_data['date'])
                all_dates = set(df['date'].unique())
                missing_dates = len(all_dates - symbol_dates)
                gap_pct = (missing_dates / len(all_dates)) * 100
            else:
                gap_pct = 100.0
            
            validation_report['data_quality'][symbol] = {
                'rows': int(actual_days),
                'completeness_pct': round(completeness, 2),
                'gap_pct': round(gap_pct, 2),
                'invalid_prices': int(invalid_prices),
                'missing_values': int(missing_values)
            }
            
            # Flag issues
            if completeness < 90:
                validation_report['issues'].append(
                    f"{symbol}: Only {completeness:.1f}% complete ({actual_days}/{expected_trading_days} days)"
                )
            if invalid_prices > 0:
                validation_report['issues'].append(
                    f"{symbol}: {invalid_prices} invalid prices"
                )
            if gap_pct > 10:
                validation_report['issues'].append(
                    f"{symbol}: {gap_pct:.1f}% data gaps"
                )
        
        # Add summary statistics
        quality_values = [q['completeness_pct'] for q in validation_report['data_quality'].values()]
        validation_report['summary'] = {
            'avg_completeness': round(np.mean(quality_values), 2),
            'min_completeness': round(np.min(quality_values), 2),
            'max_completeness': round(np.max(quality_values), 2),
            'stocks_above_95_pct': sum(1 for v in quality_values if v >= 95),
            'stocks_above_90_pct': sum(1 for v in quality_values if v >= 90),
            'total_issues': len(validation_report['issues'])
        }
        
        # Save validation report
        report_file = f"{BacktestConfig.DATA_DIR}/validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2)
        
        logger.info(f"Validation complete:")
        logger.info(f"  Average completeness: {validation_report['summary']['avg_completeness']:.1f}%")
        logger.info(f"  Stocks ≥95% complete: {validation_report['summary']['stocks_above_95_pct']}")
        logger.info(f"  Stocks ≥90% complete: {validation_report['summary']['stocks_above_90_pct']}")
        logger.info(f"  Total issues: {validation_report['summary']['total_issues']}")
        
        if validation_report['issues']:
            logger.warning(f"First 5 issues: {validation_report['issues'][:5]}")
        
        return validation_report
    
    def select_best_stocks(self, metadata: Dict, num_stocks: int = 100) -> List[str]:
        """Select best stocks for backtesting based on data quality and market cap"""
        logger.info(f"Selecting top {num_stocks} stocks for backtesting...")
        
        # Load validation report
        report_file = f"{BacktestConfig.DATA_DIR}/validation_report.json"
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                validation = json.load(f)
        else:
            logger.warning("No validation report found, using all stocks")
            return list(metadata.keys())[:num_stocks]
        
        # Score each stock
        stock_scores = []
        for symbol, meta in metadata.items():
            if symbol not in validation['data_quality']:
                continue
            
            quality = validation['data_quality'][symbol]
            
            # Score based on:
            # 1. Data completeness (0-100)
            # 2. Market cap (higher = better for index tracking)
            # 3. No invalid prices
            
            score = quality['completeness_pct']
            
            if quality['invalid_prices'] > 0:
                score -= 50  # Heavy penalty
            
            if meta['market_cap'] > 200e9:
                score += 20  # Mega cap bonus
            elif meta['market_cap'] > 10e9:
                score += 10  # Large cap bonus
            
            stock_scores.append({
                'symbol': symbol,
                'score': score,
                'market_cap': meta['market_cap'],
                'sector': meta['sector']
            })
        
        # Sort by score
        stock_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top N, ensuring sector diversity
        selected = []
        sector_counts = {}
        
        for stock in stock_scores:
            if len(selected) >= num_stocks:
                break
            
            sector = stock['sector']
            sector_counts[sector] = sector_counts.get(sector, 0)
            
            # Limit per sector to ensure diversification
            max_per_sector = num_stocks // 8  # Roughly 8-12 stocks per sector
            if sector_counts[sector] < max_per_sector or len(selected) > num_stocks * 0.8:
                selected.append(stock['symbol'])
                sector_counts[sector] += 1
        
        logger.info(f"Selected {len(selected)} stocks")
        logger.info(f"Sector distribution: {sector_counts}")
        
        # Save selected list
        selected_file = f"{BacktestConfig.METADATA_DIR}/selected_stocks.json"
        with open(selected_file, 'w') as f:
            json.dump({
                'symbols': selected,
                'count': len(selected),
                'selection_date': datetime.now().isoformat(),
                'sector_distribution': sector_counts
            }, f, indent=2)
        
        return selected
    
    def _load_checkpoint(self, filepath: str) -> Dict:
        """Load checkpoint file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_checkpoint(self, filepath: str, data: Dict):
        """Save checkpoint file"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def clear_checkpoints(self):
        """Clear all checkpoints to start fresh"""
        import shutil
        if os.path.exists(BacktestConfig.CHECKPOINT_DIR):
            shutil.rmtree(BacktestConfig.CHECKPOINT_DIR)
        os.makedirs(BacktestConfig.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(BacktestConfig.TEMP_PRICE_DIR, exist_ok=True)
        logger.info("All checkpoints cleared")
    
    def get_progress(self) -> Dict:
        """Get current download progress"""
        meta_checkpoint = self._load_checkpoint(BacktestConfig.METADATA_CHECKPOINT)
        price_checkpoint = self._load_checkpoint(BacktestConfig.PRICE_CHECKPOINT)
        
        return {
            'metadata': {
                'completed': len(meta_checkpoint.get('completed', [])),
                'failed': len(meta_checkpoint.get('failed', [])),
                'finished': meta_checkpoint.get('finished', False)
            },
            'prices': {
                'completed': len(price_checkpoint.get('completed', [])),
                'failed': len(price_checkpoint.get('failed', [])),
                'finished': price_checkpoint.get('finished', False),
                'last_batch': price_checkpoint.get('batch_completed', 0)
            }
        }
    
    def download_all(self, num_stocks: int = 100):
        """Complete download pipeline"""
        logger.info("="*60)
        logger.info("STARTING COMPLETE DATA DOWNLOAD PIPELINE")
        logger.info("="*60)
        
        # Step 1: Get S&P 500 list
        symbols = self.get_sp500_list()
        if not symbols:
            logger.error("Failed to get S&P 500 list. Aborting.")
            return
        
        # Step 2: Download metadata
        metadata = self.download_metadata_yfinance(symbols)
        
        # Step 3: Download price data
        df = self.download_prices_yfinance(
            symbols,
            BacktestConfig.START_DATE,
            BacktestConfig.END_DATE
        )
        
        if df.empty:
            logger.error("No price data downloaded. Aborting.")
            return
        
        # Step 4: Validate data
        validation = self.validate_data(df)
        
        # Step 5: Select best stocks for backtesting
        selected = self.select_best_stocks(metadata, num_stocks)
        
        # Step 6: Create filtered dataset with only selected stocks
        filtered_df = df[df['symbol'].isin(selected)]
        filtered_file = BacktestConfig.PRICE_DATA_FILE.replace('.parquet', '_filtered.parquet')
        filtered_df.to_parquet(filtered_file, index=False)
        
        logger.info("="*60)
        logger.info("DATA DOWNLOAD COMPLETE!")
        logger.info("="*60)
        logger.info(f"Total stocks downloaded: {df['symbol'].nunique()}")
        logger.info(f"Selected for backtesting: {len(selected)}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Full dataset: {BacktestConfig.PRICE_DATA_FILE}")
        logger.info(f"Filtered dataset: {filtered_file}")
        logger.info(f"Metadata: {BacktestConfig.METADATA_FILE}")
        logger.info("="*60)

# ============================================================================
# DATA LOADER FOR BACKTESTING
# ============================================================================

class BacktestDataLoader:
    """Load and prepare data for backtesting"""
    
    def __init__(self, use_filtered: bool = True):
        self.use_filtered = use_filtered
        self.prices_df: Optional[pd.DataFrame] = None
        self.metadata: Dict = {}
        self.selected_symbols: List[str] = []
    
    def load_data(self):
        """Load all necessary data for backtesting"""
        logger.info("Loading backtest data...")
        
        # Load price data
        if self.use_filtered:
            price_file = BacktestConfig.PRICE_DATA_FILE.replace('.parquet', '_filtered.parquet')
        else:
            price_file = BacktestConfig.PRICE_DATA_FILE
        
        if not os.path.exists(price_file):
            raise FileNotFoundError(
                f"Price data not found at {price_file}. "
                "Run download_all() first."
            )
        
        self.prices_df = pd.read_parquet(price_file)
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        
        # Load metadata
        with open(BacktestConfig.METADATA_FILE, 'r') as f:
            self.metadata = json.load(f)
        
        # Load selected symbols
        selected_file = f"{BacktestConfig.METADATA_DIR}/selected_stocks.json"
        if os.path.exists(selected_file):
            with open(selected_file, 'r') as f:
                data = json.load(f)
                self.selected_symbols = data['symbols']
        
        logger.info(f"Loaded {len(self.prices_df)} price records")
        logger.info(f"Symbols: {self.prices_df['symbol'].nunique()}")
        logger.info(f"Date range: {self.prices_df['date'].min()} to {self.prices_df['date'].max()}")
        
        return self.prices_df
    
    def get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get closing price for a symbol on a specific date"""
        if self.prices_df is None:
            self.load_data()
        
        result = self.prices_df[
            (self.prices_df['symbol'] == symbol) & 
            (self.prices_df['date'] == date)
        ]
        
        if not result.empty:
            return float(result.iloc[0]['close'])
        return None
    
    def get_prices_range(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get price data for a symbol over a date range"""
        if self.prices_df is None:
            self.load_data()
        
        mask = (
            (self.prices_df['symbol'] == symbol) &
            (self.prices_df['date'] >= start_date) &
            (self.prices_df['date'] <= end_date)
        )
        
        return self.prices_df[mask].copy()
    
    def get_all_prices_on_date(self, date: datetime) -> pd.DataFrame:
        """Get prices for all symbols on a specific date"""
        if self.prices_df is None:
            self.load_data()
        
        return self.prices_df[self.prices_df['date'] == date].copy()
    
    def calculate_returns(self, symbol: str, days: int = 90) -> pd.Series:
        """Calculate daily returns for a symbol"""
        if self.prices_df is None:
            self.load_data()
        
        symbol_data = self.prices_df[self.prices_df['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date').tail(days)
        
        returns = symbol_data['close'].pct_change().dropna()
        return returns

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download historical data for backtesting')
    parser.add_argument('--clear', action='store_true', help='Clear checkpoints and start fresh')
    parser.add_argument('--progress', action='store_true', help='Show current progress')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    downloader = DataDownloader()
    
    if args.progress:
        progress = downloader.get_progress()
        print("\n" + "="*60)
        print("DOWNLOAD PROGRESS")
        print("="*60)
        print(f"\nMetadata:")
        print(f"  Completed: {progress['metadata']['completed']}")
        print(f"  Failed: {progress['metadata']['failed']}")
        print(f"  Finished: {progress['metadata']['finished']}")
        print(f"\nPrices:")
        print(f"  Completed: {progress['prices']['completed']}")
        print(f"  Failed: {progress['prices']['failed']}")
        print(f"  Last Batch: {progress['prices']['last_batch']}")
        print(f"  Finished: {progress['prices']['finished']}")
        print("="*60 + "\n")
        return
    
    if args.clear:
        print("\n⚠️  WARNING: This will delete all checkpoints and start over.")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            downloader.clear_checkpoints()
            print("✓ Checkpoints cleared. Run without --clear to start fresh download.\n")
        return
    
    print("="*60)
    print("TAX LOSS HARVESTING - BACKTEST DATA DOWNLOADER")
    print("="*60)
    print()
    
    # Check if resuming
    progress = downloader.get_progress()
    is_resume = (progress['metadata']['completed'] > 0 or 
                 progress['prices']['completed'] > 0)
    
    if is_resume:
        print("🔄 RESUME MODE DETECTED")
        print(f"   Metadata: {progress['metadata']['completed']} completed")
        print(f"   Prices: {progress['prices']['completed']} completed")
        print()
        print("The download will resume from where it left off.")
        print("To start fresh, run with --clear flag first.")
        print()
    else:
        print("This script will download historical data for backtesting:")
        print(f"- S&P 500 constituents list")
        print(f"- Stock metadata (sector, industry, market cap)")
        print(f"- Historical prices from {BacktestConfig.START_DATE} to {BacktestConfig.END_DATE}")
        print()
        print("Data source: Yahoo Finance (free, no API key needed)")
        print(f"Estimated time: 10-20 minutes for 500 stocks")
        print()
        print("✓ Download is RESUMABLE - if interrupted, just run again!")
        print()
    
    if not args.resume and not is_resume:
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Download all data
    try:
        downloader.download_all(num_stocks=100)
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted!")
        print("Don't worry - progress has been saved.")
        print("Run the script again to resume from where you left off.")
        print()
        progress = downloader.get_progress()
        print(f"Progress saved:")
        print(f"  Metadata: {progress['metadata']['completed']} stocks")
        print(f"  Prices: {progress['prices']['completed']} stocks")
        return
    except Exception as e:
        logger.error(f"Error during download: {e}")
        print("\n\n⚠️  Download failed with error!")
        print("Progress has been saved. Run the script again to resume.")
        print(f"Error: {e}")
        return
    
    print()
    print("="*60)
    print("DOWNLOAD COMPLETE!")
    print("="*60)
    print()
    print("Next steps:")
    print("1. Review the data in the 'backtest_data' directory")
    print("2. Check validation_report.json for data quality issues")
    print("3. Run the backtesting script to test your strategy")
    print()

if __name__ == '__main__':
    main()
