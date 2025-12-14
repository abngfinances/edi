"""
Unified configuration for Tax Loss Harvesting Strategy
Used by all execution modes: backtest, paper trade, and live trade
"""

import os


class TLHConfig:
    """Unified configuration for TLH strategy across all execution modes"""
    
    # Trading Parameters
    INITIAL_CAPITAL = 10000.0
    NUM_STOCKS = 100
    HARVEST_THRESHOLD = 0.005  # 0.5% loss threshold
    CORRELATION_LOOKBACK_DAYS = 90
    
    # Schedule
    TRADING_HOUR = 14  # 2 PM ET
    HARVEST_FREQUENCY = 'weekly'  # 'daily', 'weekly', 'monthly'
    
    # Wash Sale
    WASH_SALE_DAYS = 30
    
    # Tax assumptions (for backtesting)
    SHORT_TERM_TAX_RATE = 0.37  # Federal + state for high earners
    LONG_TERM_TAX_RATE = 0.238  # 20% federal + 3.8% NIIT
    
    # Transaction costs (for backtesting)
    COMMISSION_PER_TRADE = 0.0
    SPREAD_BPS = 1  # 1 basis point for spread cost
    
    # API Keys (for paper/live trading)
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    FMP_API_KEY = os.getenv('FMP_API_KEY', '')
    
    # File paths
    DATA_DIR = 'data'
    POSITIONS_FILE = f'{DATA_DIR}/positions.json'
    TRANSACTIONS_FILE = f'{DATA_DIR}/transactions.json'
    WASH_SALES_FILE = f'{DATA_DIR}/wash_sales.json'
    METADATA_FILE = f'{DATA_DIR}/sp500_metadata.json'
    
    # Backtest-specific paths
    RESULTS_DIR = 'backtest_results'
    
    # Logging
    LOG_FILE = f'{DATA_DIR}/tlh_system.log'
    LOG_LEVEL = 'INFO'

