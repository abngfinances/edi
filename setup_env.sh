#!/bin/bash

# --- Environment Variables for Alpaca and Financial Modeling Prep (FMP) ---
# execute using source setup_env.sh

# The 'export' command makes the variables available to subprocesses.

# Alpaca API Key (for account identification)
export ALPACA_API_KEY="PKKVD2SIZT455SNYHFGIL2LQQA"

# Alpaca Secret Key (for signing requests)
export ALPACA_SECRET_KEY="MyHKc3nJPBTxQMUBJV7CFcBWnco6bPo5SuYqvBQDeY2"

# Alpaca Base URL (e.g., 'https://paper-api.alpaca.markets' for paper trading or 'https://api.alpaca.markets' for live trading)
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"

# Financial Modeling Prep (FMP) API Key
export FMP_API_KEY="oMMCtuznflx1p1NEQmm4e8QJSeWnARdk"

# Index symbol for backtesting and data download
export INDEX_SYMBOL="SPY"

echo "Environment variables set for the current shell session."
echo "ALPACA_API_KEY: $ALPACA_API_KEY"
echo "ALPACA_BASE_URL: $ALPACA_BASE_URL"
echo "FMP_API_KEY: $FMP_API_KEY"
echo "INDEX_SYMBOL: $INDEX_SYMBOL"
# Note: ALPACA_SECRET_KEY is intentionally not echoed for security reasons.
#
# Alpaca 2FA recovery
# d3d67141-6fdf-4758-850b-73b786f82087
#
