#!/bin/bash

# Download Index Constituents (Step 1)
# Downloads the list of stocks in an index/ETF from Alpha Vantage

set -e  # Exit on error

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/.venv/bin/activate"

# Configuration
INDEX_SYMBOL="SPY"
OUTPUT_DIR="backtest_data/metadata"
EXPECTED_COUNT=479  # Expected number of SPY constituents
API_KEY="${ALPHA_VANTAGE_API_KEY:-DUJRZATXXEQL2M9R}"  # Use env var or test key

echo "=========================================="
echo "Step 1: Download Index Constituents"
echo "=========================================="
echo "Index: ${INDEX_SYMBOL}"
echo "Expected count: ${EXPECTED_COUNT}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

python backtesting/index_downloader.py ${INDEX_SYMBOL} ${API_KEY} ${EXPECTED_COUNT} --output-dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download constituents"
    exit 1
fi

echo ""
echo "✓ Constituents downloaded successfully"
echo ""
echo "Next step: Run ./download_metadata.sh"
