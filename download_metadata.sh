#!/bin/bash

# Download Stock Metadata (Step 2)
# Downloads detailed metadata for each constituent symbol

set -e  # Exit on error

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/.venv/bin/activate"

# Configuration
INDEX_SYMBOL="SPY"
OUTPUT_DIR="backtest_data/metadata"
IGNORE_SYMBOLS="FISV,n/a"  # Edit this to add/remove problematic symbols
RATE_LIMIT_DELAY=0.5  # Delay between symbol downloads (seconds)

echo "=========================================="
echo "Step 2: Download Stock Metadata"
echo "=========================================="
echo "Index: ${INDEX_SYMBOL}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Ignoring symbols: ${IGNORE_SYMBOLS}"
echo "Rate limit delay: ${RATE_LIMIT_DELAY}s"
echo "=========================================="
echo ""

python backtesting/metadata_downloader.py ${INDEX_SYMBOL} \
    --output-dir ${OUTPUT_DIR} \
    --ignore-symbols ${IGNORE_SYMBOLS} \
    --rate-limit-delay ${RATE_LIMIT_DELAY}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download metadata"
    echo "If symbols failed, edit IGNORE_SYMBOLS in this script and re-run"
    exit 1
fi

echo ""
echo "✓ Metadata downloaded successfully"
echo ""
echo "Next step: Run ./download_prices.sh"
