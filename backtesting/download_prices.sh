#!/bin/bash

# Download Historical Prices (Step 3)
# Downloads historical price data for all constituent symbols

set -e  # Exit on error

# Activate virtual environment
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
source "${PROJECT_ROOT}/.venv/bin/activate"

# Set Python path to project root
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Configuration
INDEX_SYMBOL="SPY"
OUTPUT_DIR="backtest_data/prices"
METADATA_DIR="backtest_data/metadata"

# Date range (last 5 years by default)
END_DATE=$(date -d "yesterday" +%Y-%m-%d)
START_DATE=$(date -d "5 years ago" +%Y-%m-%d)

# Download settings
BATCH_SIZE=10
MAX_WORKERS=5
RATE_LIMIT_DELAY=2.0
IGNORE_SYMBOLS="FISV,n/a"  # Edit this to add/remove problematic symbols

echo "=========================================="
echo "Step 3: Download Historical Prices"
echo "=========================================="
echo "Index: ${INDEX_SYMBOL}"
echo "Date range: ${START_DATE} to ${END_DATE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Metadata directory: ${METADATA_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max workers: ${MAX_WORKERS}"
echo "Rate limit delay: ${RATE_LIMIT_DELAY}s"
echo "Ignoring symbols: ${IGNORE_SYMBOLS}"
echo "=========================================="
echo ""

python backtesting/price_data_downloader.py ${INDEX_SYMBOL} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --output-dir ${OUTPUT_DIR} \
    --metadata-dir ${METADATA_DIR} \
    --batch-size ${BATCH_SIZE} \
    --max-workers ${MAX_WORKERS} \
    --rate-limit-delay ${RATE_LIMIT_DELAY} \
    --ignore-symbols ${IGNORE_SYMBOLS}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download price data"
    echo "If symbols failed, edit IGNORE_SYMBOLS in this script and re-run"
    exit 1
fi

echo ""
echo "✓ Price data downloaded successfully"
echo ""
echo "=========================================="
echo "Data pipeline completed!"
echo "=========================================="
echo "Data saved to: ${OUTPUT_DIR}"
echo "To view the data, run:"
echo "  .venv/bin/python -m marimo edit --sandbox notebooks/view_data.py"
echo "=========================================="
