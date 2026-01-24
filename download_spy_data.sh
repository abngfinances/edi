#!/bin/bash

# Download SPY Data Pipeline
# Downloads constituents, metadata, and 5 years of historical prices for SPY

set -e  # Exit on error

# Configuration
INDEX_SYMBOL="SPY"
OUTPUT_DIR="backtest_data"
METADATA_DIR="${OUTPUT_DIR}/metadata"

# Calculate date range (last 5 years)
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "5 years ago" +%Y-%m-%d)

echo "=========================================="
echo "SPY Data Download Pipeline"
echo "=========================================="
echo "Index: ${INDEX_SYMBOL}"
echo "Date range: ${START_DATE} to ${END_DATE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# Step 1: Download SPY constituents
echo "Step 1/3: Downloading ${INDEX_SYMBOL} constituents..."
python backtesting/index_downloader.py ${INDEX_SYMBOL} --output-dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download constituents"
    exit 1
fi
echo "✓ Constituents downloaded successfully"
echo ""

# Step 2: Download metadata for constituents
echo "Step 2/3: Downloading metadata for constituents..."
python backtesting/metadata_downloader.py ${INDEX_SYMBOL} --output-dir ${OUTPUT_DIR}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download metadata"
    echo "If some symbols failed, you may need to use --ignore-symbols"
    exit 1
fi
echo "✓ Metadata downloaded successfully"
echo ""

# Step 3: Download historical prices
echo "Step 3/3: Downloading historical prices (${START_DATE} to ${END_DATE})..."
python backtesting/price_data_downloader.py ${INDEX_SYMBOL} \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --output-dir ${OUTPUT_DIR} \
    --metadata-dir ${METADATA_DIR} \
    --batch-size 10 \
    --max-workers 5 \
    --rate-limit-delay 2.0

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to download price data"
    exit 1
fi
echo "✓ Price data downloaded successfully"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Data saved to: ${OUTPUT_DIR}"
echo "To view the data, run:"
echo "  python -m marimo edit notebooks/view_data.py"
echo "=========================================="
