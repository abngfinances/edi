# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.17.0",
#     "pandas>=2.3.3",
#     "pyarrow>=22.0.0",
#     "plotly>=5.24.0",
#     "yfinance>=0.2.66",
#     "tqdm==4.67.1",
#     "requests>=2.31.0",
# ]
# ///

"""
Interactive Stock Price Data Downloader & Visualizer

To run this notebook in sandbox mode:
    /home/nikhil/code/edi/.venv/bin/python -m marimo edit --sandbox notebooks/download_and_visualize.py

Or from the notebooks directory:
    ../.venv/bin/python -m marimo edit --sandbox download_and_visualize.py
"""

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    """Setup: Import dependencies and configure paths"""
    import sys
    import os
    import json
    import logging
    from pathlib import Path
    from datetime import datetime, date

    # Add project root to path so imports work
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    # Import EDI components
    from backtesting.index_downloader import IndexDownloader
    from backtesting.metadata_downloader import MetadataDownloader
    from backtesting.price_data_source import PriceDownloader

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set up paths
    BACKTEST_DATA_DIR = Path(project_root) / "backtest_data"
    METADATA_DIR = BACKTEST_DATA_DIR / "metadata"

    # Create directories if they don't exist
    BACKTEST_DATA_DIR.mkdir(exist_ok=True)
    METADATA_DIR.mkdir(exist_ok=True)

    mo.md(f"""
    # 📊 EDI Stock Price Data Downloader & Visualizer

    **Project root:** `{project_root}`  
    **Data directory:** `{BACKTEST_DATA_DIR}`  
    **Metadata directory:** `{METADATA_DIR}`

    This notebook allows you to:
    - Download index constituents from Alpha Vantage
    - Download metadata for each symbol (yfinance)
    - Download historical price data (yfinance)
    - Visualize prices, splits, and dividends interactively
    """)
    return IndexDownloader, METADATA_DIR, mo


@app.cell
def _(mo):
    """Step 2: Download Index Constituents"""
    mo.md("""
    ## Step 1: Download Index Constituents

    Download the list of stocks in an index/ETF from Alpha Vantage.
    Common indexes: SPY (S&P 500), QQQ (NASDAQ-100), DIA (Dow Jones)

    Get your free API key from: https://www.alphavantage.co/support/#api-key
    """)
    return


@app.cell
def _(mo):
    """UI form for index download"""
    index_download_form = mo.ui.form(
        mo.ui.dictionary({
            "index_symbol": mo.ui.text(
                value="SPY",
                label="Index/ETF Symbol:",
                placeholder="e.g., SPY, QQQ, DIA"
            ),
            "api_key": mo.ui.text(
                value="DUJRZATXXEQL2M9R",  # Test API key from integration tests
                label="Alpha Vantage API Key:",
                placeholder="Enter your Alpha Vantage API key",
                kind="password"
            ),
            "expected_count": mo.ui.number(
                value=479,
                label="Expected constituent count (for validation):",
                start=1,
                stop=1000,
                step=1
            )
        }),
        label="📥 Download Constituents",
        bordered=True,
        submit_button_label="Download"
    )
    
    index_download_form
    return (index_download_form,)


@app.cell
def _(IndexDownloader, METADATA_DIR, index_download_form, mo):
    """Process form submission and display result"""
    index_download_result = None
    index_download_error = None

    if index_download_form.value is not None:  # Form has been submitted
        form_data = index_download_form.value
        
        # Validate inputs
        if not form_data["api_key"]:
            index_download_error = "❌ Please enter your Alpha Vantage API key"
        elif not form_data["index_symbol"]:
            index_download_error = "❌ Please enter an index symbol"
        else:
            try:
                with mo.status.spinner(title=f"Downloading {form_data['index_symbol']} constituents..."):
                    downloader = IndexDownloader(
                        api_key=form_data["api_key"],
                        index_symbol=form_data["index_symbol"].upper()
                    )

                    index_download_result = downloader.download_and_save(
                        output_dir=str(METADATA_DIR),
                        expected_count=form_data["expected_count"]
                    )

            except Exception as e:
                import traceback
                index_download_error = f"❌ Download failed: {str(e)}\n\n```\n{traceback.format_exc()}\n```"

    # Display result
    if index_download_error:
        mo.callout(mo.md(index_download_error), kind="danger")
    elif index_download_result:
        mo.callout(
            mo.md(f"""
            ✅ **Successfully downloaded {index_download_result['index_symbol']} constituents**

            - **Total symbols:** {index_download_result['total_symbols']}
            - **Expected:** {index_download_result['expected_symbols']}
            - **Output file:** `{index_download_result['output_path']}`
            - **Timestamp:** {index_download_result['download_timestamp']}
            """),
            kind="success"
        )
    else:
        mo.md("*Fill out the form and click Download to get started*")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
