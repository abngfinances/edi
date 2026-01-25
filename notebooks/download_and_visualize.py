# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo>=0.17.0",
#     "pandas>=2.3.3",
#     "pyarrow>=22.0.0",
#     "plotly>=5.24.0",
# ]
# ///

"""
Interactive Stock Price Data Viewer

View downloaded price data for index constituents with interactive plots and tables.

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
    from pathlib import Path

    # Add project root to path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import marimo as mo
    import pandas as pd
    import plotly.graph_objects as go

    # Configure paths
    METADATA_DIR = Path(project_root) / "backtest_data" / "metadata"
    PRICES_DIR = Path(project_root) / "backtest_data" / "prices"
    return METADATA_DIR, PRICES_DIR, go, json, mo, pd


@app.cell
def _(mo):
    """Configuration"""
    mo.md("""
    # Stock Price Data Viewer
    """)
    return


@app.cell
def _():
    """Set index symbol"""
    # Change this variable to view different index constituents
    index_symbol = "TEST"  # Options: SPY, TEST, etc.
    return (index_symbol,)


@app.cell
def _(METADATA_DIR, index_symbol, json, mo):
    """Load constituents from metadata file"""
    constituents_file = METADATA_DIR / f"{index_symbol.lower()}_constituents.json"

    if not constituents_file.exists():
        constituents = []
        error_message = f"Constituents file not found: {constituents_file}"
    else:
        try:
            with open(constituents_file, 'r') as constituents_f:
                data = json.load(constituents_f)
            constituents = data.get('symbols', [])
            error_message = None
        except Exception as e:
            constituents = []
            error_message = f"Error loading constituents: {e}"

    if error_message:
        _status = mo.md(f"**Error:** {error_message}")
    else:
        _status = mo.md(f"**Loaded {len(constituents)} constituents**")
    
    _status
    return (constituents,)


@app.cell
def _(constituents, mo):
    """Symbol Selection Dropdown"""
    symbol_dropdown = mo.ui.dropdown(
        options=constituents if constituents else ["No symbols available"],
        value=constituents[0] if constituents else None,
        label="Select Symbol:"
    )
    symbol_dropdown
    return (symbol_dropdown,)


@app.cell
def _(PRICES_DIR, json, pd, symbol_dropdown):
    """Load price data for selected symbol"""
    selected_symbol = symbol_dropdown.value

    if not selected_symbol or selected_symbol == "No symbols available":
        prices_df = None
        metadata = None
        load_error = "No symbol selected"
    else:
        # Load prices from parquet file
        price_file = PRICES_DIR / selected_symbol / "yfinance_1d" / "prices.parquet"
        metadata_file = PRICES_DIR / selected_symbol / "yfinance_1d" / "metadata.json"

        if not price_file.exists():
            prices_df = None
            metadata = None
            load_error = f"Price data not found: {price_file}"
        else:
            try:
                prices_df = pd.read_parquet(price_file)

                # Load metadata
                if metadata_file.exists():
                    with open(metadata_file, 'r') as metadata_f:
                        metadata = json.load(metadata_f)
                else:
                    metadata = None

                load_error = None
            except Exception as e:
                prices_df = None
                metadata = None
                load_error = f"Error loading data: {e}"
    return load_error, metadata, prices_df, selected_symbol


@app.cell
def _(load_error, metadata, mo, prices_df, selected_symbol):
    """Display symbol info and data summary"""
    if load_error:
        _summary = mo.md(f"**Error:** {load_error}")
    elif prices_df is not None:
        start_date = prices_df.index.min().strftime('%Y-%m-%d')
        end_date = prices_df.index.max().strftime('%Y-%m-%d')
        total_days = len(prices_df)

        _summary = mo.md(f"""
        ## {selected_symbol} Price Data

        - **Date Range:** {start_date} to {end_date}
        - **Total Trading Days:** {total_days:,}
        - **Splits:** {len(metadata.get('splits', {})) if metadata else 0}
        - **Dividends:** {len(metadata.get('dividends', {})) if metadata else 0}
        """)
    else:
        _summary = None
    
    _summary
    return


@app.cell
def _(go, prices_df, selected_symbol):
    """Plot price chart"""
    _fig_output = None
    if prices_df is not None and len(prices_df) > 0:
        _fig = go.Figure()

        # Add closing price line
        _fig.add_trace(go.Scatter(
            x=prices_df.index,
            y=prices_df['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#1f77b4', width=2)
        ))

        # Add volume bar chart on secondary y-axis
        _fig.add_trace(go.Bar(
            x=prices_df.index,
            y=prices_df['Volume'],
            name='Volume',
            yaxis='y2',
            marker=dict(color='rgba(128, 128, 128, 0.3)')
        ))

        # Update layout
        _fig.update_layout(
            title=f"{selected_symbol} - Closing Price & Volume",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            yaxis2=dict(
                title="Volume",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500,
            showlegend=True
        )
        _fig_output = _fig

    _fig_output
    return


@app.cell
def _(mo, prices_df):
    """Display price data table"""
    _table_output = None
    if prices_df is not None and len(prices_df) > 0:
        # Format the dataframe for display
        _display_df = prices_df.copy()
        _display_df.index = _display_df.index.strftime('%Y-%m-%d')

        # Round numeric columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in _display_df.columns:
                _display_df[col] = _display_df[col].round(2)

        # Format volume with commas
        if 'Volume' in _display_df.columns:
            _display_df['Volume'] = _display_df['Volume'].apply(lambda x: f"{int(x):,}")

        _table_output = mo.vstack([
            mo.md("### Price Data (OHLCV)"),
            mo.ui.table(_display_df, selection=None)
        ])

    _table_output
    return


@app.cell
def _(metadata, mo, pd):
    """Display splits table"""
    if metadata and metadata.get('splits'):
        _splits_data = [
            {'Date': date, 'Split Ratio': ratio}
            for date, ratio in metadata['splits'].items()
        ]
        _splits_df = pd.DataFrame(_splits_data).sort_values('Date', ascending=False)

        _splits_output = mo.vstack([
            mo.md("### Stock Splits"),
            mo.ui.table(_splits_df, selection=None)
        ])
    else:
        _splits_output = mo.md("### Stock Splits\nNo splits in this period.")

    _splits_output
    return


@app.cell
def _(metadata, mo, pd):
    """Display dividends table"""
    if metadata and metadata.get('dividends'):
        _dividends_data = [
            {'Date': date, 'Amount ($)': f"{amount:.4f}"}
            for date, amount in metadata['dividends'].items()
        ]
        _dividends_df = pd.DataFrame(_dividends_data).sort_values('Date', ascending=False)

        _dividends_output = mo.vstack([
            mo.md("### Dividends"),
            mo.ui.table(_dividends_df, selection=None)
        ])
    else:
        _dividends_output = mo.md("### Dividends\nNo dividends in this period.")

    _dividends_output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
