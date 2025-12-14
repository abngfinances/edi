"""
Core Tax Loss Harvesting Strategy

Pure business logic with no I/O, API calls, or side effects.
This strategy is used by all execution modes (backtest, paper, live).
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


class TLHStrategy:
    """
    Stateless tax loss harvesting strategy.
    All methods are pure functions that take inputs and return decisions.
    """
    
    @staticmethod
    def get_market_cap_tier(market_cap: float) -> str:
        """Get market cap tier for a stock"""
        if market_cap > 200e9:
            return "Mega"
        elif market_cap > 10e9:
            return "Large"
        return "Mid"
    
    @staticmethod
    def select_stocks(sp500_list: List[str], num_stocks: int, 
                     metadata: Dict[str, Dict]) -> List[str]:
        """
        Select stocks from S&P 500 for portfolio.
        
        Strategy:
        1. Top 50 by market cap
        2. Distribute remaining by sector (up to 5 per sector)
        
        Args:
            sp500_list: List of S&P 500 symbols
            num_stocks: Target number of stocks to select
            metadata: Dictionary mapping symbol to metadata dict with:
                - market_cap: float
                - sector: str
        
        Returns:
            List of selected symbols
        """
        # Build DataFrame with available metadata
        stock_data = []
        for symbol in sp500_list:
            if symbol in metadata:
                meta = metadata[symbol]
                stock_data.append({
                    'symbol': symbol,
                    'market_cap': meta.get('market_cap', 0),
                    'sector': meta.get('sector', 'Unknown')
                })
        
        if not stock_data:
            return sp500_list[:num_stocks]
        
        df = pd.DataFrame(stock_data).sort_values('market_cap', ascending=False)
        
        # Top 50 by market cap
        selected = df.head(50)['symbol'].tolist()
        
        # Distribute remaining by sector
        remaining_df = df[~df['symbol'].isin(selected)]
        for sector in remaining_df['sector'].unique():
            sector_df = remaining_df[remaining_df['sector'] == sector]
            count = min(5, len(sector_df))
            selected.extend(sector_df.head(count)['symbol'].tolist())
        
        return selected[:num_stocks]
    
    @staticmethod
    def should_harvest(cost_basis: float, current_value: float, 
                      threshold: float) -> Tuple[bool, float]:
        """
        Determine if a position should be harvested.
        
        Args:
            cost_basis: Original cost basis
            current_value: Current market value
            threshold: Loss threshold (e.g., 0.005 for 0.5%)
        
        Returns:
            Tuple of (should_harvest: bool, pnl_pct: float)
        """
        if cost_basis <= 0:
            return False, 0.0
        
        pnl = current_value - cost_basis
        pnl_pct = pnl / cost_basis
        
        return pnl_pct < -threshold, pnl_pct
    
    @staticmethod
    def is_wash_sale_active(symbol: str, sale_date: datetime, 
                           check_date: datetime, wash_sale_days: int) -> bool:
        """
        Check if a symbol is in wash sale period.
        
        Args:
            symbol: Stock symbol
            sale_date: Date of sale
            check_date: Date to check
            wash_sale_days: Number of days for wash sale restriction
        
        Returns:
            True if in wash sale period, False otherwise
        """
        days_since = (check_date - sale_date).days
        return 0 <= days_since < wash_sale_days
    
    @staticmethod
    def calculate_correlations(target: str, candidates: List[str],
                              price_data_func, lookback_days: int,
                              current_date: Optional[datetime] = None) -> pd.Series:
        """
        Calculate correlations between target and candidate stocks.
        
        Args:
            target: Target stock symbol
            candidates: List of candidate symbols
            price_data_func: Function that takes (symbol, start_date, end_date) and returns
                           DataFrame with 'close' column
            lookback_days: Number of days to look back for correlation
            current_date: Current date (for backtesting). If None, uses latest data.
        
        Returns:
            Series of correlations, sorted descending
        """
        if current_date is None:
            # For live/paper trading, use current date
            current_date = datetime.now()
        
        end_date = current_date
        start_date = current_date - timedelta(days=lookback_days)
        
        # Get returns for all symbols
        returns_dict = {}
        
        for symbol in [target] + candidates:
            try:
                prices = price_data_func(symbol, start_date, end_date)
                if len(prices) > 20:  # Need minimum data
                    returns = prices['close'].pct_change().dropna()
                    # Remove any NaN or inf values
                    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                    if len(returns) > 20:  # Still have enough after cleaning
                        returns_dict[symbol] = returns
            except Exception:
                continue
        
        if target not in returns_dict or len(returns_dict) < 2:
            return pd.Series()
        
        # Align returns on dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(how='any')  # Only use dates with all data
        
        if len(returns_df) < 20:
            return pd.Series()
        
        if target not in returns_df.columns:
            return pd.Series()
        
        # Calculate correlation
        try:
            corr = returns_df.corr()[target]
            corr = corr.drop(target)
            
            # Remove NaN correlations
            corr = corr.dropna()
            
            return corr.sort_values(ascending=False)
        except Exception:
            return pd.Series()
    
    @staticmethod
    def find_replacement(sold_symbol: str, available_symbols: List[str],
                        current_positions: List[str],
                        wash_sale_symbols: List[str],
                        metadata: Dict[str, Dict],
                        price_data_func,
                        correlation_lookback_days: int,
                        current_date: Optional[datetime] = None,
                        logger=None) -> Optional[str]:
        """
        Find replacement stock for tax loss harvesting.
        
        Strategy:
        1. Filter by sector and market cap tier (prefer same sector + tier)
        2. If no matches, filter by market cap tier only
        3. If still no matches, use all available
        4. Calculate correlations and select highest
        5. Fallback to market cap ranking if correlations fail
        
        Args:
            sold_symbol: Symbol being sold
            available_symbols: All available symbols for selection
            current_positions: Symbols currently in portfolio
            wash_sale_symbols: Symbols in wash sale period
            metadata: Dictionary mapping symbol to metadata dict
            price_data_func: Function to get price data (symbol, start, end) -> DataFrame
            correlation_lookback_days: Days to look back for correlation
            current_date: Current date (for backtesting)
            logger: Optional logger for warnings
        
        Returns:
            Replacement symbol, or None if no suitable replacement
        """
        if sold_symbol not in metadata:
            return None
        
        sold_meta = metadata[sold_symbol]
        sold_tier = TLHStrategy.get_market_cap_tier(sold_meta.get('market_cap', 0))
        
        # Filter available symbols (not in positions, not in wash sale, not the sold symbol)
        candidates = [
            sym for sym in available_symbols
            if sym != sold_symbol
            and sym not in current_positions
            and sym not in wash_sale_symbols
            and sym in metadata
        ]
        
        if not candidates:
            if logger:
                logger.warning(f"No replacement candidates for {sold_symbol}")
            return None
        
        # Filter by sector and tier (preferred)
        same_sector = [
            sym for sym in candidates
            if metadata[sym].get('sector') == sold_meta.get('sector')
            and TLHStrategy.get_market_cap_tier(metadata[sym].get('market_cap', 0)) == sold_tier
        ]
        
        # If no same-sector matches, use same tier
        if not same_sector:
            same_sector = [
                sym for sym in candidates
                if TLHStrategy.get_market_cap_tier(metadata[sym].get('market_cap', 0)) == sold_tier
            ]
        
        # If still none, use all available
        if not same_sector:
            same_sector = candidates
        
        # Calculate correlations
        correlations = TLHStrategy.calculate_correlations(
            sold_symbol, same_sector, price_data_func,
            correlation_lookback_days, current_date
        )
        
        if not correlations.empty:
            return correlations.idxmax()
        
        # Fallback: select by market cap if correlations fail
        if logger:
            logger.warning(f"Correlation calculation failed for {sold_symbol}, using market cap ranking")
        
        # Sort candidates by market cap
        candidates_with_mcap = [
            (sym, metadata[sym].get('market_cap', 0))
            for sym in same_sector
            if sym in metadata
        ]
        candidates_with_mcap.sort(key=lambda x: x[1], reverse=True)
        
        if candidates_with_mcap:
            return candidates_with_mcap[0][0]
        
        # Last resort: first available symbol
        return same_sector[0] if same_sector else None

