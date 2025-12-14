"""
Execution Engine Interface

Defines the interface that all execution engines (backtest, paper, live) must implement.
This ensures the strategy can work with any execution mode.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class ExecutionEngine(ABC):
    """Abstract base class for execution engines"""
    
    @abstractmethod
    def get_price(self, symbol: str, date: Optional[datetime] = None) -> Optional[float]:
        """
        Get current/latest price for a symbol.
        
        Args:
            symbol: Stock symbol
            date: Optional date (for backtesting). If None, gets latest price.
        
        Returns:
            Price as float, or None if unavailable
        """
        pass
    
    @abstractmethod
    def get_prices_range(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """
        Get historical prices for a symbol over a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        pass
    
    @abstractmethod
    def execute_buy(self, symbol: str, quantity: float, price: float, 
                    date: Optional[datetime] = None) -> bool:
        """
        Execute a buy order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            date: Optional date (for backtesting)
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute_sell(self, symbol: str, quantity: float, price: float,
                    date: Optional[datetime] = None) -> Optional[float]:
        """
        Execute a sell order.
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share
            date: Optional date (for backtesting)
        
        Returns:
            Realized P&L (positive for gain, negative for loss), or None if failed
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> Dict[str, any]:
        """
        Get current positions.
        
        Returns:
            Dictionary mapping symbol to Position object
        """
        pass
    
    @abstractmethod
    def get_metadata(self, symbols: List[str]) -> Dict[str, any]:
        """
        Get stock metadata.
        
        Args:
            symbols: List of stock symbols
        
        Returns:
            Dictionary mapping symbol to StockMetadata object
        """
        pass
    
    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols for trading.
        
        Returns:
            List of stock symbols
        """
        pass

