"""
Backtest Execution Engine

Implements ExecutionEngine interface for backtesting using historical data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy.execution import ExecutionEngine
from strategy.models import Position, StockMetadata
from backtesting.backtest_data_downloader import BacktestDataLoader


class BacktestPosition:
    """Position during backtest (compatible with Position interface)"""
    def __init__(self, symbol: str, quantity: float, cost_basis: float, purchase_date: datetime):
        self.symbol = symbol
        self.quantity = quantity
        self.cost_basis = cost_basis
        self.purchase_date = purchase_date
    
    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > 0 else 0


class BacktestExecutionEngine(ExecutionEngine):
    """Execution engine for backtesting using historical data"""
    
    def __init__(self, data_loader: BacktestDataLoader, initial_cash: float,
                 commission_per_trade: float = 0.0, spread_bps: int = 1,
                 logger=None):
        """
        Initialize backtest execution engine.
        
        Args:
            data_loader: BacktestDataLoader instance
            initial_cash: Starting cash amount
            commission_per_trade: Commission per trade
            spread_bps: Spread cost in basis points
            logger: Optional logger
        """
        self.data_loader = data_loader
        self.cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.spread_bps = spread_bps
        self.logger = logger
        
        # Internal state
        self.positions: Dict[str, BacktestPosition] = {}
        self.current_date: Optional[datetime] = None
    
    def set_current_date(self, date: datetime):
        """Set the current date for backtesting"""
        self.current_date = date
    
    def get_price(self, symbol: str, date: Optional[datetime] = None) -> Optional[float]:
        """Get price for a symbol on a date"""
        if date is None:
            date = self.current_date
        if date is None:
            raise ValueError("Date must be provided or set via set_current_date()")
        
        price = self.data_loader.get_price(symbol, date)
        
        # Handle NaN or invalid prices
        if price is None or pd.isna(price) or price <= 0:
            # Try to get price from nearby dates (up to 5 days before)
            for days_back in range(1, 6):
                alt_date = date - timedelta(days=days_back)
                alt_price = self.data_loader.get_price(symbol, alt_date)
                if alt_price and not pd.isna(alt_price) and alt_price > 0:
                    if self.logger:
                        self.logger.warning(f"Using price from {days_back} days before for {symbol} on {date.date()}")
                    return alt_price
            
            if self.logger:
                self.logger.warning(f"Invalid price for {symbol} on {date}")
            return None
        
        return price
    
    def get_prices_range(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get historical prices for a symbol over a date range"""
        return self.data_loader.get_prices_range(symbol, start_date, end_date)
    
    def calculate_transaction_cost(self, quantity: float, price: float) -> tuple:
        """Calculate transaction costs"""
        commission = self.commission_per_trade
        spread_cost = (quantity * price) * (self.spread_bps / 10000)
        return commission, spread_cost
    
    def execute_buy(self, symbol: str, quantity: float, price: float,
                   date: Optional[datetime] = None) -> bool:
        """Execute a buy order"""
        if date is None:
            date = self.current_date
        if date is None:
            raise ValueError("Date must be provided or set via set_current_date()")
        
        if not price or price <= 0:
            if self.logger:
                self.logger.warning(f"Invalid price for {symbol} on {date}")
            return False
        
        amount = quantity * price
        commission, spread = self.calculate_transaction_cost(quantity, price)
        total_cost = amount + commission + spread
        
        if self.cash < total_cost:
            if self.logger:
                self.logger.warning(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Update position
        if symbol in self.positions:
            # Add to existing position (average cost basis)
            pos = self.positions[symbol]
            new_qty = pos.quantity + quantity
            new_cost = pos.cost_basis + amount
            pos.quantity = new_qty
            pos.cost_basis = new_cost
        else:
            self.positions[symbol] = BacktestPosition(
                symbol=symbol,
                quantity=quantity,
                cost_basis=amount,
                purchase_date=date
            )
        
        self.cash -= total_cost
        return True
    
    def execute_sell(self, symbol: str, quantity: float, price: float,
                    date: Optional[datetime] = None) -> Optional[float]:
        """Execute a sell order, returns realized P&L"""
        if date is None:
            date = self.current_date
        if date is None:
            raise ValueError("Date must be provided or set via set_current_date()")
        
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        if pos.quantity < quantity:
            if self.logger:
                self.logger.warning(f"Insufficient quantity: have {pos.quantity}, need {quantity}")
            return None
        
        if not price or price <= 0:
            if self.logger:
                self.logger.warning(f"Invalid price for {symbol} on {date}")
            return None
        
        proceeds = quantity * price
        commission, spread = self.calculate_transaction_cost(quantity, price)
        net_proceeds = proceeds - commission - spread
        
        # Calculate realized P&L (proportional to quantity sold)
        cost_basis_sold = (quantity / pos.quantity) * pos.cost_basis
        realized_pnl = net_proceeds - cost_basis_sold
        
        self.cash += net_proceeds
        
        # Update or remove position
        if quantity >= pos.quantity:
            # Selling entire position
            del self.positions[symbol]
        else:
            # Partial sale
            pos.quantity -= quantity
            pos.cost_basis -= cost_basis_sold
        
        return realized_pnl
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        result = {}
        for symbol, pos in self.positions.items():
            result[symbol] = Position(
                symbol=pos.symbol,
                quantity=pos.quantity,
                cost_basis=pos.cost_basis,
                purchase_date=pos.purchase_date.isoformat()
            )
        return result
    
    def get_metadata(self, symbols: List[str]) -> Dict[str, StockMetadata]:
        """Get stock metadata"""
        metadata = {}
        for symbol in symbols:
            if symbol in self.data_loader.metadata:
                meta = self.data_loader.metadata[symbol]
                metadata[symbol] = StockMetadata(
                    symbol=symbol,
                    name=meta.get('name', symbol),
                    sector=meta.get('sector', 'Unknown'),
                    industry=meta.get('industry', 'Unknown'),
                    market_cap=meta.get('market_cap', 0)
                )
        return metadata
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols for trading"""
        return self.data_loader.selected_symbols if hasattr(self.data_loader, 'selected_symbols') else []

