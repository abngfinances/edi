"""
Common data models for TLH strategy
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Position:
    """Position in portfolio"""
    symbol: str
    quantity: float
    cost_basis: float
    purchase_date: str  # ISO format string for compatibility
    
    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > 0 else 0


@dataclass
class StockMetadata:
    """Stock metadata"""
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    
    @property
    def market_cap_tier(self) -> str:
        if self.market_cap > 200e9:
            return "Mega"
        elif self.market_cap > 10e9:
            return "Large"
        return "Mid"

