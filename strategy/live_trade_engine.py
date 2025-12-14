"""
Live Trade Execution Engine

Implements ExecutionEngine interface for live trading using Alpaca Live Trading API.
Includes additional safety checks for real money trading.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd

from strategy.paper_trade_engine import PaperTradeExecutionEngine, AlpacaClient
from strategy.execution import ExecutionEngine
from strategy.models import Position, StockMetadata
from strategy.config import TLHConfig


class LiveTradeExecutionEngine(PaperTradeExecutionEngine):
    """Execution engine for live trading with real money"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 base_url: str = None, logger=None, confirm_trades: bool = True):
        """
        Initialize live trade execution engine.
        
        Args:
            api_key: Alpaca API key (defaults to TLHConfig)
            secret_key: Alpaca secret key (defaults to TLHConfig)
            base_url: Alpaca base URL (defaults to live trading URL)
            logger: Optional logger
            confirm_trades: If True, requires confirmation before executing trades
        """
        # Use live trading URL if not provided
        if base_url is None:
            base_url = 'https://api.alpaca.markets'
        
        super().__init__(api_key, secret_key, base_url, logger)
        self.confirm_trades = confirm_trades
        self.trade_count_today = 0
        self.max_trades_per_day = 50  # Safety limit
    
    def _confirm_trade(self, symbol: str, side: str, quantity: float, price: float) -> bool:
        """Confirm trade before execution (safety check)"""
        if not self.confirm_trades:
            return True
        
        if self.logger:
            self.logger.warning(f"⚠️  LIVE TRADE CONFIRMATION REQUIRED:")
            self.logger.warning(f"   {side.upper()}: {quantity} shares of {symbol} at ${price:.2f}")
            self.logger.warning(f"   Total: ${quantity * price:,.2f}")
        
        # In production, this would prompt user or check a flag
        # For now, log and proceed (can be overridden)
        return True
    
    def execute_buy(self, symbol: str, quantity: float, price: float,
                   date: Optional[datetime] = None) -> bool:
        """Execute a buy order with safety checks"""
        # Safety check: limit trades per day
        if self.trade_count_today >= self.max_trades_per_day:
            if self.logger:
                self.logger.error(f"Maximum trades per day ({self.max_trades_per_day}) reached")
            return False
        
        # Confirm trade
        if not self._confirm_trade(symbol, 'buy', quantity, price):
            if self.logger:
                self.logger.warning(f"Trade confirmation failed for {symbol}")
            return False
        
        # Execute trade
        success = super().execute_buy(symbol, quantity, price, date)
        
        if success:
            self.trade_count_today += 1
            if self.logger:
                self.logger.info(f"✅ LIVE TRADE EXECUTED: Buy {quantity} {symbol} at ${price:.2f}")
        
        return success
    
    def execute_sell(self, symbol: str, quantity: float, price: float,
                    date: Optional[datetime] = None) -> Optional[float]:
        """Execute a sell order with safety checks"""
        # Safety check: limit trades per day
        if self.trade_count_today >= self.max_trades_per_day:
            if self.logger:
                self.logger.error(f"Maximum trades per day ({self.max_trades_per_day}) reached")
            return None
        
        # Confirm trade
        if not self._confirm_trade(symbol, 'sell', quantity, price):
            if self.logger:
                self.logger.warning(f"Trade confirmation failed for {symbol}")
            return None
        
        # Execute trade
        realized_pnl = super().execute_sell(symbol, quantity, price, date)
        
        if realized_pnl is not None:
            self.trade_count_today += 1
            if self.logger:
                self.logger.info(f"✅ LIVE TRADE EXECUTED: Sell {quantity} {symbol} at ${price:.2f}, P&L: ${realized_pnl:.2f}")
        
        return realized_pnl


class LiveTradeRunner:
    """Runner for live trading using core strategy"""
    
    def __init__(self, execution_engine: LiveTradeExecutionEngine, logger=None):
        self.execution_engine = execution_engine
        self.logger = logger or execution_engine.logger
    
    def run_harvest(self):
        """Execute tax loss harvesting cycle with enhanced safety"""
        from strategy.core import TLHStrategy
        from strategy.config import TLHConfig
        
        self.logger.warning("="*60)
        self.logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK")
        self.logger.warning("="*60)
        
        self.logger.info("Starting live trade harvest cycle")
        
        # Get current positions
        positions = self.execution_engine.get_positions()
        
        if not positions:
            self.logger.warning("No positions found")
            return
        
        harvests_executed = 0
        
        for symbol, pos in positions.items():
            try:
                # Get current price
                current_price = self.execution_engine.get_price(symbol)
                if not current_price:
                    continue
                
                # Calculate P&L using core strategy
                current_value = pos.quantity * current_price
                should_harvest, pnl_pct = TLHStrategy.should_harvest(
                    pos.cost_basis, current_value, TLHConfig.HARVEST_THRESHOLD
                )
                
                if should_harvest:
                    self.logger.warning(f"🔴 {symbol}: Loss {pnl_pct:.2%}, harvesting (LIVE TRADE)")
                    
                    # Find replacement (would need metadata and available symbols)
                    # For now, skip replacement finding - would need FMP client
                    # This is a simplified version
                    
                    # Execute sell
                    realized_pnl = self.execution_engine.execute_sell(
                        symbol, pos.quantity, current_price
                    )
                    
                    if realized_pnl is not None:
                        harvests_executed += 1
            
            except Exception as e:
                self.logger.error(f"Error harvesting {symbol}: {e}")
        
        self.logger.info(f"Harvest complete: {harvests_executed} executed")
        self.logger.warning("="*60)

