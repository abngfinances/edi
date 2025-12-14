"""
Paper Trade Execution Engine

Implements ExecutionEngine interface for paper trading using Alpaca Paper Trading API.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import requests
import time

from strategy.execution import ExecutionEngine
from strategy.models import Position, StockMetadata
from strategy.config import TLHConfig


class AlpacaClient:
    """Alpaca API client"""
    def __init__(self, base_url: str, api_key: str, secret_key: str):
        self.base_url = base_url
        self.headers = {
            'APCA-API-KEY-ID': api_key,
            'APCA-API-SECRET-KEY': secret_key
        }
    
    def _request(self, method: str, endpoint: str, data: dict = None, max_retries: int = 3) -> dict:
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(max_retries):
            try:
                if method == 'GET':
                    r = requests.get(url, headers=self.headers, params=data)
                elif method == 'POST':
                    r = requests.post(url, headers=self.headers, json=data)
                elif method == 'DELETE':
                    r = requests.delete(url, headers=self.headers)
                
                r.raise_for_status()
                return r.json()
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def get_account(self) -> dict:
        return self._request('GET', '/v2/account')
    
    def place_order(self, symbol: str, qty: float, side: str) -> dict:
        data = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': 'market',
            'time_in_force': 'day'
        }
        return self._request('POST', '/v2/orders', data)
    
    def get_bars(self, symbols: List[str], limit: int = 100) -> dict:
        params = {
            'symbols': ','.join(symbols),
            'timeframe': '1Day',
            'limit': limit
        }
        return self._request('GET', '/v2/stocks/bars', params)
    
    def get_latest_trade(self, symbol: str) -> dict:
        return self._request('GET', f'/v2/stocks/{symbol}/trades/latest')
    
    def get_positions(self) -> List[dict]:
        return self._request('GET', '/v2/positions')


class PaperTradeExecutionEngine(ExecutionEngine):
    """Execution engine for paper trading using Alpaca Paper Trading API"""
    
    def __init__(self, api_key: str = None, secret_key: str = None, base_url: str = None,
                 logger=None):
        """
        Initialize paper trade execution engine.
        
        Args:
            api_key: Alpaca API key (defaults to TLHConfig)
            secret_key: Alpaca secret key (defaults to TLHConfig)
            base_url: Alpaca base URL (defaults to TLHConfig, paper trading)
            logger: Optional logger
        """
        self.api_key = api_key or TLHConfig.ALPACA_API_KEY
        self.secret_key = secret_key or TLHConfig.ALPACA_SECRET_KEY
        self.base_url = base_url or TLHConfig.ALPACA_BASE_URL
        self.logger = logger
        
        self.alpaca = AlpacaClient(self.base_url, self.api_key, self.secret_key)
    
    def get_price(self, symbol: str, date: Optional[datetime] = None) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            trade = self.alpaca.get_latest_trade(symbol)
            return trade['trade']['p']
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_prices_range(self, symbol: str, start_date: datetime, 
                        end_date: datetime) -> pd.DataFrame:
        """Get historical prices for a symbol over a date range"""
        try:
            # Calculate number of days
            days = (end_date - start_date).days
            limit = min(days + 10, 1000)  # Alpaca limit
            
            bars = self.alpaca.get_bars([symbol], limit=limit)
            
            if not bars or 'bars' not in bars or symbol not in bars['bars']:
                return pd.DataFrame()
            
            bar_data = bars['bars'][symbol]
            
            # Convert to DataFrame
            df = pd.DataFrame(bar_data)
            df['t'] = pd.to_datetime(df['t'])
            df = df.rename(columns={
                't': 'date',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            })
            
            # Filter by date range
            df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
            
            return df[['date', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting price range for {symbol}: {e}")
            return pd.DataFrame()
    
    def execute_buy(self, symbol: str, quantity: float, price: float,
                   date: Optional[datetime] = None) -> bool:
        """Execute a buy order"""
        try:
            self.alpaca.place_order(symbol, quantity, 'buy')
            if self.logger:
                self.logger.info(f"Buy order placed: {quantity} shares of {symbol} at ${price:.2f}")
            return True
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing buy for {symbol}: {e}")
            return False
    
    def execute_sell(self, symbol: str, quantity: float, price: float,
                    date: Optional[datetime] = None) -> Optional[float]:
        """Execute a sell order"""
        try:
            # Get current position to calculate P&L
            positions = self.alpaca.get_positions()
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                if self.logger:
                    self.logger.warning(f"No position found for {symbol}")
                return None
            
            cost_basis = float(position['cost_basis'])
            proceeds = quantity * price
            realized_pnl = proceeds - (cost_basis * (quantity / float(position['qty'])))
            
            self.alpaca.place_order(symbol, quantity, 'sell')
            if self.logger:
                self.logger.info(f"Sell order placed: {quantity} shares of {symbol} at ${price:.2f}, P&L: ${realized_pnl:.2f}")
            
            return realized_pnl
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error executing sell for {symbol}: {e}")
            return None
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        try:
            positions = self.alpaca.get_positions()
            result = {}
            for pos in positions:
                result[pos['symbol']] = Position(
                    symbol=pos['symbol'],
                    quantity=float(pos['qty']),
                    cost_basis=float(pos['cost_basis']),
                    purchase_date=datetime.now().isoformat()  # Alpaca doesn't provide purchase date
                )
            return result
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_metadata(self, symbols: List[str]) -> Dict[str, StockMetadata]:
        """Get stock metadata (not available from Alpaca, returns minimal data)"""
        # Alpaca doesn't provide metadata, so return minimal data
        # In practice, this would come from FMP or another source
        result = {}
        for symbol in symbols:
            result[symbol] = StockMetadata(
                symbol=symbol,
                name=symbol,
                sector='Unknown',
                industry='Unknown',
                market_cap=0
            )
        return result
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols (S&P 500 - would need external source)"""
        # This would typically come from FMP or another data source
        # For now, return empty - caller should provide
        return []


class PaperTradeRunner:
    """Runner for paper trading using core strategy"""
    
    def __init__(self, execution_engine: PaperTradeExecutionEngine, logger=None):
        self.execution_engine = execution_engine
        self.logger = logger or execution_engine.logger
    
    def run_harvest(self):
        """Execute tax loss harvesting cycle"""
        from strategy.core import TLHStrategy
        from strategy.config import TLHConfig
        
        self.logger.info("Starting paper trade harvest cycle")
        
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
                    self.logger.info(f"{symbol}: Loss {pnl_pct:.2%}, harvesting")
                    
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

