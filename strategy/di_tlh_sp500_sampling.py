"""
Tax Loss Harvesting System for S&P 500 Index Tracking
Uses 100-stock sampling with intelligent replacement strategy

USAGE:
1. Set environment variables:
   - ALPACA_API_KEY, ALPACA_SECRET_KEY
   - FMP_API_KEY
   - ALPACA_BASE_URL (use paper trading URL for testing)

2. Run initial setup:
   python tlh_system.py --setup --capital 10000

3. Run daily/weekly harvesting:
   python tlh_system.py --harvest

4. Generate tax report:
   python tlh_system.py --tax-report --year 2024
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import requests
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging
import time
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Use unified config
from strategy.config import TLHConfig as Config

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Position:
    symbol: str
    quantity: float
    cost_basis: float
    purchase_date: str
    
    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > 0 else 0

@dataclass
class Transaction:
    timestamp: str
    type: str
    symbol: str
    quantity: float
    price: float
    fees: float
    total: float
    notes: str = ""

@dataclass
class WashSale:
    symbol: str
    sale_date: str
    loss_amount: float
    quantity_sold: float
    
    def is_active(self, check_date: datetime) -> bool:
        sale_dt = datetime.fromisoformat(self.sale_date)
        return (check_date - sale_dt).days < Config.WASH_SALE_DAYS

@dataclass
class StockMetadata:
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

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging():
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# API CLIENTS
# ============================================================================

class AlpacaClient:
    def __init__(self):
        self.base_url = Config.ALPACA_BASE_URL
        self.headers = {
            'APCA-API-KEY-ID': Config.ALPACA_API_KEY,
            'APCA-API-SECRET-KEY': Config.ALPACA_SECRET_KEY
        }
    
    def _request(self, method: str, endpoint: str, data: dict = None) -> dict:
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        
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
                logger.error(f"API error (attempt {attempt + 1}): {e}")
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

class FMPClient:
    def __init__(self):
        self.base_url = 'https://financialmodelingprep.com/api/v3'
        self.api_key = Config.FMP_API_KEY
        self.call_count = 0
    
    def _request(self, endpoint: str) -> dict:
        if self.call_count >= 250:
            logger.warning("FMP API daily limit reached")
            return None
        
        url = f"{self.base_url}{endpoint}"
        params = {'apikey': self.api_key}
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            self.call_count += 1
            return r.json()
        except Exception as e:
            logger.error(f"FMP API error: {e}")
            return None
    
    def get_sp500_list(self) -> List[str]:
        data = self._request('/sp500_constituent')
        return [item['symbol'] for item in data] if data else []
    
    def get_profile(self, symbol: str) -> Optional[StockMetadata]:
        data = self._request(f'/profile/{symbol}')
        if data and len(data) > 0:
            info = data[0]
            return StockMetadata(
                symbol=symbol,
                name=info.get('companyName', ''),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                market_cap=info.get('mktCap', 0)
            )
        return None

# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    @staticmethod
    def save_json(filepath: str, data):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(filepath: str, default=None):
        if not os.path.exists(filepath):
            return default if default is not None else {}
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_positions(positions: Dict[str, Position]):
        data = {sym: asdict(pos) for sym, pos in positions.items()}
        DataManager.save_json(Config.POSITIONS_FILE, data)
    
    @staticmethod
    def load_positions() -> Dict[str, Position]:
        data = DataManager.load_json(Config.POSITIONS_FILE, {})
        return {sym: Position(**pos) for sym, pos in data.items()}
    
    @staticmethod
    def add_transaction(txn: Transaction):
        txns = DataManager.load_json(Config.TRANSACTIONS_FILE, [])
        txns.append(asdict(txn))
        DataManager.save_json(Config.TRANSACTIONS_FILE, txns)
    
    @staticmethod
    def load_transactions() -> List[Transaction]:
        data = DataManager.load_json(Config.TRANSACTIONS_FILE, [])
        return [Transaction(**t) for t in data]
    
    @staticmethod
    def add_wash_sale(ws: WashSale):
        wash_sales = DataManager.load_json(Config.WASH_SALES_FILE, [])
        wash_sales.append(asdict(ws))
        DataManager.save_json(Config.WASH_SALES_FILE, wash_sales)
    
    @staticmethod
    def load_wash_sales() -> List[WashSale]:
        data = DataManager.load_json(Config.WASH_SALES_FILE, [])
        return [WashSale(**w) for w in data]

# ============================================================================
# CORE SYSTEM
# ============================================================================

# Import core strategy and execution engines
from strategy.core import TLHStrategy
from strategy.paper_trade_engine import PaperTradeExecutionEngine
from strategy.live_trade_engine import LiveTradeExecutionEngine

class TaxLossHarvestingSystem:
    def __init__(self, mode: str = 'paper'):
        """
        Initialize TLH system.
        
        Args:
            mode: 'paper' for paper trading, 'live' for live trading
        """
        self.mode = mode
        self.fmp = FMPClient()
        self.metadata: Dict[str, StockMetadata] = {}
        
        # Create execution engine based on mode
        if mode == 'live':
            self.execution_engine = LiveTradeExecutionEngine(logger=logger)
        else:
            self.execution_engine = PaperTradeExecutionEngine(logger=logger)
    
    def setup_portfolio(self, capital: float):
        """Initial portfolio setup"""
        logger.info(f"Setting up portfolio with ${capital:,.2f}")
        
        # Get S&P 500 list
        sp500_symbols = self.fmp.get_sp500_list()
        logger.info(f"Found {len(sp500_symbols)} S&P 500 stocks")
        
        # Build metadata
        self._build_metadata(sp500_symbols)
        
        # Select portfolio stocks
        selected = self._select_stocks(sp500_symbols, Config.NUM_STOCKS)
        
        # Initialize positions
        self._initialize_positions(selected, capital)
        
        logger.info("Portfolio setup complete")
    
    def _build_metadata(self, symbols: List[str]):
        """Build metadata cache"""
        cached = DataManager.load_json(Config.METADATA_FILE, {})
        
        for i, sym in enumerate(symbols):
            if sym in cached:
                self.metadata[sym] = StockMetadata(**cached[sym])
                continue
            
            logger.info(f"Fetching metadata {i+1}/{len(symbols)}: {sym}")
            meta = self.fmp.get_profile(sym)
            if meta:
                self.metadata[sym] = meta
            time.sleep(0.2)
        
        cache_data = {s: asdict(m) for s, m in self.metadata.items()}
        DataManager.save_json(Config.METADATA_FILE, cache_data)
    
    def _select_stocks(self, sp500: List[str], n: int) -> List[str]:
        """Select n stocks from S&P 500 using core strategy"""
        # Convert metadata to dict format expected by strategy
        metadata_dict = {}
        for symbol, meta in self.metadata.items():
            metadata_dict[symbol] = {
                'market_cap': meta.market_cap,
                'sector': meta.sector,
                'industry': meta.industry
            }
        
        # Use core strategy
        return TLHStrategy.select_stocks(sp500, n, metadata_dict)
    
    def _initialize_positions(self, symbols: List[str], capital: float):
        """Buy initial positions using execution engine"""
        per_stock = capital / len(symbols)
        positions = {}
        
        for sym in symbols:
            try:
                price = self.execution_engine.get_price(sym)
                if not price:
                    logger.warning(f"Skip {sym}: no price available")
                    continue
                
                qty = per_stock / price
                
                if qty < 0.01:
                    logger.warning(f"Skip {sym}: qty too small")
                    continue
                
                success = self.execution_engine.execute_buy(sym, qty, price)
                
                if success:
                    positions[sym] = Position(
                        symbol=sym,
                        quantity=qty,
                        cost_basis=qty * price,
                        purchase_date=datetime.now().isoformat()
                    )
                    
                    DataManager.add_transaction(Transaction(
                        timestamp=datetime.now().isoformat(),
                        type='buy',
                        symbol=sym,
                        quantity=qty,
                        price=price,
                        fees=0.0,
                        total=qty * price,
                        notes="Initial"
                    ))
                
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error buying {sym}: {e}")
        
        if len(positions) < 50:
            raise ValueError(f"Only {len(positions)} positions created. Need 50+")
        
        DataManager.save_positions(positions)
        logger.info(f"Created {len(positions)} positions")
    
    def run_harvest(self):
        """Execute tax loss harvesting using core strategy"""
        logger.info("Starting harvest cycle")
        
        positions = DataManager.load_positions()
        wash_sales = DataManager.load_wash_sales()
        
        # Filter active wash sales
        now = datetime.now()
        active_wash = [w.symbol for w in wash_sales if w.is_active(now)]
        
        harvests_executed = 0
        
        for sym, pos in positions.items():
            try:
                # Get current price
                curr_price = self.execution_engine.get_price(sym)
                if not curr_price:
                    continue
                
                # Calculate P&L using core strategy
                curr_value = pos.quantity * curr_price
                should_harvest, pnl_pct = TLHStrategy.should_harvest(
                    pos.cost_basis, curr_value, Config.HARVEST_THRESHOLD
                )
                
                if should_harvest:
                    logger.info(f"{sym}: Loss {pnl_pct:.2%}, harvesting")
                    
                    # Find replacement using core strategy
                    available_symbols = self.execution_engine.get_available_symbols()
                    if not available_symbols:
                        # Fallback to all positions
                        available_symbols = list(positions.keys())
                    
                    # Convert metadata to dict format
                    metadata_dict = {}
                    for symbol, meta in self.metadata.items():
                        metadata_dict[symbol] = {
                            'market_cap': meta.market_cap,
                            'sector': meta.sector,
                            'industry': meta.industry
                        }
                    
                    # Create price data function
                    def price_data_func(symbol, start_date, end_date):
                        return self.execution_engine.get_prices_range(symbol, start_date, end_date)
                    
                    replacement = TLHStrategy.find_replacement(
                        sold_symbol=sym,
                        available_symbols=available_symbols,
                        current_positions=list(positions.keys()),
                        wash_sale_symbols=active_wash,
                        metadata=metadata_dict,
                        price_data_func=price_data_func,
                        correlation_lookback_days=Config.CORRELATION_LOOKBACK_DAYS,
                        current_date=now,
                        logger=logger
                    )
                    
                    if replacement:
                        self._execute_harvest(pos, replacement, curr_price)
                        harvests_executed += 1
                        del positions[sym]
                    
            except Exception as e:
                logger.error(f"Error harvesting {sym}: {e}")
        
        logger.info(f"Harvest complete: {harvests_executed} executed")
    
    
    def _execute_harvest(self, pos: Position, repl: str, price: float):
        """Execute harvest trade using execution engine"""
        # Sell
        realized_pnl = self.execution_engine.execute_sell(pos.symbol, pos.quantity, price)
        
        if realized_pnl is None:
            logger.error(f"Failed to sell {pos.symbol}")
            return
        
        proceeds = pos.quantity * price
        loss = pos.cost_basis - proceeds
        
        DataManager.add_transaction(Transaction(
            timestamp=datetime.now().isoformat(),
            type='sell',
            symbol=pos.symbol,
            quantity=pos.quantity,
            price=price,
            fees=0.0,
            total=proceeds,
            notes=f"Harvest: ${loss:.2f} loss"
        ))
        
        DataManager.add_wash_sale(WashSale(
            symbol=pos.symbol,
            sale_date=datetime.now().isoformat(),
            loss_amount=loss,
            quantity_sold=pos.quantity
        ))
        
        # Buy replacement
        repl_price = self.execution_engine.get_price(repl)
        if not repl_price:
            logger.error(f"Failed to get price for replacement {repl}")
            return
        
        repl_qty = proceeds / repl_price
        
        success = self.execution_engine.execute_buy(repl, repl_qty, repl_price)
        
        if success:
            DataManager.add_transaction(Transaction(
                timestamp=datetime.now().isoformat(),
                type='buy',
                symbol=repl,
                quantity=repl_qty,
                price=repl_price,
                fees=0.0,
                total=proceeds,
                notes=f"Replacement for {pos.symbol}"
            ))
            
            # Update positions
            positions = DataManager.load_positions()
            positions[repl] = Position(
                symbol=repl,
                quantity=repl_qty,
                cost_basis=proceeds,
                purchase_date=datetime.now().isoformat()
            )
            DataManager.save_positions(positions)
            
            logger.info(f"✓ {pos.symbol} -> {repl}, Loss: ${loss:.2f}")
    
    def generate_tax_report(self, year: int) -> str:
        """Generate Form 8949 report"""
        txns = DataManager.load_transactions()
        sales = [t for t in txns if t.type == 'sell' 
                 and datetime.fromisoformat(t.timestamp).year == year]
        
        lines = [
            f"FORM 8949 - Tax Year {year}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d')}",
            "",
            f"{'Symbol':<10} {'Acquired':<12} {'Sold':<12} {'Proceeds':>12} {'Cost':>12} {'Gain/Loss':>12}",
            "-" * 72
        ]
        
        total_loss = 0.0
        buys = {t.symbol: t for t in txns if t.type == 'buy'}
        
        for sale in sales:
            buy = buys.get(sale.symbol)
            if buy:
                cost = buy.price * sale.quantity
                proceeds = sale.total
                gl = proceeds - cost
                total_loss += gl
                
                lines.append(
                    f"{sale.symbol:<10} "
                    f"{datetime.fromisoformat(buy.timestamp).strftime('%m/%d/%Y'):<12} "
                    f"{datetime.fromisoformat(sale.timestamp).strftime('%m/%d/%Y'):<12} "
                    f"${proceeds:>11,.2f} ${cost:>11,.2f} ${gl:>11,.2f}"
                )
        
        lines.extend([
            "-" * 72,
            f"Total Realized Losses: ${abs(min(0, total_loss)):,.2f}",
            "",
            "Consult a tax professional for official filing."
        ])
        
        report = "\n".join(lines)
        
        # Save to file
        filepath = f"{Config.DATA_DIR}/form_8949_{year}.txt"
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Tax report saved to {filepath}")
        return report

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Tax Loss Harvesting System')
    parser.add_argument('--setup', action='store_true', help='Setup initial portfolio')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--harvest', action='store_true', help='Run harvest cycle')
    parser.add_argument('--tax-report', action='store_true', help='Generate tax report')
    parser.add_argument('--year', type=int, default=datetime.now().year, help='Tax year')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode: paper (default) or live')
    
    args = parser.parse_args()
    
    system = TaxLossHarvestingSystem(mode=args.mode)
    
    if args.setup:
        system.setup_portfolio(args.capital)
    elif args.harvest:
        system.run_harvest()
    elif args.tax_report:
        print(system.generate_tax_report(args.year))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
