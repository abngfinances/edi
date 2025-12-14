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

class Config:
    """System configuration"""
    # API Keys
    ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
    ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
    ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    FMP_API_KEY = os.getenv('FMP_API_KEY', '')
    
    # Trading Parameters
    INITIAL_CAPITAL = 10000.0
    NUM_STOCKS = 100
    HARVEST_THRESHOLD = 0.005  # 0.5% loss threshold
    CORRELATION_LOOKBACK_DAYS = 90
    
    # Schedule
    TRADING_HOUR = 14  # 2 PM ET
    HARVEST_FREQUENCY = 'weekly'
    
    # Wash Sale
    WASH_SALE_DAYS = 30
    
    # File paths
    DATA_DIR = 'data'
    POSITIONS_FILE = f'{DATA_DIR}/positions.json'
    TRANSACTIONS_FILE = f'{DATA_DIR}/transactions.json'
    WASH_SALES_FILE = f'{DATA_DIR}/wash_sales.json'
    METADATA_FILE = f'{DATA_DIR}/sp500_metadata.json'
    
    # Logging
    LOG_FILE = f'{DATA_DIR}/tlh_system.log'
    LOG_LEVEL = logging.INFO

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

class TaxLossHarvestingSystem:
    def __init__(self):
        self.alpaca = AlpacaClient()
        self.fmp = FMPClient()
        self.metadata: Dict[str, StockMetadata] = {}
    
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
        """Select n stocks from S&P 500"""
        df = pd.DataFrame([
            {'symbol': s, 'market_cap': self.metadata[s].market_cap,
             'sector': self.metadata[s].sector}
            for s in sp500 if s in self.metadata
        ]).sort_values('market_cap', ascending=False)
        
        # Top 50 by market cap
        selected = df.head(50)['symbol'].tolist()
        
        # Distribute remaining by sector
        remaining_df = df[~df['symbol'].isin(selected)]
        for sector in remaining_df['sector'].unique():
            sector_df = remaining_df[remaining_df['sector'] == sector]
            count = min(5, len(sector_df))
            selected.extend(sector_df.head(count)['symbol'].tolist())
        
        return selected[:n]
    
    def _initialize_positions(self, symbols: List[str], capital: float):
        """Buy initial positions"""
        per_stock = capital / len(symbols)
        positions = {}
        
        for sym in symbols:
            try:
                trade = self.alpaca.get_latest_trade(sym)
                price = trade['trade']['p']
                qty = per_stock / price
                
                if qty < 0.01:
                    logger.warning(f"Skip {sym}: qty too small")
                    continue
                
                self.alpaca.place_order(sym, qty, 'buy')
                
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
        """Execute tax loss harvesting"""
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
                trade = self.alpaca.get_latest_trade(sym)
                curr_price = trade['trade']['p']
                
                # Calculate P&L
                curr_value = pos.quantity * curr_price
                pnl = curr_value - pos.cost_basis
                pnl_pct = pnl / pos.cost_basis
                
                # Check if loss exceeds threshold
                if pnl_pct < -Config.HARVEST_THRESHOLD:
                    logger.info(f"{sym}: Loss {pnl_pct:.2%}, harvesting")
                    
                    # Find replacement
                    available = [s for s in positions.keys() if s not in active_wash]
                    replacement = self._find_replacement(sym, available, active_wash)
                    
                    if replacement:
                        self._execute_harvest(pos, replacement, curr_price)
                        harvests_executed += 1
                        del positions[sym]
                    
            except Exception as e:
                logger.error(f"Error harvesting {sym}: {e}")
        
        logger.info(f"Harvest complete: {harvests_executed} executed")
    
    def _find_replacement(self, sold: str, available: List[str], 
                         wash: List[str]) -> Optional[str]:
        """Find replacement stock"""
        if sold not in self.metadata:
            return None
        
        sold_meta = self.metadata[sold]
        candidates = [
            s for s in available 
            if s != sold and s not in wash and s in self.metadata
        ]
        
        # Filter by sector and tier
        same_sector = [
            s for s in candidates
            if self.metadata[s].sector == sold_meta.sector
            and self.metadata[s].market_cap_tier == sold_meta.market_cap_tier
        ]
        
        if not same_sector:
            same_sector = [
                s for s in candidates
                if self.metadata[s].market_cap_tier == sold_meta.market_cap_tier
            ]
        
        if not same_sector:
            same_sector = candidates
        
        # Calculate correlations
        corrs = self._calc_correlations(sold, same_sector)
        return corrs.idxmax() if not corrs.empty else None
    
    def _calc_correlations(self, target: str, candidates: List[str]) -> pd.Series:
        """Calculate correlations"""
        all_syms = [target] + candidates
        bars = self.alpaca.get_bars(all_syms, limit=90)
        
        if not bars or 'bars' not in bars:
            return pd.Series()
        
        prices = {}
        for s in all_syms:
            if s in bars['bars']:
                prices[s] = [b['c'] for b in bars['bars'][s]]
        
        if target not in prices:
            return pd.Series()
        
        df = pd.DataFrame(prices)
        returns = df.pct_change().dropna()
        corrs = returns.corr()[target].drop(target)
        
        return corrs.sort_values(ascending=False)
    
    def _execute_harvest(self, pos: Position, repl: str, price: float):
        """Execute harvest trade"""
        # Sell
        self.alpaca.place_order(pos.symbol, pos.quantity, 'sell')
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
        repl_trade = self.alpaca.get_latest_trade(repl)
        repl_price = repl_trade['trade']['p']
        repl_qty = proceeds / repl_price
        
        self.alpaca.place_order(repl, repl_qty, 'buy')
        
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
    
    args = parser.parse_args()
    
    system = TaxLossHarvestingSystem()
    
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
