"""
Tax Loss Harvesting Backtesting Engine

Simulates the TLH strategy on historical data to evaluate:
1. Total return vs buy-and-hold S&P 500
2. Tax alpha (value of harvested losses)
3. Tracking error
4. Number of harvests executed
5. Transaction costs impact

This allows testing before paper trading or live trading.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress the FutureWarning about idxmax with NaN values
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Import our data loader
import sys
sys.path.append('.')
from backtest_data_downloader import BacktestDataLoader, BacktestConfig

# ============================================================================
# CONFIGURATION
# ============================================================================

class BacktestParams:
    """Backtesting parameters"""
    INITIAL_CAPITAL = 100000.0  # Start with $100k for meaningful results
    NUM_STOCKS = 100
    HARVEST_THRESHOLD = 0.005  # 0.5% loss threshold
    HARVEST_FREQUENCY = 'weekly'  # 'daily', 'weekly', 'monthly'
    
    # Tax assumptions
    SHORT_TERM_TAX_RATE = 0.37  # Federal + state for high earners
    LONG_TERM_TAX_RATE = 0.238  # 20% federal + 3.8% NIIT
    
    # Transaction costs (most brokers are $0 now, but include for realism)
    COMMISSION_PER_TRADE = 0.0
    SPREAD_BPS = 1  # 1 basis point for spread cost
    
    # Wash sale
    WASH_SALE_DAYS = 30
    
    # Correlation parameters
    CORRELATION_LOOKBACK = 90
    
    # Output
    RESULTS_DIR = 'backtest_results'
    
# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BacktestPosition:
    """Position during backtest"""
    symbol: str
    quantity: float
    cost_basis: float
    purchase_date: datetime
    
    @property
    def avg_price(self) -> float:
        return self.cost_basis / self.quantity if self.quantity > 0 else 0

@dataclass
class BacktestTrade:
    """Trade executed during backtest"""
    date: datetime
    type: str  # 'buy' or 'sell'
    symbol: str
    quantity: float
    price: float
    commission: float
    spread_cost: float
    total_cost: float
    notes: str = ""

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    # Strategy performance
    final_value: float
    total_return: float
    annualized_return: float
    
    # Benchmark comparison
    benchmark_final_value: float
    benchmark_return: float
    benchmark_annualized_return: float
    
    # Tax metrics
    total_losses_harvested: float
    tax_alpha: float  # Value of tax savings
    num_harvests: int
    
    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    tracking_error: float
    
    # Trade statistics
    total_trades: int
    total_transaction_costs: float
    avg_holding_period_days: float
    
    # Timeline data
    daily_values: pd.DataFrame
    trades: List[BacktestTrade]

# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class TLHBacktester:
    """Backtest tax loss harvesting strategy on historical data"""
    
    def __init__(self, params: BacktestParams):
        self.params = params
        self.data_loader = BacktestDataLoader(use_filtered=True)
        self.data_loader.load_data()
        
        # State
        self.positions: Dict[str, BacktestPosition] = {}
        self.cash = params.INITIAL_CAPITAL
        self.trades: List[BacktestTrade] = []
        self.wash_sales: Dict[str, datetime] = {}  # symbol -> sale_date
        self.daily_values: List[Dict] = []
        self.total_losses_harvested = 0.0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_trading_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Get all trading dates in the backtest period"""
        all_dates = self.data_loader.prices_df['date'].unique()
        all_dates = pd.to_datetime(all_dates)
        mask = (all_dates >= start_date) & (all_dates <= end_date)
        return sorted(all_dates[mask])
    
    def should_harvest_today(self, date: datetime, start_date: datetime) -> bool:
        """Determine if we should run harvest on this date"""
        if self.params.HARVEST_FREQUENCY == 'daily':
            return True
        elif self.params.HARVEST_FREQUENCY == 'weekly':
            # Trade on Mondays
            return date.weekday() == 0
        elif self.params.HARVEST_FREQUENCY == 'monthly':
            # Trade on first trading day of month
            prev_dates = [d for d in self.get_trading_dates(start_date, date) if d < date]
            if not prev_dates:
                return False
            return prev_dates[-1].month != date.month
        return False
    
    def get_price(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for a symbol on a date"""
        price = self.data_loader.get_price(symbol, date)
        
        # Handle NaN or invalid prices
        if price is None or pd.isna(price) or price <= 0:
            # Try to get price from nearby dates (up to 5 days before)
            for days_back in range(1, 6):
                alt_date = date - timedelta(days=days_back)
                alt_price = self.data_loader.get_price(symbol, alt_date)
                if alt_price and not pd.isna(alt_price) and alt_price > 0:
                    self.logger.warning(f"Using price from {days_back} days before for {symbol} on {date.date()}")
                    return alt_price
            
            self.logger.warning(f"Invalid price for {symbol} on {date}")
            return None
        
        return price
    
    def calculate_transaction_cost(self, quantity: float, price: float) -> Tuple[float, float]:
        """Calculate transaction costs"""
        commission = self.params.COMMISSION_PER_TRADE
        spread_cost = (quantity * price) * (self.params.SPREAD_BPS / 10000)
        return commission, spread_cost
    
    def execute_buy(self, symbol: str, date: datetime, amount: float) -> bool:
        """Execute a buy order"""
        price = self.get_price(symbol, date)
        if not price or price <= 0:
            self.logger.warning(f"Invalid price for {symbol} on {date}")
            return False
        
        commission, spread = self.calculate_transaction_cost(1, amount)
        total_cost = amount + commission + spread
        
        if self.cash < total_cost:
            self.logger.warning(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        quantity = amount / price
        
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
        
        # Record trade
        trade = BacktestTrade(
            date=date,
            type='buy',
            symbol=symbol,
            quantity=quantity,
            price=price,
            commission=commission,
            spread_cost=spread,
            total_cost=total_cost,
            notes=f"Buy {symbol}"
        )
        self.trades.append(trade)
        
        return True
    
    def execute_sell(self, symbol: str, date: datetime, reason: str = "") -> Optional[float]:
        """Execute a sell order, returns realized P&L"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        price = self.get_price(symbol, date)
        
        if not price or price <= 0:
            self.logger.warning(f"Invalid price for {symbol} on {date}")
            return None
        
        proceeds = pos.quantity * price
        commission, spread = self.calculate_transaction_cost(pos.quantity, price)
        net_proceeds = proceeds - commission - spread
        
        # Calculate realized P&L
        realized_pnl = net_proceeds - pos.cost_basis
        
        self.cash += net_proceeds
        
        # Record trade
        trade = BacktestTrade(
            date=date,
            type='sell',
            symbol=symbol,
            quantity=pos.quantity,
            price=price,
            commission=commission,
            spread_cost=spread,
            total_cost=proceeds,
            notes=f"Sell {symbol}: {reason}, P&L: ${realized_pnl:.2f}"
        )
        self.trades.append(trade)
        
        # Track wash sale
        if realized_pnl < 0:
            self.wash_sales[symbol] = date
            self.total_losses_harvested += abs(realized_pnl)
        
        # Remove position
        del self.positions[symbol]
        
        return realized_pnl
    
    def is_wash_sale_restricted(self, symbol: str, check_date: datetime) -> bool:
        """Check if symbol is in wash sale period"""
        if symbol not in self.wash_sales:
            return False
        
        sale_date = self.wash_sales[symbol]
        days_since = (check_date - sale_date).days
        return days_since < self.params.WASH_SALE_DAYS
    
    def calculate_correlations(self, target: str, candidates: List[str], 
                               date: datetime) -> pd.Series:
        """Calculate correlations for replacement selection"""
        end_date = date
        start_date = date - timedelta(days=self.params.CORRELATION_LOOKBACK)
        
        # Get returns for all symbols
        returns_dict = {}
        
        for symbol in [target] + candidates:
            prices = self.data_loader.get_prices_range(symbol, start_date, end_date)
            if len(prices) > 20:  # Need minimum data
                returns = prices['close'].pct_change().dropna()
                # Remove any NaN or inf values
                returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
                if len(returns) > 20:  # Still have enough after cleaning
                    returns_dict[symbol] = returns
        
        if target not in returns_dict or len(returns_dict) < 2:
            self.logger.warning(f"Insufficient data to calculate correlations for {target}")
            return pd.Series()
        
        # Align returns on dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna(how='any')  # Only use dates with all data
        
        if len(returns_df) < 20:
            self.logger.warning(f"Only {len(returns_df)} overlapping dates for correlation calculation")
            return pd.Series()
        
        if target not in returns_df.columns:
            return pd.Series()
        
        # Calculate correlation
        try:
            corr = returns_df.corr()[target]
            corr = corr.drop(target)
            
            # Remove NaN correlations
            corr = corr.dropna()
            
            if corr.empty:
                self.logger.warning(f"No valid correlations calculated for {target}")
                return pd.Series()
            
            return corr.sort_values(ascending=False)
        except Exception as e:
            self.logger.error(f"Error calculating correlations for {target}: {e}")
            return pd.Series()
    
    def find_replacement(self, sold_symbol: str, date: datetime) -> Optional[str]:
        """Find replacement stock using hybrid approach"""
        metadata = self.data_loader.metadata
        
        if sold_symbol not in metadata:
            return None
        
        sold_meta = metadata[sold_symbol]
        
        # Get all available symbols (not in positions, not in wash sale)
        available = [
            sym for sym in self.data_loader.selected_symbols
            if sym != sold_symbol 
            and sym not in self.positions
            and not self.is_wash_sale_restricted(sym, date)
        ]
        
        if not available:
            self.logger.warning(f"No replacement candidates for {sold_symbol}")
            return None
        
        # Filter by sector and market cap tier
        def get_tier(market_cap):
            if market_cap > 200e9:
                return "Mega"
            elif market_cap > 10e9:
                return "Large"
            return "Mid"
        
        sold_tier = get_tier(sold_meta.get('market_cap', 0))
        
        same_sector = [
            sym for sym in available
            if sym in metadata
            and metadata[sym].get('sector') == sold_meta.get('sector')
            and get_tier(metadata[sym].get('market_cap', 0)) == sold_tier
        ]
        
        # If no same-sector matches, use same tier
        if not same_sector:
            same_sector = [
                sym for sym in available
                if sym in metadata
                and get_tier(metadata[sym].get('market_cap', 0)) == sold_tier
            ]
        
        # If still none, use all available
        if not same_sector:
            same_sector = available
        
        # Calculate correlations
        correlations = self.calculate_correlations(sold_symbol, same_sector, date)
        
        if correlations.empty:
            # Fallback: select by market cap if correlations fail
            self.logger.warning(f"Correlation calculation failed for {sold_symbol}, using market cap ranking")
            
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
        
        return correlations.idxmax()
    
    def initialize_portfolio(self, start_date: datetime, symbols: List[str]):
        """Initialize portfolio with equal-weight positions"""
        self.logger.info(f"Initializing portfolio on {start_date} with {len(symbols)} stocks")
        
        amount_per_stock = self.params.INITIAL_CAPITAL / len(symbols)
        
        successful = 0
        for symbol in symbols:
            if self.execute_buy(symbol, start_date, amount_per_stock):
                successful += 1
        
        self.logger.info(f"Initialized {successful}/{len(symbols)} positions")
        self.logger.info(f"Cash remaining: ${self.cash:,.2f}")
    
    def run_harvest_cycle(self, date: datetime):
        """Run tax loss harvesting for current positions"""
        harvests = 0
        
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            current_price = self.get_price(symbol, date)
            
            if not current_price:
                self.logger.warning(f"Skipping {symbol} - no valid price on {date.date()}")
                continue
            
            # Calculate P&L
            current_value = pos.quantity * current_price
            pnl = current_value - pos.cost_basis
            pnl_pct = pnl / pos.cost_basis
            
            # Check if loss exceeds threshold
            if pnl_pct < -self.params.HARVEST_THRESHOLD:
                self.logger.info(f"{date.date()}: Harvesting {symbol} with {pnl_pct:.2%} loss")
                
                # Find replacement
                replacement = self.find_replacement(symbol, date)
                
                if replacement:
                    # Sell current position
                    realized_pnl = self.execute_sell(symbol, date, f"Loss harvest: {pnl_pct:.2%}")
                    
                    if realized_pnl is not None:
                        # Buy replacement with proceeds
                        buy_amount = current_value * 0.99  # Slightly less to account for costs
                        
                        # Verify replacement has valid price
                        repl_price = self.get_price(replacement, date)
                        if repl_price and repl_price > 0:
                            self.execute_buy(replacement, date, buy_amount)
                            self.logger.info(f"  Replaced with {replacement}")
                            harvests += 1
                        else:
                            self.logger.warning(f"  Replacement {replacement} has no valid price, keeping cash")
                else:
                    self.logger.warning(f"  Could not find replacement for {symbol}")
        
        return harvests
    
    def calculate_portfolio_value(self, date: datetime) -> float:
        """Calculate total portfolio value on a date"""
        value = self.cash
        
        for symbol, pos in self.positions.items():
            price = self.get_price(symbol, date)
            if price and not pd.isna(price):
                value += pos.quantity * price
            else:
                # Use last known price
                self.logger.warning(f"No price for {symbol} on {date.date()}, using cost basis")
                value += pos.cost_basis
        
        return value
    
    def run_backtest(self, start_date: str, end_date: str) -> BacktestResult:
        """Run complete backtest"""
        self.logger.info("="*60)
        self.logger.info("STARTING BACKTEST")
        self.logger.info("="*60)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get trading dates
        trading_dates = self.get_trading_dates(start_dt, end_dt)
        self.logger.info(f"Backtest period: {start_dt.date()} to {end_dt.date()}")
        self.logger.info(f"Trading days: {len(trading_dates)}")
        
        # Initialize portfolio
        self.initialize_portfolio(trading_dates[0], self.data_loader.selected_symbols[:self.params.NUM_STOCKS])
        
        # Track daily values
        total_harvests = 0
        
        for i, date in enumerate(trading_dates):
            # Run harvest if scheduled
            if i > 0 and self.should_harvest_today(date, start_dt):
                harvests = self.run_harvest_cycle(date)
                total_harvests += harvests
            
            # Record daily value
            portfolio_value = self.calculate_portfolio_value(date)
            
            self.daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'num_positions': len(self.positions)
            })
            
            # Progress update
            if (i + 1) % 50 == 0:
                self.logger.info(f"Progress: {i+1}/{len(trading_dates)} days, Value: ${portfolio_value:,.0f}")
        
        # Calculate results
        results = self._calculate_results(start_dt, end_dt, total_harvests)
        
        self.logger.info("="*60)
        self.logger.info("BACKTEST COMPLETE")
        self.logger.info("="*60)
        
        return results
    
    def _calculate_results(self, start_date: datetime, end_date: datetime, 
                          total_harvests: int) -> BacktestResult:
        """Calculate backtest metrics"""
        df = pd.DataFrame(self.daily_values)
        
        # Strategy performance
        initial_value = self.params.INITIAL_CAPITAL
        final_value = df.iloc[-1]['portfolio_value']
        total_return = (final_value - initial_value) / initial_value
        
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Calculate benchmark (SPY buy-and-hold)
        spy_start = self.get_price('SPY', start_date)
        spy_end = self.get_price('SPY', end_date)
        
        if spy_start and spy_end:
            benchmark_return = (spy_end - spy_start) / spy_start
            benchmark_final = initial_value * (1 + benchmark_return)
            benchmark_annualized = (1 + benchmark_return) ** (1 / years) - 1
        else:
            benchmark_return = 0
            benchmark_final = initial_value
            benchmark_annualized = 0
        
        # Tax alpha (assumes losses can offset gains at short-term rate)
        tax_alpha = self.total_losses_harvested * self.params.SHORT_TERM_TAX_RATE
        
        # Risk metrics
        returns = df['portfolio_value'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if volatility > 0:
            sharpe_ratio = (annualized_return - 0.04) / volatility  # Assume 4% risk-free
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Tracking error (vs benchmark)
        if spy_start:
            spy_prices = []
            for date in df['date']:
                spy_price = self.get_price('SPY', date)
                if spy_price:
                    spy_prices.append(spy_price)
                else:
                    spy_prices.append(np.nan)
            
            df['spy_value'] = (np.array(spy_prices) / spy_start) * initial_value
            tracking_diff = df['portfolio_value'] - df['spy_value']
            tracking_error = tracking_diff.std() / initial_value
        else:
            tracking_error = 0
        
        # Trade statistics
        total_trades = len(self.trades)
        total_transaction_costs = sum(t.commission + t.spread_cost for t in self.trades)
        
        # Average holding period
        hold_periods = []
        for trade in self.trades:
            if trade.type == 'sell':
                # Find matching buy
                buy_trades = [t for t in self.trades if t.symbol == trade.symbol and t.type == 'buy' and t.date < trade.date]
                if buy_trades:
                    last_buy = max(buy_trades, key=lambda t: t.date)
                    hold_period = (trade.date - last_buy.date).days
                    hold_periods.append(hold_period)
        
        avg_holding_period = np.mean(hold_periods) if hold_periods else 0
        
        return BacktestResult(
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            benchmark_final_value=benchmark_final,
            benchmark_return=benchmark_return,
            benchmark_annualized_return=benchmark_annualized,
            total_losses_harvested=self.total_losses_harvested,
            tax_alpha=tax_alpha,
            num_harvests=total_harvests,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            tracking_error=tracking_error,
            total_trades=total_trades,
            total_transaction_costs=total_transaction_costs,
            avg_holding_period_days=avg_holding_period,
            daily_values=df,
            trades=self.trades
        )
    
    def print_results(self, results: BacktestResult):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"\n📊 STRATEGY PERFORMANCE")
        print(f"  Final Value:        ${results.final_value:,.2f}")
        print(f"  Total Return:       {results.total_return:.2%}")
        print(f"  Annualized Return:  {results.annualized_return:.2%}")
        
        print(f"\n📈 BENCHMARK (SPY)")
        print(f"  Final Value:        ${results.benchmark_final_value:,.2f}")
        print(f"  Total Return:       {results.benchmark_return:.2%}")
        print(f"  Annualized Return:  {results.benchmark_annualized_return:.2%}")
        
        print(f"\n💰 TAX BENEFITS")
        print(f"  Losses Harvested:   ${results.total_losses_harvested:,.2f}")
        print(f"  Tax Alpha:          ${results.tax_alpha:,.2f}")
        print(f"  Number of Harvests: {results.num_harvests}")
        
        print(f"\n📉 RISK METRICS")
        print(f"  Sharpe Ratio:       {results.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:       {results.max_drawdown:.2%}")
        print(f"  Volatility (Ann.):  {results.volatility:.2%}")
        print(f"  Tracking Error:     {results.tracking_error:.2%}")
        
        print(f"\n🔄 TRADE STATISTICS")
        print(f"  Total Trades:       {results.total_trades}")
        print(f"  Transaction Costs:  ${results.total_transaction_costs:,.2f}")
        print(f"  Avg Holding Period: {results.avg_holding_period_days:.0f} days")
        
        print("\n" + "="*60)
        
        # Calculate after-tax outperformance
        after_tax_benefit = results.tax_alpha - results.total_transaction_costs
        print(f"\n✨ NET BENEFIT: ${after_tax_benefit:,.2f}")
        print(f"   ({after_tax_benefit/self.params.INITIAL_CAPITAL:.2%} of initial capital)")
        print("="*60 + "\n")
    
    def plot_results(self, results: BacktestResult, save_path: str = None):
        """Create visualization of backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        df = results.daily_values
        
        # Plot 1: Portfolio value over time
        ax1 = axes[0, 0]
        ax1.plot(df['date'], df['portfolio_value'], label='TLH Strategy', linewidth=2)
        if 'spy_value' in df.columns:
            ax1.plot(df['date'], df['spy_value'], label='SPY Benchmark', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative returns
        ax2 = axes[0, 1]
        returns = (df['portfolio_value'] / df['portfolio_value'].iloc[0] - 1) * 100
        ax2.plot(df['date'], returns, linewidth=2)
        ax2.set_title('Cumulative Return (%)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Return (%)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 3: Drawdown
        ax3 = axes[1, 0]
        cummax = df['portfolio_value'].cummax()
        drawdown = (df['portfolio_value'] - cummax) / cummax * 100
        ax3.fill_between(df['date'], drawdown, 0, alpha=0.3, color='red')
        ax3.plot(df['date'], drawdown, color='red', linewidth=1)
        ax3.set_title('Drawdown (%)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Number of positions over time
        ax4 = axes[1, 1]
        ax4.plot(df['date'], df['num_positions'], linewidth=2)
        ax4.set_title('Number of Positions')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Positions')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run backtest"""
    import os
    os.makedirs(BacktestParams.RESULTS_DIR, exist_ok=True)
    
    print("="*60)
    print("TAX LOSS HARVESTING - BACKTEST")
    print("="*60)
    print()
    print(f"Initial Capital:     ${BacktestParams.INITIAL_CAPITAL:,.0f}")
    print(f"Number of Stocks:    {BacktestParams.NUM_STOCKS}")
    print(f"Harvest Frequency:   {BacktestParams.HARVEST_FREQUENCY}")
    print(f"Loss Threshold:      {BacktestParams.HARVEST_THRESHOLD:.1%}")
    print()
    
    # Create backtester
    backtester = TLHBacktester(BacktestParams)
    
    # Run backtest
    results = backtester.run_backtest(
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # Print results
    backtester.print_results(results)
    
    # Save results
    # Create parameters dict manually since BacktestParams is not a dataclass
    params_dict = {
        'INITIAL_CAPITAL': BacktestParams.INITIAL_CAPITAL,
        'NUM_STOCKS': BacktestParams.NUM_STOCKS,
        'HARVEST_THRESHOLD': BacktestParams.HARVEST_THRESHOLD,
        'HARVEST_FREQUENCY': BacktestParams.HARVEST_FREQUENCY,
        'SHORT_TERM_TAX_RATE': BacktestParams.SHORT_TERM_TAX_RATE,
        'LONG_TERM_TAX_RATE': BacktestParams.LONG_TERM_TAX_RATE,
        'COMMISSION_PER_TRADE': BacktestParams.COMMISSION_PER_TRADE,
        'SPREAD_BPS': BacktestParams.SPREAD_BPS,
        'WASH_SALE_DAYS': BacktestParams.WASH_SALE_DAYS,
        'CORRELATION_LOOKBACK': BacktestParams.CORRELATION_LOOKBACK,
        'RESULTS_DIR': BacktestParams.RESULTS_DIR
    }
    results_dict = {
        'parameters': params_dict,
        'metrics': {
            'final_value': results.final_value,
            'total_return': results.total_return,
            'annualized_return': results.annualized_return,
            'benchmark_return': results.benchmark_return,
            'tax_alpha': results.tax_alpha,
            'num_harvests': results.num_harvests,
            'sharpe_ratio': results.sharpe_ratio,
            'max_drawdown': results.max_drawdown,
            'tracking_error': results.tracking_error,
            'total_trades': results.total_trades
        }
    }
    
    results_file = f"{BacktestParams.RESULTS_DIR}/backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    print(f"Results saved to {results_file}")
    
    # Create visualizations
    plot_file = results_file.replace('.json', '.png')
    backtester.plot_results(results, save_path=plot_file)
    
    # Save trade history
    trades_df = pd.DataFrame([asdict(t) for t in results.trades])
    trades_file = results_file.replace('.json', '_trades.csv')
    trades_df.to_csv(trades_file, index=False)
    print(f"Trades saved to {trades_file}")

if __name__ == '__main__':
    main()