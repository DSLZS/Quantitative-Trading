"""
V41 Core Engine - 核心引擎模块

【架构设计】
主循环严格控制在 200 行以内，所有逻辑委托给模块化组件

【核心组件】
1. DataLoader - 数据加载
2. FactorLibrary - 因子计算
3. RiskManager - 风险管理
4. 综合信号处理

【V41 特性】
1. 二阶导动量因子
2. 板块中性化
3. 动态风险暴露（低波动时 0.8%）

作者：量化系统
版本：V41.0
日期：2026-03-19
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, date
import numpy as np
import polars as pl
from loguru import logger

from .data_loader import DataLoader, DataLoaderConfig
from .risk_manager import RiskManager, RiskManagerConfig, Position
from .factor_library import FactorLibrary, FactorConfig


class V41Engine:
    """
    V41 核心引擎 - 模块化设计
    
    主循环严格控制在 200 行以内
    """
    
    INITIAL_CAPITAL = 100000.00
    MAX_POSITIONS = 8
    MAX_SINGLE_POSITION_PCT = 0.15
    BASE_RISK_PER_POSITION = 0.005
    LOW_VOL_RISK_PER_POSITION = 0.008
    VOLATILITY_THRESHOLD = 1.0
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2026-01-01')
        self.initial_capital = self.config.get('initial_capital', self.INITIAL_CAPITAL)
        
        self.data_loader = DataLoader(DataLoaderConfig())
        
        self.factor_library = FactorLibrary(FactorConfig(
            industry_neutralization_enabled=True
        ))
        
        self.risk_manager = RiskManager(
            config=RiskManagerConfig(
                base_risk_target=self.BASE_RISK_PER_POSITION,
                enhanced_risk_target=self.LOW_VOL_RISK_PER_POSITION,
                max_positions=self.MAX_POSITIONS,
                low_volatility_threshold=self.VOLATILITY_THRESHOLD,
            ),
            initial_capital=self.initial_capital
        )
        
        self.positions: Dict[str, Position] = {}
        self.daily_trades = []
        self.daily_portfolio_values = []
        self._data_cache = {}
        
        logger.info(f"V41 Engine initialized with capital: {self.initial_capital}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测 - 主循环（严格<200 行）"""
        try:
            logger.info("=" * 60)
            logger.info("V41 BACKTEST STARTING")
            logger.info("=" * 60)
            
            self._load_all_data()
            trading_dates = self._get_trading_dates()
            
            for current_date in trading_dates:
                self._run_trading_day(current_date)
            
            return self._generate_report()
            
        except Exception as e:
            logger.error(f"V41 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _load_all_data(self):
        """加载数据"""
        logger.info("Loading all data...")
        self._data_cache = self.data_loader.load_all_data()
        logger.info(f"Data loaded: {len(self._data_cache.get('price_data', []))} price records")
        logger.info(f"Symbols: {self._data_cache.get('price_data', pl.DataFrame()).get_column('symbol').unique().to_list()[:10]}...")
    
    def _get_trading_dates(self) -> List[date]:
        """获取交易日期列表"""
        if 'price_data' not in self._data_cache or self._data_cache['price_data'].is_empty():
            return []
        dates = self._data_cache['price_data']['trade_date'].unique()
        return sorted([d if isinstance(d, date) else datetime.strptime(str(d), '%Y-%m-%d').date() 
                       for d in dates])
    
    def _run_trading_day(self, current_date: date):
        """执行单日交易逻辑"""
        try:
            date_str = current_date.strftime('%Y-%m-%d')
            price_df = self._data_cache.get('price_data', pl.DataFrame())
            index_df = self._data_cache.get('index_data', pl.DataFrame())
            industry_df = self._data_cache.get('industry_data', pl.DataFrame())
            
            if price_df.is_empty():
                return
            
            daily_prices = price_df.filter(pl.col('trade_date') == date_str)
            if daily_prices.is_empty():
                return
            
            # 更新 risk_manager 的交易日计数器
            self.risk_manager.reset_daily_counters(date_str)
            
            factor_df = self.factor_library.compute_all_factors(daily_prices, industry_df)
            
            portfolio_value = self.risk_manager.get_portfolio_value(self.positions, date_str, price_df)
            market_vol = self._get_market_volatility(factor_df)
            
            self.risk_manager.update_volatility_regime(market_vol)
            risk_per_position = self.risk_manager.get_risk_per_position()
            
            # 检查止损
            sell_candidates = self.risk_manager.check_stop_loss(
                self.positions, date_str, price_df, factor_df
            )
            
            for symbol in sell_candidates:
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    exit_price = self._get_price_for_symbol(daily_prices, symbol)
                    if exit_price and pos.shares > 0:
                        self.risk_manager.close_position(
                            pos, date_str, exit_price, 'stop_loss'
                        )
                        self.daily_trades.append({
                            'trade_date': date_str,
                            'symbol': symbol,
                            'action': 'SELL',
                            'reason': 'stop_loss',
                            'shares': pos.shares,
                            'price': exit_price
                        })
                        del self.positions[symbol]
            
            # 买入逻辑
            max_positions = self.MAX_POSITIONS
            if len(self.positions) < max_positions:
                buy_candidates = self.risk_manager.rank_candidates(factor_df, self.positions)
                available_slots = max_positions - len(self.positions)
                
                for candidate in buy_candidates[:available_slots]:
                    symbol = candidate['symbol']
                    if symbol not in self.positions:
                        entry_price = self._get_price_for_symbol(daily_prices, symbol)
                        if entry_price:
                            position = self.risk_manager.open_position(
                                symbol, date_str, entry_price, 
                                portfolio_value, risk_per_position
                            )
                            if position and position.shares > 0:
                                self.positions[symbol] = position
                                self.daily_trades.append({
                                    'trade_date': date_str,
                                    'symbol': symbol,
                                    'action': 'BUY',
                                    'reason': 'signal',
                                    'shares': position.shares,
                                    'price': entry_price,
                                    'rank': candidate.get('rank', 999)
                                })
            
            # 更新组合价值（包含当前持仓）
            current_portfolio_value = self.risk_manager.get_portfolio_value(self.positions, date_str, price_df)
            self.daily_portfolio_values.append({
                'trade_date': date_str,
                'portfolio_value': current_portfolio_value,
                'positions_count': len(self.positions),
                'market_volatility': market_vol,
                'risk_per_position': risk_per_position
            })
            
        except Exception as e:
            logger.error(f"Error in trading day {current_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _get_price_for_symbol(self, df: pl.DataFrame, symbol: str) -> Optional[float]:
        """获取股票价格"""
        try:
            row = df.filter(pl.col('symbol') == symbol).select('close').row(0)
            return float(row[0]) if row else None
        except Exception:
            return None
    
    def _get_market_volatility(self, df: pl.DataFrame) -> float:
        """获取市场波动率"""
        try:
            vol_col = 'volatility_ratio' if 'volatility_ratio' in df.columns else 'vix_sim'
            if vol_col in df.columns:
                return float(df[vol_col].mean() or 1.0)
            return 1.0
        except Exception:
            return 1.0
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        try:
            total_trades = len(self.daily_trades)
            buy_trades = [t for t in self.daily_trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.daily_trades if t['action'] == 'SELL']
            
            portfolio_values = [p['portfolio_value'] for p in self.daily_portfolio_values]
            final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            max_value = max(portfolio_values) if portfolio_values else self.initial_capital
            min_value = min(portfolio_values) if portfolio_values else self.initial_capital
            max_drawdown = (max_value - min_value) / max_value if max_value > 0 else 0
            
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    daily_returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
            
            if daily_returns:
                daily_returns_np = np.array(daily_returns)
                sharpe_ratio = float(daily_returns_np.mean() / (daily_returns_np.std() + 1e-9)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            return {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'buy_trades': len(buy_trades),
                'sell_trades': len(sell_trades),
                'daily_portfolio_values': self.daily_portfolio_values,
                'trades': self.daily_trades,
                'version': 'V41'
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                'initial_capital': self.initial_capital,
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'error': str(e)
            }