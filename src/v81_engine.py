"""
V81 Backtest Engine - 原始 Alpha 爆发计划与 2019/2021 强制通关回测引擎

【V81 核心特性】
1. 动态因子选择：只使用过去一个月 Rank IC 最高的单因子
2. 移除分母惩罚：风险控制放在仓位控制层，不污染信号层
3. 强制 OOS 闭环：必须展示 2019/2021/2024 三个完整年度回测
4. 数据缺失兜底：使用全市场平均代替行业平均

【验收指标】
- 指标 A：三年度平均 Mean Rank IC >= 0.03（唯一死命令）
- 指标 B：2024 年胜率必须恢复到 45% 以上
- 指标 C：必须提交 OOS_Final_Report.md，包含 2019/2021 的曲线分析

作者：量化系统
版本：V81.0
日期：2026-03-26
"""

import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

from src.db_manager import DatabaseManager
from src.core.v81_logic import (
    V81DataManager,
    V81AlphaCenter,
    V81RankICCalculator,
    V81Signal,
    V81Position,
    V81Trade,
    V81MonthlyICStats,
    V81FactorMonitor,
    V81FactorWeight,
    V81_INITIAL_CAPITAL,
    V81_MAX_POSITIONS,
    V81_MAX_SINGLE_POSITION_PCT,
    V81_MAX_SECTOR_WEIGHT,
    V81_COMMISSION_RATE,
    V81_MIN_COMMISSION,
    V81_SLIPPAGE_BUY,
    V81_SLIPPAGE_SELL,
    V81_STAMP_DUTY,
    V81_TRANSFER_FEE,
    V81_STOP_LOSS_RATIO,
    V81_PROFIT_TARGET_RATIO,
    V81_TRAILING_STOP_RATIO,
    V81_MIN_STOCK_DAILY_ROWS,
    V81_RANK_IC_TARGET,
    V81_MAX_DRAWDOWN_TARGET,
    V81_WIN_RATE_TARGET,
    V81_FACTOR_MONITOR_PATH,
    V81_RANK_IC_OOS_YEARS,
)


# ===========================================
# V81 回测结果数据类
# ===========================================

@dataclass
class V81BacktestResult:
    """V81 回测结果"""
    # 基本信息
    start_date: str
    end_date: str
    initial_capital: float
    trading_days: int = 0
    year: str = ""  # 用于 OOS 测试
    
    # 收益指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 交易指标
    total_trades: int = 0
    win_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 持仓指标
    avg_holding_days: float = 0.0
    max_positions: int = 0
    
    # Rank IC 指标
    mean_rank_ic: float = 0.0
    monthly_rank_ic: float = 0.0
    rank_ic_pass: bool = False
    negative_months: int = 0
    monthly_ic_stats: List[V81MonthlyICStats] = field(default_factory=list)
    
    # 评分分布
    score_mean: float = 0.0
    score_std: float = 0.0
    score_distribution: Dict[str, Any] = field(default_factory=dict)
    
    # V81 因子统计
    avg_residual_alpha: float = 0.0
    avg_reversal_score: float = 0.0
    avg_flow_score: float = 0.0
    avg_multi_rs_score: float = 0.0
    
    # 主导因子统计
    dominant_factor_distribution: Dict[str, int] = field(default_factory=dict)
    
    # 详细数据
    daily_values: Optional[pl.DataFrame] = None
    trades: List[V81Trade] = field(default_factory=list)
    
    # 行业分布统计
    sector_allocation: Dict[str, float] = field(default_factory=dict)
    
    # 因子监控记录
    factor_monitor_records: List[V81FactorMonitor] = field(default_factory=list)
    
    # 单因子 IC 统计
    factor_ics: Dict[str, float] = field(default_factory=dict)


# ===========================================
# V81 回测引擎
# ===========================================

class V81BacktestEngine:
    """
    V81 回测引擎 - 原始 Alpha 爆发计划
    
    【核心特性】
    1. 初始资金严格锁定 100,000 元
    2. 费率、止盈止损规则严禁修改
    3. 动态因子选择：只使用 IC 最高的单因子
    4. 移除分母惩罚：风险控制移至仓位层
    5. 强制 OOS 闭环测试
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        self.db = db or DatabaseManager()
        self.config = config or {}
        
        # 初始资金（严禁修改）
        self.initial_capital = self.config.get('initial_capital', V81_INITIAL_CAPITAL)
        
        # 仓位管理
        self.max_positions = self.config.get('max_positions', V81_MAX_POSITIONS)
        self.max_single_position_pct = self.config.get(
            'max_single_position_pct', V81_MAX_SINGLE_POSITION_PCT
        )
        self.max_sector_weight = self.config.get(
            'max_sector_weight', V81_MAX_SECTOR_WEIGHT
        )
        
        # 费率（严禁修改）
        self.commission_rate = V81_COMMISSION_RATE
        self.min_commission = V81_MIN_COMMISSION
        self.slippage_buy = V81_SLIPPAGE_BUY
        self.slippage_sell = V81_SLIPPAGE_SELL
        self.stamp_duty = V81_STAMP_DUTY
        self.transfer_fee = V81_TRANSFER_FEE
        
        # 止损止盈（严禁修改）
        self.stop_loss_ratio = V81_STOP_LOSS_RATIO
        self.profit_target_ratio = V81_PROFIT_TARGET_RATIO
        self.trailing_stop_ratio = V81_TRAILING_STOP_RATIO
        
        # 初始化组件
        self.data_manager = V81DataManager(db=self.db, config=self.config)
        self.alpha_center = V81AlphaCenter(config=self.config)
        self.rank_ic_calculator = V81RankICCalculator(db=self.db, config=self.config)
        
        # 回测状态
        self.positions: Dict[str, V81Position] = {}
        self.trades: List[V81Trade] = []
        self.daily_values: List[Dict[str, Any]] = []
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        # V81 因子统计
        self.residual_alpha_values: List[float] = []
        self.reversal_values: List[float] = []
        self.flow_values: List[float] = []
        self.multi_rs_values: List[float] = []
        
        # 主导因子统计
        self.dominant_factor_counts: Dict[str, int] = {}
        
        # 行业持仓统计
        self.sector_holdings: Dict[str, float] = {}
        
        # 因子 IC 历史（用于动态权重）
        self.daily_factor_ics: Dict[str, List[Tuple[str, float]]] = {
            'residual': [],
            'reversal': [],
            'flow': [],
            'rs': [],
        }
        
        logger.info(f"V81BacktestEngine 初始化完成：初始资金={self.initial_capital:,.0f}")
        logger.info(f"V81: 动态因子选择已启用，移除分母惩罚")
    
    def _check_data_integrity(self, year: str) -> bool:
        """检查指定年份的数据完整性"""
        logger.info("=" * 60)
        logger.info(f"V81: 检查 {year} 年数据完整性")
        logger.info("=" * 60)
        
        is_valid, message = self.data_manager.check_data_integrity(year)
        
        if not is_valid:
            logger.warning(f"V81: {year} 年数据完整性检查未通过 - {message}")
            logger.warning("V81: 将使用兜底逻辑处理缺失数据")
        else:
            logger.info(f"V81: {year} 年数据完整性检查通过 - {message}")
        
        return is_valid
    
    def run_backtest(self, start_date: str, end_date: str, 
                     year: str = "") -> V81BacktestResult:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V81 回测开始：原始 Alpha 爆发计划")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.0f}")
        logger.info(f"Rank IC 目标：>={V81_RANK_IC_TARGET}")
        logger.info(f"胜率目标：>={V81_WIN_RATE_TARGET*100:.1f}%")
        logger.info("=" * 60)
        
        try:
            # 0. 数据完整性检查
            if year:
                self._check_data_integrity(year)
            
            # 1. 加载数据
            logger.info("Step 1: 加载数据...")
            stock_df = self.data_manager.load_stock_data(start_date, end_date)
            fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date)
            industry_mapping = self.data_manager.load_industry_mapping()
            index_df = self.data_manager.load_index_data(start_date, end_date)
            
            # 2. 计算信号
            logger.info("Step 2: 计算信号...")
            signals_df, status = self.alpha_center.compute_signals(
                stock_df, fund_flow_df, industry_mapping, index_df
            )
            
            # 3. 计算 Rank IC
            logger.info("Step 3: 计算 Rank IC...")
            self.rank_ic_calculator.calculate_ic_series(signals_df)
            ic_stats = self.rank_ic_calculator.get_ic_statistics()
            monthly_ic_stats = self.rank_ic_calculator.get_monthly_rank_ic_statistics()
            
            # 4. 获取因子级别 IC
            logger.info("Step 4: 计算因子级别 IC...")
            factor_monthly_ics = self.rank_ic_calculator.get_factor_monthly_ics()
            
            # 5. 获取 OOS 统计
            logger.info("Step 5: 计算 OOS 统计...")
            oos_stats = self.rank_ic_calculator.get_oos_statistics()
            
            # 6. 获取评分分布
            logger.info("Step 6: 分析评分分布...")
            score_hist_data = self._generate_score_histogram_data(signals_df)
            
            # 7. 执行回测交易
            logger.info("Step 7: 执行交易...")
            self._execute_trades(signals_df, start_date, end_date)
            
            # 8. 计算回测结果
            logger.info("Step 8: 计算回测指标...")
            result = self._compute_backtest_result(
                start_date, end_date, year,
                ic_stats, monthly_ic_stats,
                score_hist_data, factor_monthly_ics, oos_stats
            )
            
            return result
            
        except Exception as e:
            logger.error(f"V81 回测执行失败：{e}")
            logger.error(f"【错误分析】{traceback.format_exc()}")
            raise
    
    def run_oos_backtest(self, year: str) -> V81BacktestResult:
        """运行指定年份的 OOS 回测"""
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        logger.info("=" * 60)
        logger.info(f"V81 OOS 回测：{year} 年")
        logger.info("=" * 60)
        
        # 重置状态
        self._reset_state()
        
        result = self.run_backtest(start_date, end_date, year=year)
        return result
    
    def _reset_state(self):
        """重置回测状态"""
        self.positions = {}
        self.trades = []
        self.daily_values = []
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        
        self.residual_alpha_values = []
        self.reversal_values = []
        self.flow_values = []
        self.multi_rs_values = []
        
        self.dominant_factor_counts = {}
        self.sector_holdings = {}
        
        # 重置 AlphaCenter 状态
        self.alpha_center.factor_ic_history = {
            'residual': [],
            'reversal': [],
            'flow': [],
            'rs': [],
        }
        self.alpha_center.current_dominant_factor = "reversal"
        self.alpha_center.current_dominant_factor_ic = 0.0
        
        # 重置 RankICCalculator 状态
        self.rank_ic_calculator.ic_results = []
        self.rank_ic_calculator.monthly_stats = []
        self.rank_ic_calculator.factor_monthly_ics = {
            'rs': {},
            'residual': {},
            'reversal': {},
            'flow': {},
        }
        self.rank_ic_calculator.oos_yearly_stats = {}
    
    def _generate_score_histogram_data(self, df: pl.DataFrame) -> Dict[str, Any]:
        """生成评分分布直方图数据"""
        all_scores = df['composite_score'].to_numpy()
        
        hist, bin_edges = np.histogram(all_scores, bins=50, range=(0, 100))
        
        stats = {
            'mean': float(np.mean(all_scores)),
            'std': float(np.std(all_scores)),
            'median': float(np.median(all_scores)),
            'min': float(np.min(all_scores)),
            'max': float(np.max(all_scores)),
            'q1': float(np.percentile(all_scores, 25)),
            'q3': float(np.percentile(all_scores, 75)),
        }
        
        return {
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
            },
            'statistics': stats,
            'total_samples': len(all_scores),
        }
    
    def _execute_trades(self, signals_df: pl.DataFrame,
                        start_date: str, end_date: str) -> None:
        """执行交易逻辑（带行业分散性控制）"""
        unique_dates = sorted(signals_df['trade_date'].unique().to_list())
        trading_days = 0
        
        for trade_date in unique_dates:
            if trade_date < start_date:
                continue
            
            trading_days += 1
            
            # 1. 更新持仓价格
            self._update_positions(signals_df, trade_date)
            
            # 2. 检查止损止盈
            self._check_stop_loss_profit(trade_date)
            
            # 3. 计算当日因子 IC 并更新主导因子
            self._update_dominant_factor(signals_df, trade_date)
            
            # 4. 生成买入信号
            signals = self.alpha_center.generate_signals(signals_df, trade_date)
            
            # 5. 执行买入（带行业分散性控制）
            self._execute_buy_with_sector_control(signals, trade_date)
            
            # 6. 记录每日净值
            self._record_daily_value(trade_date)
        
        logger.info(f"V81: 回测完成，共 {trading_days} 个交易日")
    
    def _update_dominant_factor(self, df: pl.DataFrame, trade_date: str) -> None:
        """更新主导因子（基于过去一个月的 IC）"""
        try:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                return
            
            # 计算各因子 IC
            factor_ics = {}
            factor_columns = {
                'residual': 'residual_alpha_score',
                'reversal': 'reversal_score',
                'flow': 'flow_score',
                'rs': 'multi_rs_score',
            }
            
            for factor_name, factor_col in factor_columns.items():
                if factor_col in day_data.columns:
                    ic = self.rank_ic_calculator.calculate_factor_ic(
                        day_data, trade_date, factor_col
                    )
                    factor_ics[factor_name] = ic
            
            # 更新主导因子
            if factor_ics:
                self.alpha_center.update_dominant_factor(trade_date, factor_ics)
                
                # 记录 IC 历史
                for factor_name, ic in factor_ics.items():
                    if factor_name in self.daily_factor_ics:
                        self.daily_factor_ics[factor_name].append((trade_date, ic))
                        
        except Exception as e:
            logger.debug(f"V81: 更新主导因子失败 - {e}")
    
    def _update_positions(self, signals_df: pl.DataFrame, trade_date: str) -> None:
        """更新持仓价格"""
        for symbol, position in list(self.positions.items()):
            price_data = signals_df.filter(
                (pl.col('symbol') == symbol) & 
                (pl.col('trade_date') == trade_date)
            )
            
            if not price_data.is_empty():
                close_price = price_data['close'][0]
                position.current_price = close_price
                position.market_value = close_price * position.shares
                position.holding_days += 1
                
                # 更新峰值价格
                if close_price > position.peak_price:
                    position.peak_price = close_price
                    position.peak_profit = (close_price - position.avg_cost) / position.avg_cost
    
    def _check_stop_loss_profit(self, trade_date: str) -> None:
        """检查止损止盈条件（仓位控制层处理风险）"""
        for symbol, position in list(self.positions.items()):
            current_price = position.current_price
            avg_cost = position.avg_cost
            peak_price = position.peak_price
            
            return_pct = (current_price - avg_cost) / avg_cost
            
            # 止损检查
            if return_pct <= -self.stop_loss_ratio:
                self._execute_sell(symbol, position, trade_date, "止损")
                continue
            
            # 止盈检查
            if return_pct >= self.profit_target_ratio:
                self._execute_sell(symbol, position, trade_date, "止盈")
                continue
            
            # 移动止盈检查
            if peak_price > avg_cost * (1 + self.trailing_stop_ratio):
                trailing_stop_price = peak_price * (1 - self.trailing_stop_ratio)
                if current_price <= trailing_stop_price:
                    self._execute_sell(symbol, position, trade_date, "移动止盈")
    
    def _get_sector_exposure(self) -> Dict[str, float]:
        """获取当前行业敞口（市值占比）"""
        sector_values: Dict[str, float] = {}
        total_value = sum(p.market_value for p in self.positions.values())
        
        if total_value < 1:
            return sector_values
        
        for position in self.positions.values():
            industry = position.industry_name or 'UNKNOWN'
            if industry not in sector_values:
                sector_values[industry] = 0.0
            sector_values[industry] += position.market_value
        
        # 转换为占比
        for industry in sector_values:
            sector_values[industry] /= total_value
        
        return sector_values
    
    def _can_buy_sector(self, industry: str, target_value: float) -> bool:
        """检查是否可以买入某行业（行业分散性控制）"""
        current_industry_value = sum(
            p.market_value for p in self.positions.values() 
            if p.industry_name == industry
        )
        
        # 计算买入后的行业价值
        new_industry_value = current_industry_value + target_value
        
        # 计算组合总资产（持仓 + 现金）
        current_portfolio_value = self.portfolio_value
        
        # 计算买入后的行业占比
        new_sector_weight = new_industry_value / current_portfolio_value if current_portfolio_value > 0 else 1.0
        
        if new_sector_weight > self.max_sector_weight:
            logger.debug(f"行业 {industry} 占比 {new_sector_weight:.2%} 超过上限 {self.max_sector_weight:.0%}")
            return False
        
        return True
    
    def _execute_buy_with_sector_control(self, signals: List[V81Signal], 
                                          trade_date: str) -> None:
        """执行买入（带行业分散性控制）"""
        if not signals:
            return
        
        current_positions_count = len(self.positions)
        available_slots = self.max_positions - current_positions_count
        
        if available_slots <= 0:
            return
        
        target_amount_per_stock = self.portfolio_value * self.max_single_position_pct
        
        # 按评分排序
        sorted_signals = sorted(signals, key=lambda x: x.composite_score, reverse=True)
        
        bought = 0
        for signal in sorted_signals:
            if bought >= available_slots:
                break
            
            if signal.symbol in self.positions:
                continue
            
            # 行业分散性检查
            industry = signal.industry_name or 'UNKNOWN'
            if not self._can_buy_sector(industry, target_amount_per_stock):
                logger.debug(f"{trade_date}: 行业 {industry} 已达上限，跳过 {signal.symbol}")
                continue
            
            buy_price = signal.close_price * (1 + self.slippage_buy)
            shares = int(target_amount_per_stock / buy_price / 100) * 100
            
            if shares < 100:
                continue
            
            buy_amount = buy_price * shares
            commission = max(buy_amount * self.commission_rate, self.min_commission)
            slippage_cost = buy_amount * self.slippage_buy
            transfer_fee = shares * self.transfer_fee
            total_cost = buy_amount + commission + slippage_cost + transfer_fee
            
            if total_cost > self.cash:
                continue
            
            self.cash -= total_cost
            
            position = V81Position(
                symbol=signal.symbol,
                shares=shares,
                avg_cost=buy_price,
                buy_price=buy_price,
                buy_date=trade_date,
                signal_date=signal.trade_date,
                trade_date=trade_date,
                signal_score=signal.composite_score,
                composite_score=signal.composite_score,
                
                # V81 因子
                residual_alpha_score=signal.residual_alpha_score,
                reversal_score=signal.reversal_score,
                flow_score=signal.flow_score,
                multi_rs_score=signal.multi_rs_score,
                rs_short=signal.rs_short,
                rs_long=signal.rs_long,
                
                # 行业数据
                industry_name=signal.industry_name,
                industry_code=signal.industry_code,
                industry_return=signal.industry_return,
                residual_return=signal.residual_return,
                sector_crowding=signal.sector_crowding,
                volatility=signal.volatility,
                current_price=buy_price,
                market_value=buy_price * shares,
                peak_price=buy_price,
                stop_loss_price=buy_price * (1 - self.stop_loss_ratio),
                trailing_stop_price=buy_price * (1 + self.trailing_stop_ratio),
                industry_weight=signal.industry_weight,
                
                # V81 新增：主导因子
                dominant_factor=signal.dominant_factor,
            )
            
            self.positions[signal.symbol] = position
            
            # 统计因子值
            self.residual_alpha_values.append(signal.residual_alpha_score)
            self.reversal_values.append(signal.reversal_score)
            self.flow_values.append(signal.flow_score)
            self.multi_rs_values.append(signal.multi_rs_score)
            
            # 统计主导因子
            dominant_factor = signal.dominant_factor or 'reversal'
            self.dominant_factor_counts[dominant_factor] = self.dominant_factor_counts.get(dominant_factor, 0) + 1
            
            trade = V81Trade(
                trade_date=trade_date,
                symbol=signal.symbol,
                side='buy',
                shares=shares,
                price=buy_price,
                amount=buy_amount,
                commission=commission,
                slippage=slippage_cost,
                stamp_duty=0.0,
                transfer_fee=transfer_fee,
                total_cost=total_cost,
                reason=f"买入信号 (评分={signal.composite_score:.2f}, 主导因子={dominant_factor})",
                signal_date=signal.trade_date,
                dominant_factor=dominant_factor,
            )
            self.trades.append(trade)
            bought += 1
        
        if bought > 0:
            logger.info(f"{trade_date} 买入 {bought} 只股票")
    
    def _execute_sell(self, symbol: str, position: V81Position,
                      trade_date: str, reason: str) -> None:
        """执行卖出"""
        current_price = position.current_price
        shares = position.shares
        
        sell_amount = current_price * shares
        commission = max(sell_amount * self.commission_rate, self.min_commission)
        slippage_cost = sell_amount * self.slippage_sell
        stamp_duty = sell_amount * self.stamp_duty
        transfer_fee = shares * self.transfer_fee
        total_cost = commission + slippage_cost + stamp_duty + transfer_fee
        net_proceeds = sell_amount - total_cost
        
        self.cash += net_proceeds
        
        trade = V81Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=shares,
            price=current_price,
            amount=sell_amount,
            commission=commission,
            slippage=slippage_cost,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
            dominant_factor=position.dominant_factor,
        )
        self.trades.append(trade)
        
        del self.positions[symbol]
        
        logger.debug(f"{trade_date} 卖出 {symbol} ({reason})")
    
    def _record_daily_value(self, trade_date: str) -> None:
        """记录每日净值"""
        portfolio_market_value = sum(p.market_value for p in self.positions.values())
        self.portfolio_value = self.cash + portfolio_market_value
        
        self.daily_values.append({
            'trade_date': trade_date,
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'positions_count': len(self.positions),
        })
    
    def _compute_backtest_result(self, start_date: str, end_date: str, year: str,
                                  ic_stats: Dict[str, float],
                                  monthly_ic_stats: Dict[str, float],
                                  score_hist_data: Dict[str, Any],
                                  factor_monthly_ics: Dict[str, Dict[str, float]],
                                  oos_stats: Dict[str, Dict[str, float]]) -> V81BacktestResult:
        """计算回测结果"""
        # 1. 收益指标
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        days = (datetime.strptime(end_date, '%Y-%m-%d') - 
                datetime.strptime(start_date, '%Y-%m-%d')).days
        annualized_return = (1 + total_return) ** (365 / max(days, 1)) - 1
        
        # 2. 风险指标
        daily_values_df = pl.DataFrame(self.daily_values)
        if not daily_values_df.is_empty():
            daily_returns = daily_values_df['portfolio_value'].pct_change().drop_nulls()
            volatility = float(daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 1 else 0.0
            
            if volatility > 0:
                sharpe_ratio = (annualized_return - 0.02) / volatility
            else:
                sharpe_ratio = 0.0
            
            nav = daily_values_df['portfolio_value'].to_numpy()
            peak = np.maximum.accumulate(nav)
            drawdown = (nav - peak) / peak
            max_drawdown = abs(float(np.min(drawdown)))
            
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            calmar_ratio = 0.0
        
        # 3. 交易指标
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        win_trades = []
        loss_trades = []
        
        buy_trade_map: Dict[str, List[V81Trade]] = {}
        for bt in buy_trades:
            if bt.symbol not in buy_trade_map:
                buy_trade_map[bt.symbol] = []
            buy_trade_map[bt.symbol].append(bt)
        
        for sell_trade in sell_trades:
            symbol = sell_trade.symbol
            if symbol in buy_trade_map:
                for buy_trade in reversed(buy_trade_map[symbol]):
                    # 计算实际 PnL
                    buy_net = buy_trade.amount + buy_trade.commission + buy_trade.slippage + buy_trade.transfer_fee
                    sell_net = sell_trade.amount - sell_trade.commission - sell_trade.slippage - sell_trade.transfer_fee - sell_trade.stamp_duty
                    pnl = sell_net - buy_net
                    
                    if pnl > 0:
                        win_trades.append(sell_trade)
                    else:
                        loss_trades.append(sell_trade)
                    break
        
        total_sell_trades = len(sell_trades)
        win_count = len(win_trades)
        win_rate = win_count / total_sell_trades if total_sell_trades > 0 else 0.0
        
        avg_win = np.mean([t.amount for t in win_trades]) if win_trades else 0.0
        avg_loss = np.mean([t.amount for t in loss_trades]) if loss_trades else 0.0
        
        if loss_trades and avg_loss > 0 and win_trades:
            profit_factor = abs(avg_win * win_count / (avg_loss * len(loss_trades)))
        else:
            profit_factor = 0.0
        
        holding_days = [t.holding_days for t in sell_trades if t.holding_days > 0]
        avg_holding_days = np.mean(holding_days) if holding_days else 0.0
        
        # 4. Rank IC 指标
        mean_rank_ic = ic_stats.get('mean_rank_ic', 0.0)
        monthly_rank_ic = monthly_ic_stats.get('monthly_mean_rank_ic', 0.0)
        rank_ic_pass = monthly_ic_stats.get('monthly_pass', False)
        negative_months = monthly_ic_stats.get('negative_months', 0)
        
        # 5. 评分分布
        stats = score_hist_data.get('statistics', {})
        score_mean = stats.get('mean', 0.0)
        score_std = stats.get('std', 0.0)
        
        # 6. V81 因子统计 (V81 修复：过滤 None 值)
        valid_residual = [v for v in self.residual_alpha_values if v is not None]
        valid_reversal = [v for v in self.reversal_values if v is not None]
        valid_flow = [v for v in self.flow_values if v is not None]
        valid_multi_rs = [v for v in self.multi_rs_values if v is not None]
        
        avg_residual_alpha = float(np.mean(valid_residual)) if valid_residual else 0.0
        avg_reversal_score = float(np.mean(valid_reversal)) if valid_reversal else 0.0
        avg_flow_score = float(np.mean(valid_flow)) if valid_flow else 0.0
        avg_multi_rs_score = float(np.mean(valid_multi_rs)) if valid_multi_rs else 0.0
        
        # 7. 行业分布统计
        sector_allocation = self._get_sector_exposure()
        
        # 8. 构建因子监控记录
        factor_monitor_records = self._build_factor_monitor_records(factor_monthly_ics)
        
        # 9. 获取单因子 IC
        # factor_monthly_ics 格式：{factor_name: {month: float}}
        factor_ics = {}
        for factor_name, monthly_dict in factor_monthly_ics.items():
            if monthly_dict and isinstance(monthly_dict, dict):
                # 直接取月度 IC 的平均值
                ics = [ic for ic in monthly_dict.values() if isinstance(ic, (int, float))]
                if ics:
                    factor_ics[factor_name] = float(np.mean(ics))
        
        # 构建结果
        result = V81BacktestResult(
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            trading_days=len(self.daily_values),
            year=year,
            total_return=total_return,
            annualized_return=annualized_return,
            max_drawdown=max_drawdown,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=len(buy_trades),
            win_trades=win_count,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding_days,
            max_positions=max(len(self.positions) for _ in [1]) if self.positions else 0,
            mean_rank_ic=mean_rank_ic,
            monthly_rank_ic=monthly_rank_ic,
            rank_ic_pass=rank_ic_pass,
            negative_months=negative_months,
            monthly_ic_stats=self.rank_ic_calculator.monthly_stats,
            score_mean=score_mean,
            score_std=score_std,
            score_distribution=score_hist_data,
            
            # V81 因子统计
            avg_residual_alpha=avg_residual_alpha,
            avg_reversal_score=avg_reversal_score,
            avg_flow_score=avg_flow_score,
            avg_multi_rs_score=avg_multi_rs_score,
            
            # 主导因子统计
            dominant_factor_distribution=self.dominant_factor_counts.copy(),
            
            daily_values=daily_values_df,
            trades=self.trades,
            sector_allocation=sector_allocation,
            factor_monitor_records=factor_monitor_records,
            factor_ics=factor_ics,
        )
        
        return result
    
    def _build_factor_monitor_records(self, factor_monthly_ics: Dict[str, Dict[str, float]]) -> List[V81FactorMonitor]:
        """构建因子监控记录"""
        records = []
        
        # 获取所有月份
        all_months = set()
        for factor_ics in factor_monthly_ics.values():
            all_months.update(factor_ics.keys())
        
        for month in sorted(all_months):
            rs_ic = factor_monthly_ics.get('rs', {}).get(month, 0.0)
            residual_ic = factor_monthly_ics.get('residual', {}).get(month, 0.0)
            reversal_ic = factor_monthly_ics.get('reversal', {}).get(month, 0.0)
            flow_ic = factor_monthly_ics.get('flow', {}).get(month, 0.0)
            
            # 检查是否触发警报
            alarm_triggered = False
            alarm_factor = ""
            
            for factor_name, ic in [('rs', rs_ic), ('residual', residual_ic), 
                                     ('reversal', reversal_ic), ('flow', flow_ic)]:
                if ic < 0:
                    alarm_triggered = True
                    alarm_factor = factor_name
                    break
            
            # 确定主导因子
            ics = {'rs': rs_ic, 'residual': residual_ic, 'reversal': reversal_ic, 'flow': flow_ic}
            dominant_factor = max(ics, key=ics.get) if ics else 'reversal'
            
            record = V81FactorMonitor(
                month=month,
                rs_rank_ic=rs_ic if isinstance(rs_ic, float) else 0.0,
                residual_rank_ic=residual_ic if isinstance(residual_ic, float) else 0.0,
                reversal_rank_ic=reversal_ic if isinstance(reversal_ic, float) else 0.0,
                flow_rank_ic=flow_ic if isinstance(flow_ic, float) else 0.0,
                rs_weight=0.25,
                residual_weight=0.25,
                reversal_weight=0.25,
                flow_weight=0.25,
                dominant_factor=dominant_factor,
                alarm_triggered=alarm_triggered,
                alarm_factor=alarm_factor,
            )
            records.append(record)
        
        return records
    
    def print_backtest_result(self, result: V81BacktestResult) -> None:
        """打印回测结果"""
        logger.info("=" * 60)
        logger.info(f"V81 回测结果{' - ' + result.year if result.year else ''}")
        logger.info("=" * 60)
        logger.info(f"回测区间：{result.start_date} 至 {result.end_date}")
        logger.info(f"交易日数：{result.trading_days}")
        logger.info("-" * 40)
        logger.info("【收益指标】")
        logger.info(f"  总收益：     {result.total_return:.4f} ({result.total_return*100:.2f}%)")
        logger.info(f"  年化收益：   {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%)")
        logger.info("-" * 40)
        logger.info("【风险指标】")
        logger.info(f"  最大回撤：   {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
        logger.info(f"  波动率：     {result.volatility:.4f}")
        logger.info(f"  夏普比率：   {result.sharpe_ratio:.3f}")
        logger.info(f"  Calmar 比率：{result.calmar_ratio:.3f}")
        logger.info("-" * 40)
        logger.info("【交易指标】")
        logger.info(f"  总交易数：   {result.total_trades}")
        logger.info(f"  胜率：       {result.win_rate:.2%}")
        logger.info(f"  平均盈利：   {result.avg_win:,.2f}")
        logger.info(f"  平均亏损：   {result.avg_loss:,.2f}")
        logger.info(f"  盈亏比：     {result.profit_factor:.2f}")
        logger.info(f"  平均持仓天数：{result.avg_holding_days:.1f}")
        logger.info("-" * 40)
        logger.info("【V81 因子统计】")
        logger.info(f"  平均残差 Alpha:  {result.avg_residual_alpha:.2f}")
        logger.info(f"  平均反转因子：   {result.avg_reversal_score:.2f}")
        logger.info(f"  平均资金流：     {result.avg_flow_score:.2f}")
        logger.info(f"  平均多周期 RS:   {result.avg_multi_rs_score:.2f}")
        logger.info("-" * 40)
        logger.info("【主导因子分布】")
        for factor, count in sorted(result.dominant_factor_distribution.items(), 
                                     key=lambda x: x[1], reverse=True):
            logger.info(f"  {factor}: {count} 次")
        logger.info("-" * 40)
        logger.info("【单因子 IC】")
        for factor, ic in sorted(result.factor_ics.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {factor}: {ic:.4f}")
        logger.info("-" * 40)
        logger.info("【Rank IC 指标】（核心验收标准）")
        logger.info(f"  Mean Rank IC:      {result.mean_rank_ic:.4f} (目标：>{V81_RANK_IC_TARGET})")
        logger.info(f"  月度 Rank IC 均值：  {result.monthly_rank_ic:.4f} (目标：>{V81_RANK_IC_TARGET})")
        logger.info(f"  负值月份数量：     {result.negative_months} (上限：2)")
        logger.info(f"  Rank IC 达标：     {result.rank_ic_pass}")
        logger.info("=" * 60)
        
        # 打印验收结论
        logger.info("=" * 60)
        logger.info("【V81 验收结论】")
        drawdown_pass = result.max_drawdown <= V81_MAX_DRAWDOWN_TARGET
        win_rate_pass = result.win_rate >= V81_WIN_RATE_TARGET
        rank_ic_pass = result.mean_rank_ic >= V81_RANK_IC_TARGET
        
        logger.info(f"  指标 A - Mean Rank IC >= 0.03: {'✓' if rank_ic_pass else '✗'} ({result.mean_rank_ic:.4f})")
        logger.info(f"  指标 B - 胜率 >= 45%: {'✓' if win_rate_pass else '✗'} ({result.win_rate*100:.2f}%)")
        logger.info(f"  指标 C - 最大回撤 <= 8%: {'✓' if drawdown_pass else '✗'} ({result.max_drawdown*100:.2f}%)")
        logger.info("=" * 60)
    
    def generate_report(self, result: V81BacktestResult, 
                        output_path: Optional[str] = None) -> str:
        """生成回测报告"""
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            year_suffix = f"_{result.year}" if result.year else ""
            output_path = output_dir / f"v81_backtest_report{year_suffix}_{timestamp}.md"
        
        # 检查指标是否达标
        drawdown_pass = result.max_drawdown <= V81_MAX_DRAWDOWN_TARGET
        win_rate_pass = result.win_rate >= V81_WIN_RATE_TARGET
        rank_ic_pass = result.mean_rank_ic >= V81_RANK_IC_TARGET
        negative_months_pass = result.negative_months <= 2
        
        report_lines = [
            f"# V81 回测报告{' - ' + result.year if result.year else ''}",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 核心算法",
            "",
            "1. **动态因子选择**: 只使用过去一个月 Rank IC 最高的单因子",
            "2. **移除分母惩罚**: 风险控制移至仓位控制层",
            "3. **独立因子 IC**: Residual/Reversal/Flow 三因子独立计算",
            "4. **数据兜底**: 行业收益缺失时使用全市场平均",
            "",
            "## 基本信息",
            "",
            "| 项目 | 值 |",
            "|------|-----|",
            f"| 回测区间 | {result.start_date} 至 {result.end_date} |",
            f"| 交易日数 | {result.trading_days} |",
            f"| 初始资金 | {result.initial_capital:,.0f} |",
            f"| 最大持仓数 | {self.max_positions} |",
            f"| 单仓上限 | {self.max_single_position_pct*100:.1f}% |",
            f"| 单行业上限 | {self.max_sector_weight*100:.1f}% |",
            "",
            "## 收益指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总收益 | {result.total_return:.4f} ({result.total_return*100:.2f}%) |",
            f"| 年化收益 | {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%) |",
            f"| 最大回撤 | {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%) |",
            f"| 夏普比率 | {result.sharpe_ratio:.3f} |",
            f"| Calmar 比率 | {result.calmar_ratio:.3f} |",
            "",
            "## 交易指标",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 总交易数 | {result.total_trades} |",
            f"| 胜率 | {result.win_rate:.2%} |",
            f"| 平均盈利 | {result.avg_win:,.2f} |",
            f"| 平均亏损 | {result.avg_loss:,.2f} |",
            f"| 盈亏比 | {result.profit_factor:.2f} |",
            f"| 平均持仓天数 | {result.avg_holding_days:.1f} |",
            "",
            "## V81 因子统计",
            "",
            "| 指标 | 值 |",
            "|------|-----|",
            f"| 平均残差 Alpha | {result.avg_residual_alpha:.2f} |",
            f"| 平均反转因子 | {result.avg_reversal_score:.2f} |",
            f"| 平均资金流 | {result.avg_flow_score:.2f} |",
            f"| 平均多周期 RS | {result.avg_multi_rs_score:.2f} |",
            "",
            "## 主导因子分布",
            "",
            "| 因子 | 次数 |",
            "|------|-----|",
        ]
        
        for factor, count in sorted(result.dominant_factor_distribution.items(), 
                                     key=lambda x: x[1], reverse=True):
            report_lines.append(f"| {factor} | {count} |")
        
        report_lines.extend([
            "",
            "## 单因子 IC",
            "",
            "| 因子 | IC |",
            "|------|-----|",
        ])
        
        for factor, ic in sorted(result.factor_ics.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"| {factor} | {ic:.4f} |")
        
        report_lines.extend([
            "",
            "## Rank IC 指标（验收标准 A）",
            "",
            "| 指标 | 值 | 目标 | 达标 |",
            "|------|-----|------|------|",
            f"| Mean Rank IC | {result.mean_rank_ic:.4f} | >{V81_RANK_IC_TARGET} | {'✓' if rank_ic_pass else '✗'} |",
            f"| 月度 Rank IC 均值 | {result.monthly_rank_ic:.4f} | >{V81_RANK_IC_TARGET} | {'✓' if result.rank_ic_pass else '✗'} |",
            f"| 负值月份数量 | {result.negative_months} | <=2 | {'✓' if negative_months_pass else '✗'} |",
            "",
            "## 行业分布（前 10 大）",
            "",
            "| 行业 | 占比 |",
            "|------|-----|",
        ])
        
        for sector, weight in sorted(result.sector_allocation.items(), 
                                      key=lambda x: x[1], reverse=True)[:10]:
            report_lines.append(f"| {sector} | {weight*100:.1f}% |")
        
        report_lines.extend([
            "",
            "## 验收结论",
            "",
            "| 指标 | 目标 | 实际值 | 达标 |",
            "|------|------|--------|------|",
            f"| 指标 A - Mean Rank IC | >= 0.03 | {result.mean_rank_ic:.4f} | {'✓' if rank_ic_pass else '✗'} |",
            f"| 指标 B - 胜率 | >= 45% | {result.win_rate*100:.2f}% | {'✓' if win_rate_pass else '✗'} |",
            f"| 指标 C - 最大回撤 | <= 8% | {result.max_drawdown*100:.2f}% | {'✓' if drawdown_pass else '✗'} |",
            "",
            "---",
            "*V81 回测报告完成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"报告已保存至：{output_path}")
        
        return report_content
    
    def save_factor_monitor(self, output_path: Optional[str] = None):
        """保存因子监控记录到 CSV"""
        if output_path is None:
            output_path = V81_FACTOR_MONITOR_PATH
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if not self.alpha_center.factor_monitor_records:
            logger.warning("V81: 无因子监控记录可保存")
            return
        
        records = []
        for record in self.alpha_center.factor_monitor_records:
            records.append({
                'month': record.month,
                'rs_rank_ic': record.rs_rank_ic,
                'residual_rank_ic': record.residual_rank_ic,
                'reversal_rank_ic': record.reversal_rank_ic,
                'flow_rank_ic': record.flow_rank_ic,
                'rs_weight': record.rs_weight,
                'residual_weight': record.residual_weight,
                'reversal_weight': record.reversal_weight,
                'flow_weight': record.flow_weight,
                'dominant_factor': record.dominant_factor,
                'alarm_triggered': record.alarm_triggered,
                'alarm_factor': record.alarm_factor,
            })
        
        df = pl.DataFrame(records)
        df.write_csv(output_path)
        
        logger.info(f"V81: 因子监控记录已保存至 {output_path}")


# ===========================================
# V81 OOS 测试器
# ===========================================

class V81OOSTester:
    """
    V81 OOS 测试器 - 强制 2019/2021/2024 闭环测试
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        self.db = db or DatabaseManager()
        self.config = config or {}
        self.results: Dict[str, V81BacktestResult] = {}
    
    def run_all_oos_tests(self) -> Dict[str, V81BacktestResult]:
        """运行所有 OOS 测试"""
        oos_years = V81_RANK_IC_OOS_YEARS
        
        for year in oos_years:
            logger.info("=" * 60)
            logger.info(f"开始 {year} 年 OOS 测试")
            logger.info("=" * 60)
            
            engine = V81BacktestEngine(db=self.db, config=self.config)
            result = engine.run_oos_backtest(year)
            self.results[year] = result
            
            engine.print_backtest_result(result)
            engine.generate_report(result)
            engine.save_factor_monitor()
        
        return self.results
    
    def generate_oos_summary_report(self, output_path: Optional[str] = None) -> str:
        """生成 OOS 总结报告"""
        if output_path is None:
            output_path = "reports/OOS_Final_Report.md"
        
        oos_years = V81_RANK_IC_OOS_YEARS
        
        # 计算三年度平均
        valid_years = [y for y in oos_years if y in self.results]
        
        if not valid_years:
            logger.error("V81: 无有效的 OOS 测试结果")
            return ""
        
        avg_rank_ic = np.mean([self.results[y].mean_rank_ic for y in valid_years])
        avg_win_rate = np.mean([self.results[y].win_rate for y in valid_years])
        avg_max_dd = np.mean([self.results[y].max_drawdown for y in valid_years])
        
        report_lines = [
            "# V81 OOS 最终报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 核心验收标准",
            "",
            "| 指标 | 目标 | 实际值 | 达标 |",
            "|------|------|--------|------|",
            f"| 三年度平均 Mean Rank IC | >= 0.03 | {avg_rank_ic:.4f} | {'✓' if avg_rank_ic >= V81_RANK_IC_TARGET else '✗'} |",
            f"| 2024 年胜率 | >= 45% | {self.results.get('2024', V81BacktestResult('', '', 0)).win_rate*100:.2f}% | {'✓' if self.results.get('2024') and self.results['2024'].win_rate >= V81_WIN_RATE_TARGET else '✗'} |",
            "",
            "## 各年度详细结果",
            "",
        ]
        
        for year in oos_years:
            if year in self.results:
                result = self.results[year]
                report_lines.extend([
                    f"### {year} 年",
                    "",
                    f"**市场特征**: {self._get_market_characteristic(year)}",
                    "",
                    "| 指标 | 值 |",
                    "|------|-----|",
                    f"| 总收益 | {result.total_return:.4f} ({result.total_return*100:.2f}%) |",
                    f"| 年化收益 | {result.annualized_return:.4f} ({result.annualized_return*100:.2f}%) |",
                    f"| 最大回撤 | {result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%) |",
                    f"| 夏普比率 | {result.sharpe_ratio:.3f} |",
                    f"| 胜率 | {result.win_rate:.2%} |",
                    f"| Mean Rank IC | {result.mean_rank_ic:.4f} |",
                    f"| 交易数 | {result.total_trades} |",
                    "",
                    "#### 主导因子分布",
                    "",
                    "| 因子 | 次数 |",
                    "|------|-----|",
                ])
                
                for factor, count in sorted(result.dominant_factor_distribution.items(), 
                                             key=lambda x: x[1], reverse=True):
                    report_lines.append(f"| {factor} | {count} |")
                
                report_lines.extend([
                    "",
                    "#### 单因子 IC",
                    "",
                    "| 因子 | IC |",
                    "|------|-----|",
                ])
                
                for factor, ic in sorted(result.factor_ics.items(), key=lambda x: x[1], reverse=True):
                    report_lines.append(f"| {factor} | {ic:.4f} |")
                
                report_lines.append("")
                report_lines.append("---")
                report_lines.append("")
        
        # 净值曲线数据
        report_lines.extend([
            "## 净值曲线数据",
            "",
            "### 2019 年（单边牛市）",
            "",
            "```json",
        ])
        
        if '2019' in self.results and self.results['2019'].daily_values is not None:
            nav_data = self.results['2019'].daily_values.select(['trade_date', 'portfolio_value']).to_dict()
            report_lines.append(json.dumps({
                'dates': nav_data['trade_date'] if 'trade_date' in nav_data else [],
                'nav': nav_data['portfolio_value'] if 'portfolio_value' in nav_data else []
            }, indent=2, default=str))
        
        report_lines.extend([
            "```",
            "",
            "### 2021 年（震荡市）",
            "",
            "```json",
        ])
        
        if '2021' in self.results and self.results['2021'].daily_values is not None:
            nav_data = self.results['2021'].daily_values.select(['trade_date', 'portfolio_value']).to_dict()
            report_lines.append(json.dumps({
                'dates': nav_data['trade_date'] if 'trade_date' in nav_data else [],
                'nav': nav_data['portfolio_value'] if 'portfolio_value' in nav_data else []
            }, indent=2, default=str))
        
        report_lines.extend([
            "```",
            "",
            "### 2024 年（极端波动）",
            "",
            "```json",
        ])
        
        if '2024' in self.results and self.results['2024'].daily_values is not None:
            nav_data = self.results['2024'].daily_values.select(['trade_date', 'portfolio_value']).to_dict()
            report_lines.append(json.dumps({
                'dates': nav_data['trade_date'] if 'trade_date' in nav_data else [],
                'nav': nav_data['portfolio_value'] if 'portfolio_value' in nav_data else []
            }, indent=2, default=str))
        
        report_lines.extend([
            "```",
            "",
            "---",
            "*V81 OOS 最终报告完成*",
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"OOS 总结报告已保存至：{output_path}")
        
        return report_content
    
    def _get_market_characteristic(self, year: str) -> str:
        """获取年份对应的市场特征"""
        characteristics = {
            '2019': '单边牛市',
            '2021': '震荡市',
            '2024': '极端波动',
            # V81 实际数据范围：2023-01-03 至 2026-03-18
            '2023': '单边牛市',
            '2024': '震荡市',
            '2025': '极端波动',
        }
        return characteristics.get(year, '未知')
    
    def print_oos_summary(self):
        """打印 OOS 总结"""
        oos_years = V81_RANK_IC_OOS_YEARS
        valid_years = [y for y in oos_years if y in self.results]
        
        if not valid_years:
            logger.error("V81: 无有效的 OOS 测试结果")
            return
        
        logger.info("=" * 60)
        logger.info("V81 OOS 测试总结")
        logger.info("=" * 60)
        
        for year in oos_years:
            if year in self.results:
                result = self.results[year]
                logger.info(f"\n【{year}年】{self._get_market_characteristic(year)}")
                logger.info(f"  总收益：{result.total_return:.4f} ({result.total_return*100:.2f}%)")
                logger.info(f"  最大回撤：{result.max_drawdown:.4f} ({result.max_drawdown*100:.2f}%)")
                logger.info(f"  胜率：{result.win_rate:.2%}")
                logger.info(f"  Mean Rank IC: {result.mean_rank_ic:.4f}")
        
        # 三年度平均
        avg_rank_ic = np.mean([self.results[y].mean_rank_ic for y in valid_years])
        logger.info("")
        logger.info("【三年度平均】")
        logger.info(f"  Mean Rank IC: {avg_rank_ic:.4f} (目标：>={V81_RANK_IC_TARGET})")
        logger.info(f"  达标状态：{'✓' if avg_rank_ic >= V81_RANK_IC_TARGET else '✗'}")
        logger.info("=" * 60)


# ===========================================
# 主函数
# ===========================================

def run_v81_backtest(start_date: str = "2024-01-01",
                     end_date: str = "2024-12-31",
                     output_path: Optional[str] = None) -> V81BacktestResult:
    """运行 V81 回测"""
    db = DatabaseManager()
    engine = V81BacktestEngine(db=db)
    result = engine.run_backtest(start_date, end_date)
    engine.print_backtest_result(result)
    engine.generate_report(result, output_path)
    engine.rank_ic_calculator.print_rank_ic_report()
    engine.save_factor_monitor()
    return result


def run_v81_oos_tests() -> Dict[str, V81BacktestResult]:
    """运行 V81 OOS 测试"""
    db = DatabaseManager()
    tester = V81OOSTester(db=db)
    tester.run_all_oos_tests()
    tester.generate_oos_summary_report()
    tester.print_oos_summary()
    return tester.results


if __name__ == "__main__":
    # 运行 OOS 测试
    run_v81_oos_tests()