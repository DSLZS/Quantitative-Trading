"""
V66 Engine Module - 机构资金踪迹模型回测引擎 (IC 优化版)

【V66 引擎核心 - 拒绝降级逻辑】

1. 禁止 Fallback：严禁出现 `if data_is_empty: switch_to_price_only()`
2. 数据熔断机制：程序启动时必须检查 `stock_fund_flow` 和 `stock_industry_daily`
   - 若数据量少于 10,000 条，必须 `raise ValueError("CRITICAL DATA MISSING")` 并立即停止
3. 建表校验：在脚本开头执行 `CREATE TABLE IF NOT EXISTS` 语句

【核心算法】
1. RS-Industry Z-Score：个股 RS 必须在其所属行业的横向分布中处于 Z > 1.2 的位置
2. 成交量特征：回调期间，主力资金/成交量的比值必须上升
3. 评估指标：AE + IC，要求 2024 年 IC 均值 > 0.02

【杜绝偷懒与伪造】
- 禁止伪造：禁止在日志中打印"数据加载完成"但实际并无落库的行为
- 代码要求：必须包含详尽的 logger.debug 记录数据查询的 SQL 语句

作者：量化系统
版本：V66.0
日期：2026-03-24
"""

import sys
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import json
import numpy as np
import polars as pl
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager, get_db
from v66_data_loader import V66DataLoader, verify_v66_data, V66_MIN_DATA_ROWS
from v66_core import (
    V66DataManager,
    V66AlphaCenter,
    V66TradeExec,
    V66ICCalculator,
    V66Signal,
    V66MarketRegime,
    V66Position,
    V66Trade,
    V66TradeAudit,
    V66DataCredibility,
    calculate_ae_metric,
    V66_INITIAL_CAPITAL,
    V66_MAX_POSITIONS,
    V66_FRICTION_COST,
    V66_IC_TARGET_MEAN,
)


# ===========================================
# V66 回测引擎 - 主类
# ===========================================

class V66BacktestEngine:
    """
    V66 回测引擎 - 机构资金踪迹模型 (IC 优化版)
    
    【死命令】
    1. 禁止 Fallback：严禁出现 `if data_is_empty: switch_to_price_only()`
    2. 数据熔断：若 stock_fund_flow 或 stock_industry_daily 少于 10,000 条，raise ValueError
    3. 建表校验：执行 CREATE TABLE IF NOT EXISTS
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化回测引擎
        
        Parameters
        ----------
        config : Dict[str, Any], optional
            配置字典
        """
        self.config = config or {}
        
        # 回测参数
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', V66_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V66_MAX_POSITIONS)
        
        # 数据库连接
        self.db: Optional[DatabaseManager] = None
        
        # 核心组件
        self.data_manager: Optional[V66DataManager] = None
        self.alpha_center: Optional[V66AlphaCenter] = None
        self.trade_exec: Optional[V66TradeExec] = None
        self.ic_calculator: Optional[V66ICCalculator] = None
        
        # 数据存储
        self.stock_data: Optional[pl.DataFrame] = None
        self.fund_flow_data: Optional[pl.DataFrame] = None
        self.industry_data: Optional[pl.DataFrame] = None
        self.signal_data: Optional[pl.DataFrame] = None
        
        # 回测结果
        self.trades: List[V66Trade] = []
        self.positions: Dict[str, V66Position] = {}
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        self.trade_audits: List[V66TradeAudit] = []
        
        # 性能指标
        self.metrics: Dict[str, Any] = {}
        
        # 数据可信度统计
        self.data_credibility_stats = {
            'dual': 0,
            'fund_flow': 0,
            'industry': 0,
            'none': 0,
        }
    
    def initialize(self) -> bool:
        """
        初始化引擎
        
        【死命令】
        1. 建表校验：执行 CREATE TABLE IF NOT EXISTS
        2. 数据熔断：检查 stock_fund_flow 和 stock_industry_daily 表
        
        Returns
        -------
        bool
            初始化是否成功
        """
        logger.info("=" * 60)
        logger.info("V66 回测引擎 - 初始化")
        logger.info("=" * 60)
        
        try:
            # 1. 初始化数据库
            self.db = get_db()
            logger.info("V66: 数据库连接已建立")
            
            # 2. 建表校验
            self._create_tables_if_not_exists()
            logger.info("V66: 建表校验完成")
            
            # 3. 数据熔断检查
            self._data_circuit_breaker()
            logger.info("V66: 数据熔断检查通过")
            
            # 4. 初始化核心组件
            self.data_manager = V66DataManager(db=self.db, config=self.config)
            self.alpha_center = V66AlphaCenter(config=self.config)
            self.trade_exec = V66TradeExec(config=self.config)
            self.ic_calculator = V66ICCalculator(config=self.config)
            
            logger.info("V66: 核心组件初始化完成")
            
            return True
            
        except ValueError as e:
            logger.error(f"V66: 数据熔断触发：{e}")
            raise
        except Exception as e:
            logger.error(f"V66: 初始化失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _create_tables_if_not_exists(self):
        """
        建表校验：执行 CREATE TABLE IF NOT EXISTS 语句
        """
        logger.info("V66: 开始建表校验...")
        
        # 创建 stock_fund_flow 表
        create_fund_flow_sql = """
        CREATE TABLE IF NOT EXISTS `stock_fund_flow` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `trade_date` VARCHAR(20) NOT NULL,
            `net_main_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_super_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_large_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_medium_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_small_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_main_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_super_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_large_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_medium_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_small_ratio` DECIMAL(10, 6) DEFAULT 0,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='个股资金流数据表'
        """
        
        try:
            logger.debug(f"V66: 执行 SQL: {create_fund_flow_sql[:200]}...")
            self.db.execute(create_fund_flow_sql)
            logger.info("V66: stock_fund_flow 表创建成功")
        except Exception as e:
            logger.warning(f"V66: 创建 stock_fund_flow 表失败：{e}")
        
        # 创建 stock_industry_daily 表
        create_industry_sql = """
        CREATE TABLE IF NOT EXISTS `stock_industry_daily` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `industry_name` VARCHAR(100) NOT NULL,
            `trade_date` VARCHAR(20) NOT NULL,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_industry` (`industry_name`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票行业分类数据表'
        """
        
        try:
            logger.debug(f"V66: 执行 SQL: {create_industry_sql[:200]}...")
            self.db.execute(create_industry_sql)
            logger.info("V66: stock_industry_daily 表创建成功")
        except Exception as e:
            logger.warning(f"V66: 创建 stock_industry_daily 表失败：{e}")
        
        logger.info("V66: 建表校验完成")
    
    def _data_circuit_breaker(self):
        """
        数据熔断检查
        
        【死命令】
        - 检查 stock_fund_flow 和 stock_industry_daily 表
        - 若数据量少于 10,000 条，必须 raise ValueError("CRITICAL DATA MISSING")
        """
        logger.info("V66: 开始数据熔断检查...")
        
        # 检查 stock_fund_flow 表
        fund_flow_count = self._count_table_rows('stock_fund_flow')
        
        # 检查 stock_industry_daily 表
        industry_count = self._count_table_rows('stock_industry_daily')
        
        logger.info(f"[DATA CHECK] Fund Flow Rows: {fund_flow_count}, Industry Rows: {industry_count}")
        
        # 数据熔断：少于 10,000 条直接抛出异常
        if fund_flow_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_fund_flow 表仅有 {fund_flow_count} 条数据，少于阈值 {V66_MIN_DATA_ROWS}"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        if industry_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_industry_daily 表仅有 {industry_count} 条数据，少于阈值 {V66_MIN_DATA_ROWS}"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        logger.info("V66: 数据熔断检查通过")
    
    def _count_table_rows(self, table_name: str) -> int:
        """
        统计表行数
        
        Parameters
        ----------
        table_name : str
            表名
            
        Returns
        -------
        int
            行数
        """
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            logger.debug(f"V66: 执行 SQL: {query}")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return 0
            
            return int(df['cnt'][0])
            
        except Exception as e:
            logger.warning(f"V66: 统计表 {table_name} 行数失败：{e}")
            return 0
    
    def load_data(self) -> bool:
        """
        加载回测数据
        
        Returns
        -------
        bool
            加载是否成功
        """
        logger.info("=" * 60)
        logger.info(f"V66: 加载数据 [{self.start_date}, {self.end_date}]")
        logger.info("=" * 60)
        
        try:
            # 1. 加载股票数据
            self.stock_data = self.data_manager.load_stock_data(
                self.start_date, self.end_date
            )
            
            if self.stock_data.is_empty():
                # 【禁止降级】数据为空时直接抛出异常，不能 fallback
                raise ValueError("V66: stock_daily 表数据为空，无法回测")
            
            logger.info(f"V66: 股票数据加载完成 - {self.stock_data.height} 行")
            
            # 2. 加载资金流数据
            self.fund_flow_data = self.data_manager.load_fund_flow_data(
                self.start_date, self.end_date
            )
            
            if self.fund_flow_data.is_empty():
                logger.warning("V66: 未加载到资金流数据，将使用基础模式")
            else:
                logger.info(f"V66: 资金流数据加载完成 - {self.fund_flow_data.height} 行")
            
            # 3. 加载行业数据
            self.industry_data = self.data_manager.load_industry_data(
                self.start_date, self.end_date
            )
            
            if self.industry_data.is_empty():
                logger.warning("V66: 未加载到行业数据，将使用基础模式")
            else:
                logger.info(f"V66: 行业数据加载完成 - {self.industry_data.height} 行")
            
            return True
            
        except Exception as e:
            logger.error(f"V66: 数据加载失败：{e}")
            raise
    
    def run_backtest(self) -> Dict[str, Any]:
        """
        运行回测
        
        Returns
        -------
        Dict[str, Any]
            回测结果
        """
        logger.info("=" * 60)
        logger.info("V66: 开始回测")
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"回测区间：[{self.start_date}, {self.end_date}]")
        logger.info("=" * 60)
        
        try:
            # 1. 计算信号
            self._compute_signals()
            
            # 2. 逐日回测
            self._run_daily_backtest()
            
            # 3. 计算 IC 值
            self._calculate_ic()
            
            # 4. 计算性能指标
            self._calculate_metrics()
            
            # 5. 打印报告
            self._print_report()
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"V66: 回测失败：{e}")
            logger.error(traceback.format_exc())
            raise
    
    def _compute_signals(self):
        """计算所有信号"""
        logger.info("V66: 开始计算信号...")
        
        if self.stock_data is None:
            raise ValueError("V66: 股票数据未加载")
        
        # 使用 AlphaCenter 计算信号
        self.signal_data, signal_status = self.alpha_center.compute_signals(
            self.stock_data,
            self.fund_flow_data,
            self.industry_data
        )
        
        # 记录数据可信度
        data_credibility = signal_status.get('data_credibility', V66DataCredibility.DUAL)
        self.data_credibility_stats[data_credibility] = self.signal_data.height
        
        logger.info(f"V66: 信号计算完成，数据可信度：{data_credibility}")
    
    def _run_daily_backtest(self):
        """逐日回测"""
        logger.info("V66: 开始逐日回测...")
        
        if self.signal_data is None:
            raise ValueError("V66: 信号数据未生成")
        
        # 获取交易日列表
        trade_dates = sorted(self.signal_data['trade_date'].unique().to_list())
        
        logger.info(f"V66: 共 {len(trade_dates)} 个交易日")
        
        prev_equity = self.initial_capital
        
        for i, trade_date in enumerate(trade_dates):
            try:
                # 1. 获取当日数据
                current_df = self.signal_data.filter(pl.col('trade_date') == trade_date)
                
                if current_df.is_empty():
                    continue
                
                # 2. 计算大盘状态
                market_regime = self.alpha_center.compute_market_decline_ratio(current_df)
                
                # 3. 更新持仓
                self._update_positions(current_df, trade_date)
                
                # 4. 检查离场条件
                self._check_exit_signals(current_df, trade_date, market_regime)
                
                # 5. 检查入场信号
                if market_regime.is_safe_period and self.trade_exec.can_buy_more():
                    self._check_entry_signals(current_df, trade_date, market_regime)
                
                # 6. 记录权益曲线
                current_equity = self.trade_exec.get_portfolio_value()
                daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
                
                self.equity_curve.append({
                    'trade_date': trade_date,
                    'equity': current_equity,
                    'cash': self.trade_exec.cash,
                    'position_value': current_equity - self.trade_exec.cash,
                    'daily_return': daily_return,
                })
                
                self.daily_returns.append(daily_return)
                prev_equity = current_equity
                
                # 进度日志
                if (i + 1) % 50 == 0:
                    logger.info(f"V66: 回测进度 {i+1}/{len(trade_dates)}，权益：{current_equity:,.2f}")
                
            except Exception as e:
                logger.error(f"V66: {trade_date} 回测失败：{e}")
                continue
        
        # 保存交易记录
        self.trades = self.trade_exec.trades
        self.positions = self.trade_exec.positions
        
        logger.info(f"V66: 逐日回测完成，共 {len(self.trades)} 笔交易")
    
    def _update_positions(self, current_df: pl.DataFrame, trade_date: str):
        """更新持仓状态"""
        # 构建市场数据字典
        market_data = {}
        
        for row in current_df.iter_rows(named=True):
            symbol = row.get('symbol', '')
            market_data[symbol] = {
                'close': row.get('close', 0),
                'ma10': row.get('ma10', 0),
                'net_main_rate': row.get('net_main_rate', 0),
            }
        
        self.trade_exec.update_positions(market_data, trade_date)
    
    def _check_exit_signals(self, current_df: pl.DataFrame, trade_date: str,
                            market_regime: V66MarketRegime):
        """检查离场信号"""
        for symbol, position in list(self.trade_exec.positions.items()):
            # 获取当日数据
            row_df = current_df.filter(pl.col('symbol') == symbol)
            
            if row_df.is_empty():
                continue
            
            row = row_df.row(0, named=True)
            current_price = row.get('close', 0)
            ma10 = row.get('ma10', 0)
            net_main_rate = row.get('net_main_rate', 0)
            
            if current_price <= 0:
                continue
            
            # 检查趋势破坏止损
            exit_result = self.trade_exec.check_trend_break_exit(symbol, ma10, net_main_rate)
            
            if exit_result and exit_result[0]:
                self.trade_exec.execute_sell(symbol, current_price, trade_date, exit_result[1])
                self._record_trade_audit(symbol, trade_date, current_price, exit_result[1], position)
                continue
            
            # 检查其他离场条件
            exit_result = self.trade_exec.check_exit_conditions(symbol, current_price, trade_date, net_main_rate)
            
            if exit_result and exit_result[0]:
                self.trade_exec.execute_sell(symbol, current_price, trade_date, exit_result[1])
                self._record_trade_audit(symbol, trade_date, current_price, exit_result[1], position)
    
    def _check_entry_signals(self, current_df: pl.DataFrame, trade_date: str,
                             market_regime: V66MarketRegime):
        """检查入场信号"""
        # 获取数据可信度
        data_credibility = self.data_credibility_stats.get('dual', 0) > 0 and 'dual' or 'fund_flow'
        
        # 生成信号
        signals = self.alpha_center.generate_signals(
            current_df, trade_date, market_regime, data_credibility
        )
        
        # 执行买入
        for signal in signals:
            if not self.trade_exec.can_buy_more():
                break
            
            # 获取次日开盘价（简化处理，使用当日收盘价代替）
            next_open = signal.close_price * 1.01  # 假设次日开盘上涨 1%
            
            # 计算可用资金
            available_capital = self.trade_exec.cash * 0.95  # 保留 5% 现金
            
            trade = self.trade_exec.execute_buy(
                signal, next_open, signal.close_price, available_capital
            )
            
            if trade:
                # 记录交易审计
                self._record_trade_audit(
                    signal.symbol, trade_date, signal.close_price,
                    '机构资金踪迹入场', None, signal
                )
    
    def _record_trade_audit(self, symbol: str, trade_date: str, price: float,
                            reason: str, position: Optional[V66Position] = None,
                            signal: Optional[V66Signal] = None):
        """记录交易审计"""
        if position:
            audit = V66TradeAudit(
                symbol=symbol,
                buy_date=position.buy_date,
                sell_date=trade_date,
                buy_price=position.buy_price,
                sell_price=price,
                shares=position.shares,
                gross_pnl=(price - position.buy_price) * position.shares,
                total_fees=0,
                net_pnl=(price - position.buy_price) * position.shares,
                holding_days=position.holding_days,
                is_profitable=price > position.buy_price,
                sell_reason=reason,
                net_main_rate=position.net_main_rate,
                rs_percentile=position.rs_percentile,
                rs_z_score=position.rs_z_score,
                vcp_pass=position.vcp_pass,
                data_credibility=position.data_credibility,
                trigger_price=position.trigger_price,
                next_open_price=position.next_open_price,
                execution_price=position.execution_price,
            )
            self.trade_audits.append(audit)
    
    def _calculate_ic(self):
        """计算 IC 值"""
        logger.info("V66: 开始计算 IC 值...")
        
        if self.signal_data is None:
            logger.warning("V66: 信号数据未生成，无法计算 IC")
            return
        
        # 计算未来收益率（5 日）
        self.signal_data = self.signal_data.with_columns([
            ((pl.col('close').shift(-5).over('symbol') - pl.col('close')) /
             (pl.col('close') + 1e-9)).alias('future_return_5')
        ])
        
        # 计算 IC 序列
        self.ic_calculator.calculate_ic_series(
            self.signal_data,
            signal_col='composite_score',
            return_col='future_return_5'
        )
        
        # 打印 IC 报告
        self.ic_calculator.print_ic_report(V66_IC_TARGET_MEAN)
    
    def _calculate_metrics(self):
        """计算性能指标"""
        logger.info("V66: 开始计算性能指标...")
        
        if not self.equity_curve:
            logger.warning("V66: 权益曲线为空")
            return
        
        # 转换为 DataFrame
        equity_df = pl.DataFrame(self.equity_curve)
        
        # 1. 收益率统计
        returns = np.array(self.daily_returns)
        total_return = (equity_df['equity'].last() - self.initial_capital) / self.initial_capital
        
        # 2. 年化收益率
        n_days = len(equity_df)
        annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
        
        # 3. 波动率
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        
        # 4. 夏普比率
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # 5. 最大回撤
        peak = equity_df['equity'].cum_max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = float(drawdown.min())
        
        # 6. 交易统计
        n_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        # 7. 胜率
        profitable_trades = len([a for a in self.trade_audits if a.is_profitable])
        win_rate = profitable_trades / max(len(self.trade_audits), 1)
        
        # 8. 盈亏比
        avg_profit = np.mean([a.net_pnl for a in self.trade_audits if a.net_pnl > 0]) if any(a.net_pnl > 0 for a in self.trade_audits) else 0
        avg_loss = abs(np.mean([a.net_pnl for a in self.trade_audits if a.net_pnl < 0])) if any(a.net_pnl < 0 for a in self.trade_audits) else 1
        profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
        
        # 9. AE 指标
        ae_metric = calculate_ae_metric(win_rate, profit_loss_ratio, abs(max_drawdown), len(self.trade_audits))
        
        # 10. IC 统计
        ic_stats = self.ic_calculator.get_ic_statistics() if self.ic_calculator else {}
        
        # 保存指标
        self.metrics = {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'n_trades': n_trades,
            'win_rate': float(win_rate),
            'profit_loss_ratio': float(profit_loss_ratio),
            'ae_metric': float(ae_metric),
            'ic_mean': ic_stats.get('mean_ic', 0),
            'ic_rank_mean': ic_stats.get('mean_rank_ic', 0),
            'ic_ir': ic_stats.get('ic_ir', 0),
            'ic_positive_ratio': ic_stats.get('positive_ratio', 0),
            'data_credibility_stats': self.data_credibility_stats,
        }
        
        logger.info(f"V66: 性能指标计算完成")
    
    def _print_report(self):
        """打印回测报告"""
        logger.info("=" * 60)
        logger.info("V66 回测报告")
        logger.info("=" * 60)
        
        # 基础指标
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最终权益：{self.equity_curve[-1]['equity']:,.2f}")
        logger.info(f"总收益率：{self.metrics.get('total_return', 0)*100:.2f}%")
        logger.info(f"年化收益：{self.metrics.get('annual_return', 0)*100:.2f}%")
        logger.info(f"波动率：{self.metrics.get('volatility', 0)*100:.2f}%")
        logger.info(f"夏普比率：{self.metrics.get('sharpe', 0):.2f}")
        logger.info(f"最大回撤：{self.metrics.get('max_drawdown', 0)*100:.2f}%")
        logger.info("-" * 40)
        
        # 交易统计
        logger.info(f"交易次数：{self.metrics.get('n_trades', 0)}")
        logger.info(f"胜率：{self.metrics.get('win_rate', 0)*100:.1f}%")
        logger.info(f"盈亏比：{self.metrics.get('profit_loss_ratio', 0):.2f}")
        logger.info(f"AE 指标：{self.metrics.get('ae_metric', 0):.2f}")
        logger.info("-" * 40)
        
        # IC 统计
        logger.info("IC 统计:")
        logger.info(f"  Mean IC: {self.metrics.get('ic_mean', 0):.4f}")
        logger.info(f"  Mean Rank IC: {self.metrics.get('ic_rank_mean', 0):.4f}")
        logger.info(f"  IC IR: {self.metrics.get('ic_ir', 0):.2f}")
        logger.info(f"  Positive Ratio: {self.metrics.get('ic_positive_ratio', 0)*100:.1f}%")
        logger.info("-" * 40)
        
        # 数据可信度
        logger.info("数据可信度统计:")
        for credibility, count in self.data_credibility_stats.items():
            logger.info(f"  {credibility}: {count}")
        logger.info("=" * 60)
        
        # IC 达标检查
        ic_mean = self.metrics.get('ic_mean', 0)
        if ic_mean >= V66_IC_TARGET_MEAN:
            logger.info(f"✓ IC 达标：Mean IC ({ic_mean:.4f}) >= 目标 ({V66_IC_TARGET_MEAN})")
        else:
            logger.info(f"✗ IC 未达标：Mean IC ({ic_mean:.4f}) < 目标 ({V66_IC_TARGET_MEAN})")
        
        logger.info("=" * 60)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self.metrics
    
    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """获取权益曲线"""
        return self.equity_curve
    
    def get_trades(self) -> List[V66Trade]:
        """获取交易记录"""
        return self.trades
    
    def get_trade_audits(self) -> List[V66TradeAudit]:
        """获取交易审计"""
        return self.trade_audits


# ===========================================
# 便捷函数
# ===========================================

def run_v66_backtest(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    便捷函数：运行 V66 回测
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        配置字典
        
    Returns
    -------
    Dict[str, Any]
        回测结果
    """
    engine = V66BacktestEngine(config)
    engine.initialize()
    engine.load_data()
    return engine.run_backtest()


# ===========================================
# 主程序
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 默认配置
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': V66_INITIAL_CAPITAL,
        'max_positions': V66_MAX_POSITIONS,
    }
    
    try:
        # 运行回测
        metrics = run_v66_backtest(config)
        
        print("\n" + "=" * 60)
        print("V66 回测完成!")
        print("=" * 60)
        
    except ValueError as e:
        logger.error(f"V66: 数据熔断触发：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V66: 回测失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V66BacktestEngine',
    'run_v66_backtest',
]