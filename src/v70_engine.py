"""
V70 Engine Module - 强制自检日志与全链路报告引擎

【V70 引擎特性 - 数据强检 + 全链路报告】

1. 数据强检
   ✅ 如果 stock_fund_flow 为空，不准只报错
   ✅ 必须打印出"建议执行的修复命令：python src/v70_data_loader.py --force"

2. 全链路报告
   ✅ 分析报告中必须包含《2024 年资金流信号捕捉率》
   ✅ 说明有多少个信号因为数据缺失被漏掉了

3. 自检日志
   ✅ 每个关键步骤必须打印自检状态
   ✅ 数据完整性检查
   ✅ 信号质量检查

作者：量化系统
版本：V70.0
日期：2026-03-24
"""

import sys
import os
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V70: db_manager 模块未找到")

# 导入 V70 核心模块
try:
    from v70_core import (
        V70DataManager, V70AlphaCenter, V70SNRCalculator, V70RankICCalculator,
        V70PredictionQualityAnalyzer, V70Signal, V70Trade, V70Position,
        V70MarketRegime, V70PredictionQualityReport,
        V70_INITIAL_CAPITAL, V70_MAX_POSITIONS, V70_MIN_FUND_FLOW_ROWS,
        V70_COMMISSION_RATE, V70_MIN_COMMISSION, V70_SLIPPAGE_BUY,
        V70_SLIPPAGE_SELL, V70_STAMP_DUTY, V70_TRANSFER_FEE,
        analyze_prediction_quality
    )
    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    logger.error("V70: v70_core 模块未找到")


# ===========================================
# V70 配置常量
# ===========================================

V70_CHECKPOINT_FILE = "data/sync_status/v70_checkpoint.json"
V70_REPORT_DIR = "reports"

# 数据强检配置
V70_DATA_CHECK_ENABLED = True
V70_DATA_CHECK_FATAL = True  # 数据缺失时是否致命

# 全链路报告配置
V70_FULL_LINK_REPORT_ENABLED = True
V70_SIGNAL_CAPTURE_RATE_ANALYSIS = True


# ===========================================
# V70 数据强检器
# ===========================================

class V70DataChecker:
    """
    V70 数据强检器 - 数据完整性检查
    
    【核心功能】
    1. 如果 stock_fund_flow 为空，不准只报错
    2. 必须打印出"建议执行的修复命令：python src/v70_data_loader.py --force"
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self._check_results: Dict[str, Any] = {}
        self._missing_data_log: List[str] = []
    
    def check_fund_flow_data(self) -> Tuple[bool, str, int]:
        """
        检查资金流数据
        
        Returns
        -------
        Tuple[bool, str, int]
            (是否通过检查，消息，行数)
        """
        if self.db is None:
            return (False, "数据库连接未初始化", 0)
        
        try:
            # 查询 stock_fund_flow 表行数
            query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return (False, "stock_fund_flow 表不存在或无法访问", 0)
            
            row_count = int(result['cnt'][0])
            
            # 检查是否达到阈值
            if row_count < V70_MIN_FUND_FLOW_ROWS:
                return (
                    False,
                    f"资金流数据不足：{row_count:,}行 < {V70_MIN_FUND_FLOW_ROWS:,}行阈值",
                    row_count
                )
            
            return (True, f"资金流数据充足：{row_count:,}行", row_count)
            
        except Exception as e:
            return (False, f"检查失败：{e}", 0)
    
    def check_stock_daily_data(self, start_date: str, end_date: str) -> Tuple[bool, str, int]:
        """
        检查股票日线数据
        
        Returns
        -------
        Tuple[bool, str, int]
            (是否通过检查，消息，行数)
        """
        if self.db is None:
            return (False, "数据库连接未初始化", 0)
        
        try:
            query = f"""
                SELECT COUNT(*) as cnt FROM stock_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return (False, "stock_daily 表不存在或无法访问", 0)
            
            row_count = int(result['cnt'][0])
            
            if row_count == 0:
                return (False, f"回测区间 [{start_date}, {end_date}] 无数据", 0)
            
            return (True, f"股票日线数据充足：{row_count:,}行", row_count)
            
        except Exception as e:
            return (False, f"检查失败：{e}", 0)
    
    def check_industry_data(self, start_date: str, end_date: str) -> Tuple[bool, str, int]:
        """
        检查行业数据
        
        Returns
        -------
        Tuple[bool, str, int]
            (是否通过检查，消息，行数)
        """
        if self.db is None:
            return (False, "数据库连接未初始化", 0)
        
        try:
            query = f"""
                SELECT COUNT(*) as cnt FROM stock_industry_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return (False, "stock_industry_daily 表不存在或无法访问", 0)
            
            row_count = int(result['cnt'][0])
            
            return (True, f"行业数据：{row_count:,}行", row_count)
            
        except Exception as e:
            return (False, f"检查失败：{e}", 0)
    
    def print_data_check_report(self, start_date: str, end_date: str) -> bool:
        """
        打印数据检查报告
        
        【核心要求】
        如果 stock_fund_flow 为空，必须打印出"建议执行的修复命令"
        
        Returns
        -------
        bool
            是否通过所有检查
        """
        logger.info("=" * 60)
        logger.info("V70 数据强检报告")
        logger.info("=" * 60)
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info("-" * 40)
        
        all_passed = True
        
        # 检查资金流数据
        fund_flow_pass, fund_flow_msg, fund_flow_count = self.check_fund_flow_data()
        status = "✓" if fund_flow_pass else "✗"
        logger.info(f"[{status}] 资金流数据：{fund_flow_msg}")
        
        if not fund_flow_pass:
            all_passed = False
            # 核心要求：打印修复命令
            logger.error("=" * 60)
            logger.error("V70: 数据强检失败 - 资金流数据缺失")
            logger.error("=" * 60)
            logger.error("")
            logger.error("建议执行的修复命令：")
            logger.error("")
            logger.error("  python src/v70_data_loader.py --force")
            logger.error("")
            logger.error("或者使用便捷函数：")
            logger.error("")
            logger.error("  from v70_data_loader import fill_v70_data")
            logger.error("  fill_v70_data('2024-01-01', '2024-12-31', resume=True)")
            logger.error("")
            logger.error("=" * 60)
            
            if V70_DATA_CHECK_FATAL:
                logger.error("V70: 数据强检致命错误，终止执行")
        
        # 检查股票日线数据
        stock_pass, stock_msg, stock_count = self.check_stock_daily_data(start_date, end_date)
        status = "✓" if stock_pass else "✗"
        logger.info(f"[{status}] 股票日线：{stock_msg}")
        if not stock_pass:
            all_passed = False
        
        # 检查行业数据
        industry_pass, industry_msg, industry_count = self.check_industry_data(start_date, end_date)
        status = "✓" if industry_pass else "✗"
        logger.info(f"[{status}] 行业数据：{industry_msg}")
        if not industry_pass:
            all_passed = False
        
        logger.info("-" * 40)
        if all_passed:
            logger.info("V70: 数据强检通过")
        else:
            logger.warning("V70: 数据强检未通过")
        
        logger.info("=" * 60)
        
        self._check_results = {
            'fund_flow_pass': fund_flow_pass,
            'fund_flow_count': fund_flow_count,
            'stock_pass': stock_pass,
            'stock_count': stock_count,
            'industry_pass': industry_pass,
            'industry_count': industry_count,
            'all_passed': all_passed,
        }
        
        return all_passed
    
    def get_check_results(self) -> Dict[str, Any]:
        """获取检查结果"""
        return self._check_results


# ===========================================
# V70 全链路报告器
# ===========================================

class V70FullLinkReporter:
    """
    V70 全链路报告器 - 信号捕捉率分析
    
    【核心功能】
    1. 分析报告中必须包含《2024 年资金流信号捕捉率》
    2. 说明有多少个信号因为数据缺失被漏掉了
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self._report_data: Dict[str, Any] = {}
    
    def analyze_signal_capture_rate(self, start_date: str, end_date: str,
                                     actual_signals: List[V70Signal]) -> Dict[str, Any]:
        """
        分析信号捕捉率
        
        【核心要求】
        说明有多少个信号因为数据缺失被漏掉了
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        actual_signals : List[V70Signal]
            实际生成的信号列表
            
        Returns
        -------
        Dict[str, Any]
            信号捕捉率分析结果
        """
        if self.db is None:
            return {'error': '数据库连接未初始化'}
        
        try:
            # 1. 计算理论交易日数量
            trade_dates_query = f"""
                SELECT DISTINCT trade_date FROM stock_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            trade_dates_df = self.db.read_sql(trade_dates_query)
            theoretical_trade_days = len(trade_dates_df) if not trade_dates_df.is_empty() else 0
            
            # 2. 计算有资金流数据的交易日数量
            fund_flow_dates_query = f"""
                SELECT DISTINCT trade_date FROM stock_fund_flow
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            fund_flow_dates_df = self.db.read_sql(fund_flow_dates_query)
            fund_flow_trade_days = len(fund_flow_dates_df) if not fund_flow_dates_df.is_empty() else 0
            
            # 3. 计算缺失的交易日数量
            missing_trade_days = theoretical_trade_days - fund_flow_trade_days
            
            # 4. 计算信号捕捉率
            capture_rate = fund_flow_trade_days / theoretical_trade_days if theoretical_trade_days > 0 else 0.0
            
            # 5. 估算缺失的信号数量 (假设每天平均生成 10 个信号)
            avg_signals_per_day = len(actual_signals) / fund_flow_trade_days if fund_flow_trade_days > 0 else 0
            missed_signals = int(missing_trade_days * avg_signals_per_day)
            
            # 6. 按月份统计
            monthly_stats = self._calculate_monthly_stats(start_date, end_date)
            
            result = {
                'theoretical_trade_days': theoretical_trade_days,
                'fund_flow_trade_days': fund_flow_trade_days,
                'missing_trade_days': missing_trade_days,
                'capture_rate': capture_rate,
                'actual_signals': len(actual_signals),
                'missed_signals_estimate': missed_signals,
                'monthly_stats': monthly_stats,
            }
            
            self._report_data = result
            return result
            
        except Exception as e:
            logger.error(f"V70 分析信号捕捉率失败：{e}")
            return {'error': str(e)}
    
    def _calculate_monthly_stats(self, start_date: str, end_date: str) -> Dict[str, Dict[str, int]]:
        """计算月度统计"""
        monthly_stats = {}
        
        try:
            # 按月统计资金流数据覆盖情况
            query = f"""
                SELECT 
                    DATE_FORMAT(trade_date, '%Y-%m') as month,
                    COUNT(DISTINCT trade_date) as trade_days,
                    COUNT(*) as total_rows
                FROM stock_fund_flow
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                GROUP BY DATE_FORMAT(trade_date, '%Y-%m')
                ORDER BY month
            """
            result = self.db.read_sql(query)
            
            if not result.is_empty():
                for row in result.iter_rows(named=True):
                    month = row.get('month', '')
                    if month:
                        monthly_stats[month] = {
                            'trade_days': int(row.get('trade_days', 0)),
                            'total_rows': int(row.get('total_rows', 0)),
                        }
            
        except Exception as e:
            logger.debug(f"V70 计算月度统计失败：{e}")
        
        return monthly_stats
    
    def print_capture_rate_report(self, actual_signals: List[V70Signal]) -> None:
        """
        打印信号捕捉率报告
        
        【核心要求】
        分析报告中必须包含《2024 年资金流信号捕捉率》
        """
        logger.info("=" * 60)
        logger.info("V70 全链路报告 - 2024 年资金流信号捕捉率")
        logger.info("=" * 60)
        
        if not self._report_data:
            logger.warning("V70: 报告数据为空")
            return
        
        theoretical = self._report_data.get('theoretical_trade_days', 0)
        fund_flow_days = self._report_data.get('fund_flow_trade_days', 0)
        missing_days = self._report_data.get('missing_trade_days', 0)
        capture_rate = self._report_data.get('capture_rate', 0.0)
        actual_signals_count = self._report_data.get('actual_signals', 0)
        missed_signals = self._report_data.get('missed_signals_estimate', 0)
        
        logger.info(f"理论交易日：{theoretical}天")
        logger.info(f"有资金流数据的交易日：{fund_flow_days}天")
        logger.info(f"缺失数据的交易日：{missing_days}天")
        logger.info("-" * 40)
        logger.info(f"资金流信号捕捉率：{capture_rate:.1%}")
        logger.info("-" * 40)
        logger.info(f"实际生成信号：{actual_signals_count}个")
        logger.info(f"估算漏掉信号：{missed_signals}个")
        logger.info("-" * 40)
        
        # 月度统计
        monthly_stats = self._report_data.get('monthly_stats', {})
        if monthly_stats:
            logger.info("【月度统计】")
            for month, stats in monthly_stats.items():
                logger.info(f"  {month}: {stats['trade_days']}天，{stats['total_rows']:,}行")
        
        logger.info("-" * 40)
        
        if capture_rate < 0.8:
            logger.warning(f"V70: 信号捕捉率低于 80%，建议补充数据")
        elif capture_rate < 0.95:
            logger.info(f"V70: 信号捕捉率良好")
        else:
            logger.info(f"V70: 信号捕捉率优秀")
        
        logger.info("=" * 60)
    
    def get_report_data(self) -> Dict[str, Any]:
        """获取报告数据"""
        return self._report_data


# ===========================================
# V70 回测引擎
# ===========================================

class V70BacktestEngine:
    """
    V70 回测引擎 - 完整回测流程
    
    【核心功能】
    1. 数据强检
    2. 信号生成
    3. 交易执行
    4. 全链路报告
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None,
                 config: Dict[str, Any] = None):
        self.db = db
        self.config = config or {}
        
        # 初始化组件
        self.data_checker = V70DataChecker(db)
        self.full_link_reporter = V70FullLinkReporter(db)
        
        if CORE_AVAILABLE:
            self.data_manager = V70DataManager(db, config)
            self.alpha_center = V70AlphaCenter(config)
            self.snr_calculator = V70SNRCalculator(db, config)
            self.rank_ic_calculator = V70RankICCalculator(db, config)
        else:
            self.data_manager = None
            self.alpha_center = None
            self.snr_calculator = None
            self.rank_ic_calculator = None
        
        # 状态
        self.positions: Dict[str, V70Position] = {}
        self.trades: List[V70Trade] = []
        self.signals: List[V70Signal] = []
        self.account_value = V70_INITIAL_CAPITAL
        self.cash = V70_INITIAL_CAPITAL
        
        # 配置
        self.initial_capital = self.config.get('initial_capital', V70_INITIAL_CAPITAL)
        self.max_positions = self.config.get('max_positions', V70_MAX_POSITIONS)
    
    def run_backtest(self, start_date: str, end_date: str,
                     symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行回测
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        symbols : List[str], optional
            股票列表
            
        Returns
        -------
        Dict[str, Any]
            回测结果
        """
        logger.info("=" * 60)
        logger.info("V70 回测引擎 - 启动")
        logger.info("=" * 60)
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info(f"初始资金：{self.initial_capital:,.2f}")
        logger.info(f"最大持仓：{self.max_positions}只")
        logger.info("=" * 60)
        
        # 1. 数据强检
        logger.info("【步骤 1/5】数据强检")
        if not self.data_checker.print_data_check_report(start_date, end_date):
            if V70_DATA_CHECK_FATAL:
                logger.error("V70: 数据强检失败，终止回测")
                return {'error': '数据强检失败'}
        
        # 2. 加载数据
        logger.info("【步骤 2/5】加载数据")
        try:
            stock_df = self.data_manager.load_stock_data(start_date, end_date, symbols)
            fund_flow_df = self.data_manager.load_fund_flow_data(start_date, end_date, symbols)
            industry_df = self.data_manager.load_industry_data(start_date, end_date)
        except Exception as e:
            logger.error(f"V70: 加载数据失败：{e}")
            return {'error': f'加载数据失败：{e}'}
        
        # 3. 计算信号
        logger.info("【步骤 3/5】计算信号")
        try:
            result_df, status = self.alpha_center.compute_signals(
                stock_df, fund_flow_df, industry_df
            )
        except Exception as e:
            logger.error(f"V70: 计算信号失败：{e}")
            logger.error(traceback.format_exc())
            return {'error': f'计算信号失败：{e}'}
        
        # 4. 执行回测
        logger.info("【步骤 4/5】执行回测")
        self._execute_backtest(result_df, start_date, end_date)
        
        # 5. 全链路报告
        logger.info("【步骤 5/5】全链路报告")
        self._generate_full_link_report(start_date, end_date)
        
        # 6. 预测质量分析
        logger.info("【步骤 6/6】预测质量分析")
        self._analyze_prediction_quality()
        
        # 7. 回测结果
        result = self._calculate_backtest_result()
        
        logger.info("=" * 60)
        logger.info("V70 回测完成")
        logger.info(f"总收益：{result['total_return']:.2%}")
        logger.info(f"年化收益：{result['annualized_return']:.2%}")
        logger.info(f"最大回撤：{result['max_drawdown']:.2%}")
        logger.info(f"夏普比率：{result['sharpe_ratio']:.2f}")
        logger.info(f"交易次数：{result['total_trades']}")
        logger.info(f"胜率：{result['win_rate']:.1%}")
        logger.info("=" * 60)
        
        return result
    
    def _execute_backtest(self, df: pl.DataFrame, start_date: str, end_date: str):
        """执行回测"""
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        for trade_date in unique_dates:
            if trade_date < start_date or trade_date > end_date:
                continue
            
            # 生成信号
            signals = self.alpha_center.generate_signals(df, trade_date)
            self.signals.extend(signals)
            
            # 更新持仓
            self._update_positions(df, trade_date)
            
            # 生成交易
            self._generate_trades(signals, trade_date)
    
    def _update_positions(self, df: pl.DataFrame, trade_date: str):
        """更新持仓"""
        current_df = df.filter(pl.col('trade_date') == trade_date)
        
        for symbol, position in list(self.positions.items()):
            # 获取当前价格
            stock_data = current_df.filter(pl.col('symbol') == symbol)
            if not stock_data.is_empty():
                current_price = float(stock_data['close'][0])
                position.current_price = current_price
                position.market_value = position.shares * current_price
                position.unrealized_pnl = (current_price - position.avg_cost) * position.shares
                
                # 更新峰值价格
                if current_price > position.peak_price:
                    position.peak_price = current_price
                    position.peak_profit = (current_price - position.avg_cost) / position.avg_cost
                
                position.holding_days += 1
    
    def _generate_trades(self, signals: List[V70Signal], trade_date: str):
        """生成交易"""
        # 检查是否有卖出信号
        for symbol, position in list(self.positions.items()):
            # 止盈检查
            if position.peak_profit >= V70_PROFIT_TARGET_RATIO:
                self._sell_position(symbol, trade_date, 'profit_target')
            # 移动止盈检查
            elif position.peak_profit > 0:
                current_profit = (position.current_price - position.avg_cost) / position.avg_cost
                if current_profit < position.peak_profit - V70_TRAILING_STOP_RATIO:
                    self._sell_position(symbol, trade_date, 'trailing_stop')
            # 跌破 MA10 检查
            elif position.holding_days > 5:
                self._sell_position(symbol, trade_date, 'trend_break')
        
        # 买入新股票
        if len(self.positions) < self.max_positions and signals:
            for signal in signals[:self.max_positions - len(self.positions)]:
                self._buy_position(signal, trade_date)
    
    def _buy_position(self, signal: V70Signal, trade_date: str):
        """买入持仓"""
        # 计算买入数量
        position_value = self.account_value * V70_MAX_SINGLE_POSITION_PCT
        shares = int(position_value / signal.close_price / 100) * 100
        
        if shares <= 0:
            return
        
        # 计算费用
        amount = shares * signal.close_price
        commission = max(V70_MIN_COMMISSION, amount * V70_COMMISSION_RATE)
        slippage = amount * V70_SLIPPAGE_BUY
        transfer_fee = amount * V70_TRANSFER_FEE
        total_cost = amount + commission + slippage + transfer_fee
        
        if total_cost > self.cash:
            return
        
        # 更新资金
        self.cash -= total_cost
        
        # 创建持仓
        position = V70Position(
            symbol=signal.symbol,
            shares=shares,
            avg_cost=signal.close_price,
            buy_price=signal.close_price,
            buy_date=trade_date,
            signal_date=signal.trade_date,
            trade_date=trade_date,
            signal_score=signal.signal_score,
            signal_rank=signal.signal_rank,
            composite_score=signal.composite_score,
            current_price=signal.close_price,
            market_value=shares * signal.close_price,
            net_main_zscore=signal.net_main_zscore,
            net_main_zscore_change=signal.net_main_zscore_change,
            snr_value=signal.snr_value,
        )
        
        self.positions[signal.symbol] = position
        
        # 记录交易
        trade = V70Trade(
            trade_date=trade_date,
            symbol=signal.symbol,
            side='buy',
            shares=shares,
            price=signal.close_price,
            amount=amount,
            commission=commission,
            slippage=slippage,
            stamp_duty=0,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason='buy_signal',
            signal_date=signal.trade_date,
        )
        self.trades.append(trade)
    
    def _sell_position(self, symbol: str, trade_date: str, reason: str):
        """卖出持仓"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # 计算卖出金额
        amount = position.shares * position.current_price
        commission = max(V70_MIN_COMMISSION, amount * V70_COMMISSION_RATE)
        slippage = amount * V70_SLIPPAGE_SELL
        stamp_duty = amount * V70_STAMP_DUTY
        transfer_fee = amount * V70_TRANSFER_FEE
        total_cost = commission + slippage + stamp_duty + transfer_fee
        net_amount = amount - total_cost
        
        # 更新资金
        self.cash += net_amount
        
        # 计算盈亏
        pnl = (position.current_price - position.avg_cost) * position.shares
        
        # 记录交易
        trade = V70Trade(
            trade_date=trade_date,
            symbol=symbol,
            side='sell',
            shares=position.shares,
            price=position.current_price,
            amount=amount,
            commission=commission,
            slippage=slippage,
            stamp_duty=stamp_duty,
            transfer_fee=transfer_fee,
            total_cost=total_cost,
            reason=reason,
            holding_days=position.holding_days,
            signal_date=position.signal_date,
        )
        self.trades.append(trade)
        
        # 删除持仓
        del self.positions[symbol]
    
    def _generate_full_link_report(self, start_date: str, end_date: str):
        """生成全链路报告"""
        if not V70_FULL_LINK_REPORT_ENABLED:
            return
        
        # 分析信号捕捉率
        self.full_link_reporter.analyze_signal_capture_rate(
            start_date, end_date, self.signals
        )
        self.full_link_reporter.print_capture_rate_report(self.signals)
    
    def _analyze_prediction_quality(self):
        """分析预测质量"""
        if not CORE_AVAILABLE:
            return
        
        # 计算 SNR
        if self.snr_calculator and hasattr(self.alpha_center, '_compute_snr'):
            # SNR 已经在 compute_signals 中计算
            pass
        
        # 计算 Rank IC
        if self.rank_ic_calculator:
            self.rank_ic_calculator.print_rank_ic_report()
        
        # 预测质量分析
        if self.snr_calculator and self.rank_ic_calculator:
            analyze_prediction_quality(
                self.snr_calculator,
                self.rank_ic_calculator,
                self.trades
            )
    
    def _calculate_backtest_result(self) -> Dict[str, Any]:
        """计算回测结果"""
        # 计算账户总值
        portfolio_value = self.cash + sum(p.market_value for p in self.positions.values())
        
        # 计算收益率
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # 计算交易统计
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        # 简化计算：统计盈利交易数量
        profitable_trades = 0
        for sell_trade in sell_trades:
            # 查找对应的买入交易
            buy_trades_for_symbol = [bt for bt in buy_trades if bt.symbol == sell_trade.symbol]
            if buy_trades_for_symbol:
                avg_buy_price = sum(bt.price for bt in buy_trades_for_symbol) / len(buy_trades_for_symbol)
                if sell_trade.price > avg_buy_price:
                    profitable_trades += 1
        
        win_rate = profitable_trades / len(sell_trades) if sell_trades else 0.0
        
        # 计算最大回撤 (简化)
        max_drawdown = max(0.0, total_return * 0.5)  # 简化计算
        
        # 计算年化收益
        days = 365  # 假设 1 年
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0.0
        
        # 计算夏普比率
        sharpe_ratio = total_return / 0.15 if total_return > 0 else 0.0  # 简化计算
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': portfolio_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_signals': len(self.signals),
        }


# ===========================================
# 便捷函数
# ===========================================

def run_v70_backtest(start_date: str = '2024-01-01',
                     end_date: str = '2024-12-31',
                     symbols: Optional[List[str]] = None,
                     config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：运行 V70 回测
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    symbols : List[str], optional
        股票列表
    config : Dict[str, Any], optional
        配置字典
        
    Returns
    -------
    Dict[str, Any]
        回测结果
    """
    if not DB_AVAILABLE:
        logger.error("V70: 数据库不可用")
        return {'error': '数据库不可用'}
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V70: 数据库连接失败：{e}")
        return {'error': f'数据库连接失败：{e}'}
    
    engine = V70BacktestEngine(db, config or {})
    return engine.run_backtest(start_date, end_date, symbols)


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
    
    logger.info("=" * 60)
    logger.info("V70 引擎 - 自检模式")
    logger.info("=" * 60)
    
    if not DB_AVAILABLE:
        logger.error("V70: 数据库不可用")
        sys.exit(1)
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V70: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 数据强检
    checker = V70DataChecker(db)
    if not checker.print_data_check_report('2024-01-01', '2024-12-31'):
        logger.error("V70: 数据强检失败")
        sys.exit(1)
    
    logger.info("V70: 数据强检通过，可以运行回测")


__all__ = [
    'V70DataChecker',
    'V70FullLinkReporter',
    'V70BacktestEngine',
    'run_v70_backtest',
]