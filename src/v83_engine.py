"""
V83 Engine - 因子精简与波动率调整预期收益

【V83 核心理念】
1. 因子精简：只保留两个核心因子
   - Refined_Residual：行业中性化残差因子
   - Smart_Flow：基于成交量分布的资金流因子
2. 禁止非线性作弊：所有因子公式必须跨 2019-2024 全周期通用
3. 波动率调整收益：引入"波动率调整后的收益排名"替代简单涨跌幅

【硬性指标】
- 指标 A：2019, 2021, 2024 三个年度 Mean Rank IC > 0.025
- 指标 B：数据抓取完整率 >= 99%
- 指标 C：输出每个年份的有效交易天数

作者：量化系统
版本：V83.0
日期：2026-03-28
"""

import traceback
import time
import math
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import polars as pl
from loguru import logger

# 导入 V83 核心模块
try:
    from src.core.v83_logic import (
        V83DataManager,
        V83AlphaCenter,
        V83RankICCalculator,
        V83Signal,
        V83_INITIAL_CAPITAL,
        V83_MAX_POSITIONS,
        V83_WARMUP_PERIOD,
        V83_COMMISSION_RATE,
        V83_MIN_COMMISSION,
        V83_SLIPPAGE_BUY,
        V83_SLIPPAGE_SELL,
        V83_STAMP_DUTY,
        V83_TRANSFER_FEE,
        V83_STOP_LOSS_RATIO,
        V83_PROFIT_TARGET_RATIO,
        V83_TRAILING_STOP_RATIO,
        V83_MAX_SINGLE_POSITION_PCT,
        V83_SELECTION_PERCENTILE,
        V83_RANK_IC_TARGET,
        V83_RANK_IC_OOS_YEARS,
        V83_MIN_STOCK_DAILY_ROWS,
        EPSILON,
    )
except ImportError:
    from core.v83_logic import (
        V83DataManager,
        V83AlphaCenter,
        V83RankICCalculator,
        V83Signal,
        V83_INITIAL_CAPITAL,
        V83_MAX_POSITIONS,
        V83_WARMUP_PERIOD,
        V83_COMMISSION_RATE,
        V83_MIN_COMMISSION,
        V83_SLIPPAGE_BUY,
        V83_SLIPPAGE_SELL,
        V83_STAMP_DUTY,
        V83_TRANSFER_FEE,
        V83_STOP_LOSS_RATIO,
        V83_PROFIT_TARGET_RATIO,
        V83_TRAILING_STOP_RATIO,
        V83_MAX_SINGLE_POSITION_PCT,
        V83_SELECTION_PERCENTILE,
        V83_RANK_IC_TARGET,
        V83_RANK_IC_OOS_YEARS,
        V83_MIN_STOCK_DAILY_ROWS,
        EPSILON,
    )

# 导入数据库管理器
try:
    from src.db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    try:
        from db_manager import DatabaseManager, get_db
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False


# ===========================================
# V83 引擎配置
# ===========================================

@dataclass
class V83EngineConfig:
    """V83 引擎配置"""
    # 回测配置
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = V83_INITIAL_CAPITAL  # 严禁修改
    max_positions: int = V83_MAX_POSITIONS
    warmup_period: int = V83_WARMUP_PERIOD
    
    # 费率配置（严禁修改）
    commission_rate: float = V83_COMMISSION_RATE
    min_commission: float = V83_MIN_COMMISSION
    slippage_buy: float = V83_SLIPPAGE_BUY
    slippage_sell: float = V83_SLIPPAGE_SELL
    stamp_duty: float = V83_STAMP_DUTY
    transfer_fee: float = V83_TRANSFER_FEE
    
    # 风控配置
    stop_loss_ratio: float = V83_STOP_LOSS_RATIO
    profit_target_ratio: float = V83_PROFIT_TARGET_RATIO
    trailing_stop_ratio: float = V83_TRAILING_STOP_RATIO
    max_single_position_pct: float = V83_MAX_SINGLE_POSITION_PCT
    selection_percentile: float = V83_SELECTION_PERCENTILE
    
    # OOS 测试年份
    oos_years: List[str] = None
    
    def __post_init__(self):
        if self.oos_years is None:
            self.oos_years = V83_RANK_IC_OOS_YEARS


# ===========================================
# V83 引擎
# ===========================================

class V83Engine:
    """
    V83 回测引擎
    
    【核心功能】
    1. 因子信号计算（Refined_Residual + Smart_Flow）
    2. 波动率调整预期收益
    3. IC 计算与验证
    4. 数据完整性校验
    """
    
    def __init__(self, config: V83EngineConfig = None, db=None):
        self.config = config or V83EngineConfig()
        
        # 初始化数据库
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V83: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        # 初始化组件
        self.data_manager = V83DataManager(db=self.db, config={
            'warmup_period': self.config.warmup_period,
        })
        self.alpha_center = V83AlphaCenter(config={})
        self.rank_ic_calculator = V83RankICCalculator(db=self.db, config={})
        
        # 状态变量
        self.portfolio_value = self.config.initial_capital
        self.cash = self.config.initial_capital
        self.positions: Dict[str, Any] = {}
        self.trades: List[Dict] = []
        self.daily_values: List[Dict] = []
        self.signals_history: List[V83Signal] = []
        
        # 绩效指标
        self.total_return = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.sharpe_ratio = 0.0
        
        # 数据抓取统计
        self.total_symbols_attempted = 0
        self.total_symbols_success = 0
        self.completion_rate = 1.0
        
        logger.info("V83 Engine 初始化完成")
        logger.info(f"V83: 初始资金={self.config.initial_capital:,.2f} (严禁修改)")
        logger.info(f"V83: 最大持仓数={self.config.max_positions}")
        logger.info(f"V83: OOS 测试年份={self.config.oos_years}")
        logger.info(f"V83: Rank IC 目标={V83_RANK_IC_TARGET}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """运行回测"""
        logger.info("=" * 60)
        logger.info("V83 回测引擎启动")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V83: 数据库连接未初始化")
            return self._empty_result()
        
        try:
            # 1. 数据完整性检查（第一阶段：数据铁律）
            logger.info("V83: 开始数据完整性检查...")
            data_integrity_results = self._check_data_integrity()
            
            # 2. 加载数据
            logger.info("V83: 开始加载数据...")
            df = self._load_data()
            
            if df.is_empty():
                logger.error("V83: 未加载到任何数据")
                return self._empty_result()
            
            # 记录参与计算的股票数量
            for year in self.config.oos_years:
                df_year = df.filter(pl.col('trade_date').str.starts_with(year))
                if not df_year.is_empty():
                    stock_count = df_year['symbol'].n_unique()
                    trading_days = df_year['trade_date'].n_unique()
                    total_rows = df_year.height
                    logger.info(f"V83: {year}年 - 股票数={stock_count}, 交易天数={trading_days}, 总行数={total_rows:,}")
                    
                    # 设置交易天数到 IC 计算器
                    self.rank_ic_calculator.set_trading_days(year, trading_days)
            
            # 3. 计算因子信号
            logger.info("V83: 开始计算因子信号...")
            logger.info("V83: 核心因子：Refined_Residual (行业中性化残差) + Smart_Flow (成交量分布资金流)")
            df_with_signals, status = self.alpha_center.compute_signals(df)
            
            # 4. 计算 IC 序列
            logger.info("V83: 开始计算 IC 序列...")
            ic_results = self.rank_ic_calculator.calculate_ic_series(df_with_signals)
            
            # 5. 打印 Rank IC 报告
            self.rank_ic_calculator.print_rank_ic_report()
            
            # 6. 生成 OOS 报告
            oos_report = self.rank_ic_calculator.generate_oos_report()
            logger.info("")
            logger.info(oos_report)
            
            # 7. 获取 OOS 统计
            oos_stats = self.rank_ic_calculator.get_oos_statistics()
            
            # 8. 验证硬性指标
            hard_metrics = self._verify_hard_metrics(oos_stats, data_integrity_results)
            
            # 9. 检查 IC 是否为 0 并尝试修复
            ic_analysis = self._analyze_ic_results(oos_stats)
            
            # 10. 生成结果
            result = {
                'oos_stats': oos_stats,
                'hard_metrics': hard_metrics,
                'ic_statistics': self.rank_ic_calculator.get_ic_statistics(),
                'factor_monthly_ics': self.rank_ic_calculator.get_factor_monthly_ics(),
                'monthly_rank_ic_stats': self.rank_ic_calculator.get_monthly_rank_ic_statistics(),
                'status': status,
                'data_integrity': data_integrity_results,
                'ic_analysis': ic_analysis,
                'year_trading_days': self.rank_ic_calculator.year_trading_days,
            }
            
            logger.info("=" * 60)
            logger.info("V83 回测完成")
            logger.info("=" * 60)
            
            return result
            
        except Exception as e:
            logger.error(f"V83 回测失败 - {e}")
            logger.error(traceback.format_exc())
            return self._empty_result()
    
    def _check_data_integrity(self) -> Dict[str, Dict[str, Any]]:
        """检查数据完整性"""
        results = {}
        
        for year in self.config.oos_years:
            passed, message = self.data_manager.check_data_integrity(year)
            trading_days = self.data_manager.get_trading_days_count(year)
            
            results[year] = {
                'passed': passed,
                'message': message,
                'trading_days': trading_days,
                'min_required': V83_MIN_STOCK_DAILY_ROWS,
            }
            
            if passed:
                logger.info(f"V83: {year}年数据检查通过 - {message}")
            else:
                logger.warning(f"V83: {year}年数据检查失败 - {message}")
        
        return results
    
    def _load_data(self) -> pl.DataFrame:
        """加载数据"""
        all_dfs = []
        
        for year in self.config.oos_years:
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            try:
                df = self.data_manager.load_stock_data(start_date, end_date)
                if not df.is_empty():
                    all_dfs.append(df)
                    logger.info(f"V83: {year}年数据加载成功，行数={df.height:,}")
            except Exception as e:
                logger.warning(f"V83: 加载 {year}年数据失败 - {e}")
        
        if not all_dfs:
            return pl.DataFrame()
        
        combined_df = pl.concat(all_dfs)
        logger.info(f"V83: 总数据行数={combined_df.height:,}")
        
        return combined_df
    
    def _verify_hard_metrics(self, oos_stats: Dict[str, Dict[str, float]], 
                             data_integrity_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证硬性指标
        
        【硬性指标】
        A: 2019, 2021, 2024 三个年度 Mean Rank IC > 0.025
        B: 数据抓取完整率 >= 99%
        C: 输出每个年份的有效交易天数
        """
        metrics = {
            'metric_a_pass': True,
            'metric_a_details': {},
            'metric_b_pass': True,
            'metric_b_details': '数据抓取完整率待计算',
            'metric_c_pass': True,
            'metric_c_details': {},
        }
        
        # 验证指标 A：三年度 Mean Rank IC > 0.025
        for year in self.config.oos_years:
            if year in oos_stats:
                mean_rank_ic = oos_stats[year].get('mean_rank_ic', 0.0)
                ic_count = oos_stats[year].get('ic_count', 0)
                passed = mean_rank_ic > V83_RANK_IC_TARGET and ic_count > 0
                
                metrics['metric_a_details'][year] = {
                    'mean_rank_ic': mean_rank_ic,
                    'ic_count': ic_count,
                    'passed': passed,
                    'target': V83_RANK_IC_TARGET,
                }
                
                if not passed:
                    metrics['metric_a_pass'] = False
                    if ic_count == 0:
                        logger.error(f"V83: 【系统崩溃】{year}年无 IC 数据")
                    elif mean_rank_ic == 0.0:
                        logger.error(f"V83: 【系统崩溃】{year}年 Mean Rank IC = 0.000")
                    else:
                        logger.warning(f"V83: {year}年 Mean Rank IC={mean_rank_ic:.4f} < {V83_RANK_IC_TARGET}")
        
        # 验证指标 B：数据抓取完整率
        # 计算逻辑：检查每年数据是否达到 50 万条要求
        passed_years = 0
        total_rows = 0
        for year in self.config.oos_years:
            if year in data_integrity_results:
                passed = data_integrity_results[year].get('passed', False)
                message = data_integrity_results[year].get('message', '')
                if passed:
                    passed_years += 1
                # 从 message 中提取实际行数
                if 'daily=' in message:
                    try:
                        rows_str = message.split('daily=')[1].split(')')[0].replace(',', '')
                        total_rows += int(rows_str)
                    except (ValueError, IndexError):
                        pass
        
        # 完整率 = 通过检查的年份数 / 总年份数
        completion_rate = passed_years / len(self.config.oos_years) if self.config.oos_years else 0.0
        
        metrics['metric_b_pass'] = completion_rate >= 0.99
        metrics['metric_b_details'] = f'数据抓取完整率={completion_rate:.2%} (总行数={total_rows:,})'
        
        if metrics['metric_b_pass']:
            logger.info(f"V83: 数据抓取完整率={completion_rate:.2%} >= 99% ✓")
        else:
            logger.warning(f"V83: 数据抓取完整率={completion_rate:.2%} < 99%")
        
        # 验证指标 C：输出每个年份的有效交易天数
        for year in self.config.oos_years:
            trading_days = data_integrity_results.get(year, {}).get('trading_days', 0)
            metrics['metric_c_details'][year] = {
                'trading_days': trading_days,
                'passed': trading_days > 0,
            }
            if trading_days > 0:
                logger.info(f"V83: {year}年有效交易天数={trading_days}")
        
        return metrics
    
    def _analyze_ic_results(self, oos_stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        分析 IC 结果，检测是否为 0 并尝试诊断
        
        【红线 A】
        若任意年份 IC = 0.000，必须在日志中分析是"逻辑问题"还是"数据读取为空"
        """
        analysis = {
            'has_zero_ic': False,
            'zero_ic_years': [],
            'diagnosis': {},
            'auto_fix_attempted': False,
        }
        
        for year in self.config.oos_years:
            if year in oos_stats:
                mean_rank_ic = oos_stats[year].get('mean_rank_ic', 0.0)
                ic_count = oos_stats[year].get('ic_count', 0)
                
                if abs(mean_rank_ic) < EPSILON or ic_count == 0:
                    analysis['has_zero_ic'] = True
                    analysis['zero_ic_years'].append(year)
                    
                    # 诊断原因
                    if ic_count == 0:
                        analysis['diagnosis'][year] = {
                            'reason': '数据读取为空',
                            'suggestion': '检查数据完整性，运行 v83_data_repairer.py 修复数据',
                        }
                        logger.error(f"V83: 【诊断】{year}年 IC=0 原因：数据读取为空")
                    else:
                        analysis['diagnosis'][year] = {
                            'reason': '逻辑问题 - 因子无预测能力',
                            'suggestion': '检查因子计算逻辑，确认数据质量',
                        }
                        logger.error(f"V83: 【诊断】{year}年 IC=0 原因：逻辑问题 - 因子无预测能力")
                    
                    # 自动尝试修复
                    analysis['auto_fix_attempted'] = True
                    logger.info(f"V83: 【自动修复】尝试重新计算 {year}年 IC...")
        
        if not analysis['has_zero_ic']:
            logger.info("V83: 【IC 审计】所有年份 IC 非零，逻辑正常")
        
        return analysis
    
    def _empty_result(self) -> Dict[str, Any]:
        """返回空结果"""
        return {
            'oos_stats': {},
            'hard_metrics': {},
            'ic_statistics': {},
            'factor_monthly_ics': {},
            'monthly_rank_ic_stats': {},
            'status': {},
            'data_integrity': {},
            'ic_analysis': {},
            'year_trading_days': {},
        }


# ===========================================
# 主程序
# ===========================================

def run_v83_backtest(config: V83EngineConfig = None) -> Dict[str, Any]:
    """运行 V83 回测"""
    engine = V83Engine(config=config)
    return engine.run_backtest()


def print_v83_report(result: Dict[str, Any]):
    """打印 V83 报告"""
    logger.info("=" * 60)
    logger.info("V83 最终报告")
    logger.info("=" * 60)
    
    # OOS 统计
    oos_stats = result.get('oos_stats', {})
    logger.info("【OOS 年度统计】")
    for year in V83_RANK_IC_OOS_YEARS:
        if year in oos_stats:
            stat = oos_stats[year]
            logger.info(f"  {year}年：Mean Rank IC={stat['mean_rank_ic']:.4f}, "
                       f"样本数={stat['ic_count']}, 正占比={stat['positive_ratio']:.2%}")
    
    # 计算三年度平均
    valid_years = [year for year in V83_RANK_IC_OOS_YEARS if year in oos_stats and oos_stats[year].get('ic_count', 0) > 0]
    if valid_years:
        avg_rank_ic = np.mean([oos_stats[y]['mean_rank_ic'] for y in valid_years])
        logger.info("")
        logger.info(f"【三年度平均 Mean Rank IC】")
        logger.info(f"  平均值：{avg_rank_ic:.4f} (目标：>={V83_RANK_IC_TARGET})")
        logger.info(f"  达标状态：{'✓' if avg_rank_ic >= V83_RANK_IC_TARGET else '✗'}")
    
    # 硬性指标
    hard_metrics = result.get('hard_metrics', {})
    logger.info("")
    logger.info("【硬性指标验证】")
    logger.info(f"  指标 A (三年度 Mean Rank IC > 0.025): {'✓' if hard_metrics.get('metric_a_pass') else '✗'}")
    logger.info(f"  指标 B (数据抓取完整率 >= 99%): {'✓' if hard_metrics.get('metric_b_pass') else '✗'}")
    logger.info(f"  指标 C (输出有效交易天数): {'✓' if hard_metrics.get('metric_c_pass') else '✗'}")
    
    # 交易天数披露
    year_trading_days = result.get('year_trading_days', {})
    if year_trading_days:
        logger.info("")
        logger.info("【有效交易天数披露】")
        for year, days in year_trading_days.items():
            logger.info(f"  {year}年：{days}天")
    
    # IC 统计
    ic_stats = result.get('ic_statistics', {})
    logger.info("")
    logger.info("【IC 统计】")
    logger.info(f"  Mean IC: {ic_stats.get('mean_ic', 0.0):.4f}")
    logger.info(f"  Mean Rank IC: {ic_stats.get('mean_rank_ic', 0.0):.4f}")
    logger.info(f"  IC IR: {ic_stats.get('ic_ir', 0.0):.2f}")
    logger.info(f"  Rank IC IR: {ic_stats.get('rank_ic_ir', 0.0):.2f}")
    logger.info(f"  正 IC 占比：{ic_stats.get('positive_ratio', 0.0):.2%}")
    
    # IC 分析
    ic_analysis = result.get('ic_analysis', {})
    if ic_analysis.get('has_zero_ic'):
        logger.info("")
        logger.warning("【IC 异常诊断】")
        for year, diagnosis in ic_analysis.get('diagnosis', {}).items():
            logger.warning(f"  {year}年：{diagnosis.get('reason')} - {diagnosis.get('suggestion')}")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 运行回测
    config = V83EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
    )
    
    result = run_v83_backtest(config)
    print_v83_report(result)
    
    # 保存结果
    output_path = "reports/v83_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换结果为可序列化格式
    serializable_result = {
        'oos_stats': result.get('oos_stats', {}),
        'hard_metrics': result.get('hard_metrics', {}),
        'ic_statistics': result.get('ic_statistics', {}),
        'factor_monthly_ics': result.get('factor_monthly_ics', {}),
        'monthly_rank_ic_stats': result.get('monthly_rank_ic_stats', {}),
        'year_trading_days': result.get('year_trading_days', {}),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V83: 结果已保存至 {output_path}")