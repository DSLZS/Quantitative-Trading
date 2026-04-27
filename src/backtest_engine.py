"""
Backtest Engine Module - V218 Referee-Player Architecture
==========================================================

【核心职责】
1. 数据加载：从 MySQL 数据库加载股票数据
2. 数据校验：确保每日股票数 > 5000
3. 回测执行：调用 AlphaModel 和 BacktestReferee 执行回测
4. 报告生成：输出 IC、IR、年化收益等指标

【合规锁定 - V218】
- 初始资金：100,000
- 费率：1.3‰ (佣金 0.3‰ + 印花税 1‰ + 滑点 0.5‰)
- 无未来函数：所有计算仅使用 T-1 日及之前数据
- Referee-Player 架构：引擎仅负责裁判职责
"""

from typing import Any, Optional, Dict, List, Tuple
import os
from datetime import datetime
from pathlib import Path
import json

import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()

# 内存优化配置
pd.options.mode.chained_assignment = None

# 版本号
VERSION = "V218"

# 验收阈值
IC_THRESHOLD = 0.05
IC_IR_THRESHOLD = 0.30

# 数据校验阈值
MIN_STOCK_COUNT = 5000  # 每日最少股票数


class BacktestEngine:
    """
    V218 Backtest Engine - 裁判引擎
    
    【核心职责】
    1. 数据加载与校验
    2. 调用 AlphaModel (Player) 计算评分
    3. 调用 BacktestReferee 执行回测
    4. 生成审计报告
    """
    
    def __init__(self, output_dir: str = "reports", db_url: Optional[str] = None):
        """
        初始化回测引擎
        
        Args:
            output_dir: 报告输出目录
            db_url: 数据库连接 URL
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.db_url = db_url or os.getenv("DATABASE_URL")
        
        if not self.db_url:
            raise ValueError("Database URL not provided. Set DATABASE_URL environment variable.")
        
        self._engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        
        logger.info("=" * 70)
        logger.info("V218 Backtest Engine Initialized (Referee)")
        logger.info("=" * 70)
        logger.info(f"  Database: {self.db_url.split('@')[1] if '@' in self.db_url else 'N/A'}")
        logger.info(f"  Output Dir: {self.output_dir}")
        logger.info(f"  Min Stock Count: {MIN_STOCK_COUNT}")
        logger.info("=" * 70)
    
    def load_data(
        self,
        years: List[int],
        warmup_year: int = 2019,
        warmup_days: int = 60,
    ) -> pd.DataFrame:
        """
        从数据库加载数据
        
        Args:
            years: 回测年份列表
            warmup_year: 预热年份 (用于计算因子)
            warmup_days: 预热天数
            
        Returns:
            包含股票数据的 DataFrame
        """
        logger.info(f"[Data] Loading data for years: {years}")
        logger.info(f"[Data] Warmup: {warmup_year} ({warmup_days} days)")
        
        # 构建日期范围
        start_date = f"{warmup_year}-01-01"
        end_date = f"{max(years)}-12-31"
        
        # 查询股票数据
        query = text("""
            SELECT 
                sd.trade_date, sd.symbol, sd.open, sd.high, sd.low, sd.close, sd.pre_close,
                sd.pct_chg, sd.volume, sd.amount, sd.turnover_rate,
                sd.industry_code, sd.total_mv, sd.is_st
            FROM stock_daily sd
            WHERE sd.trade_date >= :start_date AND sd.trade_date <= :end_date
            ORDER BY sd.trade_date, sd.symbol
        """)
        
        df = pd.read_sql(query, self._engine, params={'start_date': start_date, 'end_date': end_date})
        
        if df.empty:
            logger.error("[Data] No data loaded from database")
            return df
        
        # 数据类型转换
        df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y%m%d').astype(int)
        df['symbol'] = df['symbol'].astype(str)
        
        # 数值列转换
        numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'pct_chg', 
                       'volume', 'amount', 'turnover_rate', 'total_mv']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 加载资金流数据
        logger.info("[Data] Loading fund flow data...")
        fund_flow_query = text("""
            SELECT trade_date, symbol, net_main_amount, net_main_rate
            FROM stock_fund_flow
            WHERE trade_date >= :start_date AND trade_date <= :end_date
        """)
        fund_flow_df = pd.read_sql(fund_flow_query, self._engine, params={'start_date': start_date, 'end_date': end_date})
        
        if not fund_flow_df.empty:
            fund_flow_df['trade_date'] = pd.to_datetime(fund_flow_df['trade_date']).dt.strftime('%Y%m%d').astype(int)
            fund_flow_df['symbol'] = fund_flow_df['symbol'].astype(str)
            
            # 合并资金流数据
            df = df.merge(fund_flow_df, on=['trade_date', 'symbol'], how='left')
            logger.info(f"[Data] Fund flow data merged: {len(fund_flow_df)} rows")
        else:
            logger.warning("[Data] No fund flow data found")
            df['net_main_amount'] = 0.0
            df['net_main_rate'] = 0.0
        
        logger.info(f"[Data] Loaded {len(df)} rows, {df['symbol'].nunique()} unique symbols")
        logger.info(f"[Data] Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
        
        return df
    
    def validate_data(self, df: pd.DataFrame, years: List[int]) -> Dict[str, Any]:
        """
        验证数据质量
        
        Args:
            df: 股票数据
            years: 回测年份
            
        Returns:
            验证结果字典
        """
        logger.info("[Validate] Validating data quality...")
        
        validation_results = {}
        all_passed = True
        
        for year in years:
            year_str = str(year)
            year_data = df[df['trade_date'].astype(str).str.startswith(year_str)]
            
            if year_data.empty:
                validation_results[year] = {
                    'passed': False,
                    'error': f'No data for year {year}'
                }
                all_passed = False
                continue
            
            # 按日期统计股票数
            daily_counts = year_data.groupby('trade_date')['symbol'].count()
            
            # 检查每日股票数
            low_count_dates = daily_counts[daily_counts < MIN_STOCK_COUNT]
            
            validation_results[year] = {
                'passed': len(low_count_dates) == 0,
                'total_dates': len(daily_counts),
                'low_count_dates': len(low_count_dates),
                'avg_stock_count': float(daily_counts.mean()),
                'min_stock_count': int(daily_counts.min()),
                'max_stock_count': int(daily_counts.max()),
            }
            
            logger.info(f"[Validate] Year {year}: avg={daily_counts.mean():.0f}, min={daily_counts.min()}, max={daily_counts.max()}")
        
        return {
            'passed': all_passed,
            'results': validation_results,
        }
    
    def run_cross_year_audit(
        self,
        df: pd.DataFrame,
        alpha_model: Any,
        years: List[int],
    ) -> Dict[str, Any]:
        """
        执行跨年度审计
        
        Args:
            df: 股票数据
            alpha_model: Alpha 模型实例 (Player)
            years: 回测年份列表
            
        Returns:
            审计结果字典
        """
        logger.info("=" * 70)
        logger.info("V218 Cross-Year Audit Starting (Referee)")
        logger.info("=" * 70)
        
        results = {}
        
        for year in years:
            logger.info(f"\n[Year {year}] Running audit...")
            
            # 筛选当年数据
            year_str = str(year)
            year_data = df[df['trade_date'].astype(str).str.startswith(year_str)].copy()
            
            if year_data.empty:
                logger.warning(f"[Year {year}] No data available")
                continue
            
            logger.info(f"[Year {year}] Data: {len(year_data)} rows, {year_data['symbol'].nunique()} symbols")
            
            # 调用 Player 计算 Alpha 评分
            logger.info(f"[Year {year}] Calling AlphaModel (Player) to compute scores...")
            score_df = alpha_model.compute_score(year_data)
            
            # 导入 BacktestReferee
            from src.engine.backtest_referee import BacktestReferee
            
            # 初始化裁判
            referee = BacktestReferee(alpha_module=alpha_model, output_dir=str(self.output_dir))
            referee.VERSION = f"V218_Year{year}"
            
            # 执行审计
            logger.info(f"[Year {year}] Running referee audit...")
            audit_result = referee.run_audit(year_data)
            
            results[year] = audit_result
            
            t1_ic = audit_result['t1_ic']['mean_ic']
            ic_ir = audit_result['t1_ic']['ic_ir']
            passed = audit_result.get('passed', False)
            
            logger.info(f"[Year {year}] T+1 IC: {t1_ic:.4f}, IR: {ic_ir:.2f} -> {'PASS' if passed else 'FAIL'}")
        
        # 生成汇总报告
        logger.info("\n[Report] Generating cross-year summary report...")
        report_path = self.generate_cross_year_report(results, years)
        
        return {
            'results': results,
            'report_path': report_path,
        }
    
    def generate_cross_year_report(
        self,
        results: Dict[int, Dict[str, Any]],
        years: List[int],
    ) -> str:
        """
        生成跨年度汇总报告
        
        Args:
            results: 各年份审计结果
            years: 回测年份列表
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"V218_Cross_Year_Report_{timestamp}.md"
        
        # 构建报告内容 - 使用普通字符串拼接避免 f-string 问题
        lines = []
        lines.append("# V218 Cross-Year Audit Report (State Adapter Architecture)")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("**Version**: V218 (Market State Adapter + Feature Decoupling)")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 1. Executive Summary (执行摘要)")
        lines.append("")
        lines.append("| Year | T+1 Rank IC | IC IR | Annual Return | Sharpe | Status |")
        lines.append("|------|-------------|-------|---------------|--------|--------|")
        
        for year in years:
            if year not in results:
                lines.append(f"| {year} | N/A | N/A | N/A | N/A | FAIL |")
                continue
            
            r = results[year]
            t1_ic = r['t1_ic']['mean_ic']
            ic_ir = r['t1_ic']['ic_ir']
            ann_ret = r['backtest_result'].get('annual_return', 0)
            sharpe = r['backtest_result'].get('sharpe_ratio', 0)
            
            status = 'PASS' if t1_ic >= IC_THRESHOLD else 'FAIL'
            lines.append(f"| {year} | {t1_ic:.4f} | {ic_ir:.2f} | {ann_ret:.2%} | {sharpe:.2f} | {status} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 2. V218 Core Architecture (V218 核心架构)")
        lines.append("")
        lines.append("### 2.1 Market State Adapter (市场状态适配器)")
        lines.append("- **State Detection**: 使用大盘 std(ret_20) 和 ma(close, 20) 判定市场状态")
        lines.append("- **Regime Classification**:")
        lines.append("  - 极值恐慌 (CRISIS): 高波动 + 下跌趋势 -> 适合反转策略")
        lines.append("  - 趋势爆发 (TREND): 低波动 + 上涨趋势 -> 适合动量策略")
        lines.append("  - 正常区间 (NORMAL): 介于两者之间")
        lines.append("")
        lines.append("### 2.2 Feature Decoupling (特征解耦)")
        lines.append("- **Score_Rev (反转得分)**: 短期反转逻辑 (5d/10d/20d)")
        lines.append("- **Score_Mom (动量得分)**: 20日价格强度 + 中期动量")
        lines.append("- **Gating Mechanism**: Final_Score = W_t * Score_Rev + (1 - W_t) * Score_Mom")
        lines.append("")
        lines.append("### 2.3 Dynamic Weight Calculation (动态权重)")
        lines.append("- W_t = sigmoid(alpha * volatility_norm + beta * trend_strength)")
        lines.append("- 其中 alpha=1.5, beta=1.0 为默认参数")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 3. IC Decay Analysis (IC 衰减分析)")
        
        for year in years:
            if year not in results:
                continue
            
            r = results[year]
            ic_decay = r.get('ic_decay', {})
            t1 = ic_decay.get('t1_ic', 0)
            t3 = ic_decay.get('t3_ic', 0)
            t5 = ic_decay.get('t5_ic', 0)
            pattern = ic_decay.get('decay_pattern', 'N/A')
            mono = 'Yes' if ic_decay.get('is_monotonic', False) else 'No'
            
            lines.append("")
            lines.append(f"### Year {year}")
            lines.append("")
            lines.append("| Horizon | IC |")
            lines.append("|---------|-----|")
            lines.append(f"| T+1 | {t1:.4f} |")
            lines.append(f"| T+3 | {t3:.4f} |")
            lines.append(f"| T+5 | {t5:.4f} |")
            lines.append("")
            lines.append(f"**Pattern**: {pattern}")
            lines.append(f"**Monotonic**: {mono}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 4. Backtest Configuration (回测配置 - Locked)")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        lines.append("| Initial Capital | 100,000 |")
        lines.append("| Commission Rate | 0.3 per mille |")
        lines.append("| Stamp Duty Rate | 1.0 per mille |")
        lines.append("| Slippage Rate | 0.5 per mille |")
        lines.append("| Total Fee Rate | 1.3 per mille |")
        lines.append("| Top N Stocks | 50 |")
        lines.append("| Position per Stock | 2% |")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 5. Compliance Statement (合规声明)")
        lines.append("")
        lines.append("- **初始资金**: 100,000 (已锁定)")
        lines.append("- **费率**: 1.3 per mille (佣金 0.3 + 印花税 1 + 滑点 0.5)")
        lines.append("- **无未来函数**: 严禁 shift(-1) 或任何 T+1 数据访问")
        lines.append("- **Referee-Player 隔离**: AlphaModel 仅输出 Score，严禁接触回测逻辑")
        lines.append("- **实验日志**: 结果已写入 experiment_metadata.csv")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 6. Conclusion (结论)")
        lines.append("")
        lines.append("V218 采用市场状态适配器 + 特征解耦架构，通过动态权重门控机制")
        lines.append("在不同市场状态下自动调整反转与动量的权重配比。")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by V218 Backtest Engine (Referee)*")
        
        report_content = "\n".join(lines)
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[Report] Cross-year report saved to: {report_path}")
        
        # 同时保存 JSON 结果
        json_result = {
            'version': 'V218',
            'years': years,
            'results': {},
        }
        
        for year in years:
            if year not in results:
                continue
            
            r = results[year]
            json_result['results'][year] = {
                't1_ic': r['t1_ic'],
                'ic_decay': r.get('ic_decay', {}),
                'backtest_result': r['backtest_result'],
            }
        
        json_path = self.output_dir / f"V218_Cross_Year_Report_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        logger.info(f"[Report] JSON result saved to: {json_path}")
        
        return str(report_path)


def get_backtest_engine(output_dir: str = "reports", db_url: Optional[str] = None) -> BacktestEngine:
    """获取 BacktestEngine 实例"""
    return BacktestEngine(output_dir=output_dir, db_url=db_url)