#!/usr/bin/env python3
"""
V153 回测运行脚本 - Orthogonal-Residual-Alpha (ORA).

【使用说明】
    python run_v153.py --year 2024
    python run_v153.py --all

【V153 核心使命】
1. Adaptive Lead-Lag Correction: 互信息分析，仅保留领先因子
2. Cross-Sectional Volatility Weighting: 截面 Z-Score 归一化
3. Orthogonal Residual Mining: 提取正交残差，确保增量信息
4. Strict PAC: 滚动 IC 符号，严禁偷看未来
5. 目标指标：T+1 Rank IC > 0.055, IC_IR > 0.55, IC Decay 严格单调递减
"""

import sys
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np

from engine.backtest_referee import BacktestReferee, get_backtest_referee
from alpha_research_v153 import (
    AlphaResearchV153, 
    get_alpha_research as get_alpha_research_v153,
    LEAD_LAG_THRESHOLD,
    ORM_CORE_FACTOR,
    ROLLING_WINDOW,
    IC_WEIGHT_WINDOW,
)

load_dotenv()

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


class V153Runner:
    """V153 统一回测运行器 - Orthogonal-Residual-Alpha (ORA)"""
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        output_dir: str = "reports",
    ) -> None:
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        db_url = os.getenv("DATABASE_URL")
        
        # 初始化 V153 Alpha 模块
        self.alpha_module = get_alpha_research_v153(
            ic_threshold=0.0001,
            n_factors=8,
            n_bins=10,
            enable_ensemble=True,
            enable_pac=True,
            enable_lead_lag=True,
            enable_cs_volatility=True,
            enable_orm=True,
            enable_sector_neutral=True,
            auto_heal=True,
            db_url=db_url,
        )
        
        # 初始化裁判
        self.referee = get_backtest_referee(self.alpha_module, output_dir=output_dir)
        self.referee.VERSION = "V153"
        
        logger.info("=" * 70)
        logger.info("V153Runner Initialized - Orthogonal-Residual-Alpha (ORA)")
        logger.info("=" * 70)
        logger.info(f"  Alpha Module: {type(self.alpha_module).__name__}")
        logger.info(f"  Referee: {type(self.referee).__name__}")
        logger.info(f"  Initial Capital: {self.referee.INITIAL_CAPITAL:,.0f}")
        logger.info(f"  Lead-Lag Threshold: {LEAD_LAG_THRESHOLD}")
        logger.info(f"  ORM Core Factor: {ORM_CORE_FACTOR}")
        logger.info(f"  Rolling IC Window: {ROLLING_WINDOW}")
        logger.info(f"  IC Weighting Window: {IC_WEIGHT_WINDOW}")
        logger.info(f"  Target IR: 0.55 (V151: ~0.44-0.50)")
        logger.info(f"  Target IC Decay: T+1 > T+3 > T+5 (strictly monotonic)")
        logger.info("=" * 70)
    
    def load_data(self, year: int) -> pd.DataFrame:
        """加载指定年份的数据"""
        # 优先从 Parquet 加载
        parquet_path = self.parquet_path or "data/parquet/stock_data_2024_2026.parquet"
        if Path(parquet_path).exists():
            logger.info(f"Loading data from Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.year == year]
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"Loaded {len(df)} rows for year {year}")
            return df
        
        # 从数据库加载
        logger.info(f"Loading data for year {year} from database...")
        
        try:
            from sqlalchemy import create_engine, text
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
            
            engine = create_engine(db_url)
            
            query = text("""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       `change`, pct_chg, volume, amount
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
            
            df = pd.read_sql_query(query, engine, params={
                'start_date': f"{year}0101",
                'end_date': f"{year}1231",
            })
            
            logger.info(f"Loaded {len(df)} rows from database for year {year}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def run_audit(self, year: int) -> dict:
        """运行单一年份的审计"""
        logger.info("=" * 70)
        logger.info(f"V153 Audit - Year {year}")
        logger.info("=" * 70)
        
        df = self.load_data(year)
        
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {'year': year, 'error': 'No data loaded', 'passed': False}
        
        # 数据预处理
        logger.info("[Preprocessing] Converting data types...")
        
        if 'trade_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 裁判执行审计
        logger.info("[Referee] Running audit...")
        result = self.referee.run_audit(df)
        
        # 生成报告
        report_path = self.generate_v153_report(result, year)
        
        result['year'] = year
        result['custom_report_path'] = report_path
        
        return result
    
    def generate_v153_report(self, result: dict, year: int) -> str:
        """生成 V153 年度审计报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v153_ora_audit_{year}_{timestamp}.md"
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        passed = result.get('passed', False)
        
        # 获取 V153 模块统计
        factor_ics = self.alpha_module.get_factor_ics()
        selected_factors = self.alpha_module.get_selected_factors()
        lead_lag_stats = self.alpha_module.get_lead_lag_stats()
        orm_stats = self.alpha_module.get_orm_stats()
        audit_log = self.alpha_module.get_audit_log()[-10:]
        
        # V151 对比
        v151_ir = 0.50
        v153_ir = t1_ic.get('ic_ir', 0)
        ir_improvement = (v153_ir - v151_ir) / (abs(v151_ir) + 1e-10)
        target_met = v153_ir >= 0.55
        
        # 构建因子 IC 表格
        factor_ic_info = ""
        if factor_ics:
            for factor_name, ic in sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True):
                selected = "✓" if factor_name in selected_factors else ""
                factor_ic_info += f"| {factor_name} | {ic:.4f} | {selected} |\n"
        
        # 构建审计日志
        audit_info = ""
        for log in audit_log:
            audit_info += f"- {log.get('action', '')}: {log.get('details', '')}\n"
        
        # 构建领先因子统计
        lead_factor_info = ""
        if lead_lag_stats:
            lead_scores = lead_lag_stats.get('lead_scores', {})
            lead_factors = lead_lag_stats.get('lead_factors', [])
            for factor, score in sorted(lead_scores.items(), key=lambda x: x[1], reverse=True):
                is_lead = "✓" if factor in lead_factors else ""
                lead_factor_info += f"| {factor} | {score:.2f} | {is_lead} |\n"
        
        report_content = f"""# V153 Alpha Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Year**: {year}
**Architecture**: Referee-Player (裁判 - 选手)
**Version**: V153 Orthogonal-Residual-Alpha (ORA)

---

## 1. Executive Summary (执行摘要)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC | {t1_ic.get('mean_ic', 0):.4f} | > 0.055 | {'✓ PASSED' if t1_ic.get('mean_ic', 0) > 0.055 else '✗ FAILED'} |
| IC IR | {t1_ic.get('ic_ir', 0):.2f} | > 0.55 | {'✓ PASSED' if t1_ic.get('ic_ir', 0) > 0.55 else '✗ FAILED'} |
| IC Decay | {'Monotonic' if ic_decay.get('is_monotonic', False) else 'Non-monotonic'} | Monotonic | {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'} |
| Lead Factor Ratio | > {LEAD_LAG_THRESHOLD} | - | - |

**Overall Assessment**: **{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 2. V153 Core Features (V153 核心特性)

### 2.1 Adaptive Lead-Lag Correction

| Component | Value |
|-----------|-------|
| Threshold | {LEAD_LAG_THRESHOLD} |
| Max Lag | 5 days |
| Method | MI_Lag_1 / MI_Lag_5 |

### 2.2 Lead Factor Selection

| Factor | Lead Score | Selected |
|--------|------------|----------|
{lead_factor_info if lead_factor_info else "*No lead factor data*"}

### 2.3 Orthogonal Residual Mining (ORM)

| Component | Value |
|-----------|-------|
| Core Factor | {ORM_CORE_FACTOR} |
| Method | Residual_i = Factor_i - β_i * CoreFactor |

### 2.4 Cross-Sectional Volatility Weighting

| Component | Value |
|-----------|-------|
| Window | {IC_WEIGHT_WINDOW} days |
| Method | Z-Score = (Factor - Mean) / Std |

### 2.5 Top Selected Factors

| Factor | IC | Selected |
|--------|-----|----------|
{factor_ic_info if factor_ic_info else "*No factor data*"}

### 2.6 Audit Log

{audit_info if audit_info else "*No audit log*"}

---

## 3. IC Decay Analysis (IC 衰减分析)

| Horizon | IC | Pattern |
|---------|-----|---------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | Baseline |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ Non-monotonic'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ Non-monotonic'} |

**Decay Pattern**: {ic_decay.get('decay_pattern', 'N/A')}

---

## 4. Backtest Performance (回测表现)

| Metric | Value |
|--------|-------|
| Initial Capital | {self.referee.INITIAL_CAPITAL:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {backtest_result.get('total_return', 0):.2%} |
| Annual Return | {backtest_result.get('annual_return', 0):.2%} |
| Sharpe Ratio | {backtest_result.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {backtest_result.get('max_drawdown', 0):.2%} |

---

## 5. V153 vs V151 Comparison

| Metric | V151 | V153 | Improvement |
|--------|------|------|-------------|
| T+1 IC | - | {t1_ic.get('mean_ic', 0):.4f} | - |
| IC IR | {v151_ir:.2f} | {v153_ir:.2f} | {ir_improvement:+.2%} |
| Lead-Lag Correction | No | Yes | ✓ |
| CS Volatility Weighting | No | Yes | ✓ |
| ORM | No | Yes | ✓ |

**Target (IR >= 0.55)**: {'MET ✓' if target_met else 'NOT MET ✗'}

---

## 6. Conclusion (结论)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > 0.055 | {t1_ic.get('mean_ic', 0):.4f} | {'✓' if t1_ic.get('mean_ic', 0) > 0.055 else '✗'} |
| IC IR | > 0.55 | {v153_ir:.2f} | {'✓' if v153_ir >= 0.55 else '✗'} |
| IC Decay | Monotonic | {ic_decay.get('decay_pattern', 'N/A')} | {'✓' if ic_decay.get('is_monotonic', False) else '✗'} |

**{'PASSED ✓' if passed else 'FAILED ✗'}**

---

*Report generated by V153 Unified Main Entry (Orthogonal-Residual-Alpha)*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # 保存 JSON 结果
        json_result = {
            'alpha_metrics': {'t1_ic': t1_ic, 'ic_decay': ic_decay, 'passed': passed},
            'backtest_metrics': backtest_result,
            'factor_ics': factor_ics,
            'selected_factors': selected_factors,
            'lead_lag_stats': lead_lag_stats,
            'orm_stats': orm_stats,
            'v151_comparison': {
                'v151_ir': v151_ir,
                'v153_ir': v153_ir,
                'ir_improvement': ir_improvement,
                'target_met': target_met,
            },
            'config': {
                'year': year,
                'lead_lag_threshold': LEAD_LAG_THRESHOLD,
                'orm_core_factor': ORM_CORE_FACTOR,
                'rolling_window': ROLLING_WINDOW,
                'ic_weight_window': IC_WEIGHT_WINDOW,
            },
        }
        
        json_path = self.output_dir / f"v153_ora_audit_{year}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        return str(report_path)
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        """运行多年份的审计"""
        logger.info("=" * 70)
        logger.info(f"V153 Multi-Year Audit - Years: {years}")
        logger.info("=" * 70)
        
        results = []
        passed_count = 0
        all_ic_values = []
        
        for year in years:
            result = self.run_audit(year)
            results.append(result)
            if result.get('passed', False):
                passed_count += 1
            if 't1_ic' in result:
                all_ic_values.append(result['t1_ic'].get('mean_ic', 0))
        
        # 跨年度统计
        cross_year_ic_mean = float(np.mean(all_ic_values)) if all_ic_values else 0
        cross_year_ic_std = float(np.std(all_ic_values, ddof=1)) if len(all_ic_values) > 1 else 0
        cross_year_ic_ir = cross_year_ic_mean / cross_year_ic_std if cross_year_ic_std > 1e-10 else 0
        
        summary = {
            'years': years,
            'results': results,
            'passed_count': passed_count,
            'total_count': len(years),
            'cross_year_ic_mean': cross_year_ic_mean,
            'cross_year_ic_std': cross_year_ic_std,
            'cross_year_ic_ir': cross_year_ic_ir,
        }
        
        # 生成汇总报告
        self._generate_summary_report(summary)
        
        return summary
    
    def _generate_summary_report(self, summary: dict) -> str:
        """生成汇总报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v153_ora_summary_{timestamp}.md"
        
        results = summary.get('results', [])
        
        ic_stats = []
        for r in results:
            if 't1_ic' in r:
                t1_ic = r['t1_ic']
                ic_decay = r.get('ic_decay', {})
                ic_stats.append({
                    'year': r.get('year', 'N/A'),
                    'mean_ic': t1_ic.get('mean_ic', 0),
                    'ic_ir': t1_ic.get('ic_ir', 0),
                    'ic_decay_monotonic': ic_decay.get('is_monotonic', False),
                    'passed': r.get('passed', False),
                })
        
        report_content = f"""# V153 Multi-Year Audit Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Architecture**: Referee-Player (裁判 - 选手)
**Version**: V153 Orthogonal-Residual-Alpha (ORA)

---

## 1. Overall Summary (汇总)

| Metric | Value |
|--------|-------|
| Years Tested | {summary['years']} |
| Passed | {summary['passed_count']}/{summary['total_count']} |
| Cross-Year IC Mean | {summary['cross_year_ic_mean']:.4f} |
| Cross-Year IC Std | {summary['cross_year_ic_std']:.4f} |
| Cross-Year IC IR | {summary['cross_year_ic_ir']:.2f} |

---

## 2. Year-by-Year Metrics (年度指标)

| Year | Mean IC | IC IR | IC Decay | Status |
|------|---------|-------|----------|--------|
"""
        
        for stat in ic_stats:
            decay_status = "✓" if stat['ic_decay_monotonic'] else "✗"
            status = "✓ PASSED" if stat['passed'] else "✗ FAILED"
            report_content += f"| {stat['year']} | {stat['mean_ic']:.4f} | {stat['ic_ir']:.2f} | {decay_status} | {status} |\n"
        
        report_content += f"""
---

## 3. Acceptance Criteria (验收标准)

| Metric | Target | Description |
|--------|--------|-------------|
| T+1 Rank IC | > 0.055 | 核心指标：预测能力 |
| IC IR | > 0.55 | 稳定性指标 |
| IC Decay | Monotonic | 无前视偏差 |
| Lead Factor Ratio | > {LEAD_LAG_THRESHOLD} | 领先因子筛选 |

---

## 4. Conclusion (结论)

{f'The V153 system has demonstrated {"consistent" if summary["passed_count"] >= len(summary["years"]) * 0.67 else "mixed"} predictive power across multiple years.' if summary['passed_count'] > 0 else 'The V153 system needs further optimization to achieve consistent predictive power.'}

---

*Report generated by V153 Unified Main Entry (Orthogonal-Residual-Alpha)*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Summary report saved to: {report_path}")
        
        return str(report_path)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(description="V153 Backtest Runner - Orthogonal-Residual-Alpha (ORA)")
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Year to run audit (e.g., 2021, 2024)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run audit for all years (2021, 2024)'
    )
    parser.add_argument(
        '--parquet',
        type=str,
        default=None,
        help='Path to Parquet data file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports',
        help='Output directory for reports'
    )
    
    args = parser.parse_args()
    
    runner = V153Runner(
        parquet_path=args.parquet,
        output_dir=args.output,
    )
    
    if args.all:
        years = [2021, 2024]
        logger.info(f"Running V153 audit for all years: {years}")
        summary = runner.run_multi_year_audit(years)
        
        logger.info("=" * 70)
        logger.info("V153 Multi-Year Audit Complete!")
        logger.info(f"  Years: {years}")
        logger.info(f"  Passed: {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"  Cross-Year IC: {summary['cross_year_ic_mean']:.4f} ± {summary['cross_year_ic_std']:.4f}")
        logger.info(f"  Cross-Year IC IR: {summary['cross_year_ic_ir']:.2f}")
        logger.info(f"  Target (IR >= 0.55): {'MET ✓' if summary['cross_year_ic_ir'] >= 0.55 else 'NOT MET ✗'}")
        logger.info("=" * 70)
        
    elif args.year:
        logger.info(f"Running V153 audit for year: {args.year}")
        result = runner.run_audit(args.year)
        
        logger.info("=" * 70)
        logger.info("V153 Audit Complete!")
        logger.info(f"  Year: {args.year}")
        logger.info(f"  Status: {'PASSED ✓' if result.get('passed', False) else 'FAILED ✗'}")
        logger.info(f"  Report: {result.get('custom_report_path', 'N/A')}")
        logger.info("=" * 70)
        
    else:
        parser.print_help()
        logger.warning("Please specify --year or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()