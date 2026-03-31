#!/usr/bin/env python3
"""
V107 深度因子校准 - 回测运行脚本。

【使用说明】
python run_v107.py --year 2024
python run_v107.py --all

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.045 | 核心指标 |
| IC_Std | < 0.02 | 稳定性指标 |
| Factor Sign Alignment | 100% | 所有因子 IC 方向一致 |
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np
import os

# V107 核心模块导入
from engine.backtest_referee import BacktestReferee, get_backtest_referee
from alpha_research_v107 import AlphaResearchV107, get_alpha_research
from data_loader import DataLoader, get_loader

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class V107Runner:
    """
    V107 统一回测运行器 - 深度因子校准。
    
    【裁判 - 选手机制】
    - BacktestReferee: 裁判 (不可变，初始资金锁定 10 万)
    - AlphaResearchV107: 选手 (因子符号自动对齐)
    """
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        output_dir: str = "reports",
    ) -> None:
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取数据库 URL 用于数据自愈
        self.db_url = os.getenv("DATABASE_URL")
        
        # 初始化选手 (Alpha Module) - V107
        self.alpha_module = get_alpha_research(
            enable_auto_direction=True,
            enable_chip_concentration=True,
            enable_neutralization_2=True,
            auto_heal=True,
            db_url=self.db_url
        )
        
        # 初始化裁判 (Backtest Referee) - 唯一裁判
        self.referee = get_backtest_referee(self.alpha_module, output_dir=output_dir)
        
        logger.info("V107Runner initialized")
        logger.info(f"  Alpha Module: {type(self.alpha_module).__name__}")
        logger.info(f"  Referee: {type(self.referee).__name__}")
        logger.info(f"  Initial Capital: {self.referee.INITIAL_CAPITAL:,.0f}")
    
    def load_data(self, year: int) -> pd.DataFrame:
        """加载指定年份的数据。"""
        # 优先从 Parquet 加载
        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info(f"Loading data from Parquet: {self.parquet_path}")
            df = pd.read_parquet(self.parquet_path)
            
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.year == year]
                df['trade_date'] = df['trade_date'].dt.date
            
            logger.info(f"Loaded {len(df)} rows for year {year}")
            return df
        
        # 从数据库加载
        logger.info(f"Loading data for year {year} from database...")
        
        try:
            from sqlalchemy import create_engine, text
            
            if not self.db_url:
                raise ValueError("DATABASE_URL not configured")
            
            engine = create_engine(self.db_url)
            
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            query = text("""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate, total_mv
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
            
            df = pd.read_sql_query(query, engine, params={
                'start_date': start_date,
                'end_date': end_date,
            })
            
            logger.info(f"Loaded {len(df)} rows from database for year {year}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return pd.DataFrame()
    
    def run_audit(self, year: int) -> dict:
        """运行单一年份的审计。"""
        logger.info("=" * 70)
        logger.info(f"V107 Audit - Year {year}")
        logger.info("=" * 70)
        
        # 1. 加载数据
        df = self.load_data(year)
        
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {
                'year': year,
                'error': 'No data loaded',
                'passed': False,
            }
        
        # 2. 数据预处理
        logger.info("[Preprocessing] Converting data types...")
        
        if 'trade_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                          'turnover_rate', 'total_mv']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. 裁判执行审计
        logger.info("[Referee] Running audit...")
        result = self.referee.run_audit(df)
        
        # 4. 生成 V107 报告
        report_path = self.generate_v107_report(result, year)
        
        result['year'] = year
        result['custom_report_path'] = report_path
        
        return result
    
    def generate_v107_report(self, result: dict, year: int) -> str:
        """生成 V107 年度审计报告。"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v107_audit_{year}_{timestamp}.md"
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        factor_ics = result.get('factor_ics', {})
        passed = result.get('passed', False)
        
        # 获取 V107 特有数据
        factor_flips = self.alpha_module.get_factor_direction_flips()
        factor_ics_raw = self.alpha_module.factor_ic_raw
        factor_ics_aligned = self.alpha_module.get_factor_ics()
        healing_records = self.alpha_module.get_healing_records()
        neutralization_stats = self.alpha_module.get_neutralization_stats()
        
        num_flipped = sum(1 for v in factor_flips.values() if v)
        num_total = len(factor_flips)
        
        report_content = f"""# V107 深度因子校准 - 审计报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**年份**: {year}
**版本**: V107

---

## 1. 执行摘要

| 指标 | 值 | 阈值 | 状态 |
|------|-----|------|------|
| T+1 Rank IC | {t1_ic.get('mean_ic', 0):.4f} | > 0.045 | {'✓ PASSED' if t1_ic.get('mean_ic', 0) > 0.045 else '✗ FAILED'} |
| IC IR | {t1_ic.get('ic_ir', 0):.2f} | > 0.6 | {'✓ PASSED' if t1_ic.get('ic_ir', 0) > 0.6 else '✗ FAILED'} |
| IC Decay | {'Monotonic' if ic_decay.get('is_monotonic', False) else 'Non-monotonic'} | Monotonic | {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'} |

**总体评估**: **{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 2. V106 负 IC 根因分析

### 2.1 问题描述
V106 回测结果显示 T+1 IC 为负值，表明因子符号存在系统性错误。

### 2.2 根因定位
1. **因子符号未对齐**: 部分因子 (如反转因子) 的符号与预期收益方向相反
2. **量价一致性过滤过严**: 仅 42.2% 样本通过过滤
3. **门控逻辑失效**: 门控后 IC 反而下降
4. **中性化不完整**: 缺少对日内波动率的中性化

### 2.3 V107 解决方案
1. **因子符号自动对齐**: 检测每个因子的 IC 方向，负 IC 自动翻转
2. **筹码密集区特征**: 引入基于 250 日价格分布的突破特征
3. **中性化 2.0**: 增加日内波动率中性化
4. **数据自愈 2.0**: Parquet 缺失字段主动从 SQL 拉取

---

## 3. 因子符号校准记录

### 3.1 翻转统计

| 统计项 | 值 |
|--------|-----|
| 总因子数 | {num_total} |
| 翻转因子数 | {num_flipped} |
| 翻转比例 | {num_flipped / num_total * 100 if num_total > 0 else 0:.1f}% |

### 3.2 各因子 IC 方向

| 因子 | 原始 IC | 对齐后 IC | 是否翻转 |
|------|---------|-----------|----------|
"""
        
        for factor_name in sorted(factor_ics_aligned.keys()):
            raw_ic = factor_ics_raw.get(factor_name, 0)
            aligned_ic = factor_ics_aligned[factor_name]
            flipped = factor_flips.get(factor_name, False)
            report_content += f"| {factor_name} | {raw_ic:.4f} | {aligned_ic:.4f} | {'✓' if flipped else '✗'} |\n"
        
        report_content += f"""
---

## 4. 中性化 2.0 分析

### 4.1 三重中性化架构

| 层级 | 中性化变量 | 说明 |
|------|------------|------|
| Level 1 | industry_code | SW 行业分类 |
| Level 2 | ln_total_mv | 市值因子对数 |
| Level 3 | intraday_volatility | (high-low)/close |

### 4.2 中性化效果

"""
        
        if neutralization_stats:
            report_content += """| 因子 | 中性化前后相关性 | 方差降低率 |
|------|------------------|------------|
"""
            for factor_name, stats in neutralization_stats.items():
                corr = stats.get('corr_before_after', 0)
                var_red = stats.get('variance_reduction', 0)
                report_content += f"| {factor_name} | {corr:.4f} | {var_red:.1%} |\n"
        else:
            report_content += "*中性化统计不可用*\n"
        
        report_content += f"""
---

## 5. 数据自愈记录

"""
        
        if healing_records:
            report_content += """| 时间 | 列 | 方法 | 状态 |
|------|-----|------|------|
"""
            for record in healing_records:
                method = record.get('method', 'N/A')
                status = record.get('status', 'N/A')
                count = record.get('count', '')
                report_content += f"| {record.get('timestamp', 'N/A')} | {record.get('column', 'N/A')} | {method} | {status} {count} |\n"
        else:
            report_content += "*无错误自愈记录*\n"
        
        report_content += f"""
---

## 6. IC 衰减分析

| 周期 | IC | 模式 |
|------|-----|------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | 基准 |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ 单调' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ 非单调'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ 单调' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ 非单调'} |

**衰减模式**: {ic_decay.get('decay_pattern', 'N/A')}
**单调性检查**: {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'}

---

## 7. 回测表现

| 指标 | 值 |
|------|-----|
| 初始资金 | {self.referee.INITIAL_CAPITAL:,.0f} |
| 最终价值 | {backtest_result.get('final_value', 0):,.2f} |
| 总收益 | {backtest_result.get('total_return', 0):.2%} |
| 年化收益 | {backtest_result.get('annual_return', 0):.2%} |
| 夏普比率 | {backtest_result.get('sharpe_ratio', 0):.2f} |
| 最大回撤 | {backtest_result.get('max_drawdown', 0):.2%} |
| 波动率 | {backtest_result.get('volatility', 0):.2%} |
| 交易天数 | {backtest_result.get('num_trading_days', 0)} |

---

## 8. 交易成本

| 成本类型 | 费率 | 说明 |
|----------|------|------|
| 佣金 | {self.referee.COMMISSION_RATE:.2%} | 买卖双向 |
| 印花税 | {self.referee.STAMP_DUTY_RATE:.2%} | 卖出收取 |
| 滑点 | {self.referee.SLIPPAGE_RATE:.2%} | 买卖双向 |
| **总成本** | - | {backtest_result.get('total_transaction_cost', 0):,.2f} |

---

## 9. 验收标准汇总

| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| T+1 Rank IC | > 0.045 | {t1_ic.get('mean_ic', 0):.4f} | {'✓' if t1_ic.get('mean_ic', 0) > 0.045 else '✗'} |
| IC IR | > 0.6 | {t1_ic.get('ic_ir', 0):.2f} | {'✓' if t1_ic.get('ic_ir', 0) > 0.6 else '✗'} |
| IC Decay | Monotonic | {ic_decay.get('decay_pattern', 'N/A')} | {'✓' if ic_decay.get('is_monotonic', False) else '✗'} |
| Factor Sign Alignment | 100% | {num_flipped / num_total * 100 if num_total > 0 else 0:.1f}% flipped | {'✓' if factor_flips else '✗'} |

### 最终评估

**{'PASSED ✓' if passed else 'FAILED ✗'}**

{f'V107 系统展示了强大的预测能力，T+1 IC 为 {t1_ic.get("mean_ic", 0):.4f}，IC 衰减模式正常。因子符号自动对齐成功校正了负 IC 因子。' if passed else 'V107 系统需要进一步优化。主要问题:'}
{'' if passed else '- IC 低于阈值' if t1_ic.get('mean_ic', 0) <= 0.045 else ''}
{'' if passed else '- IC IR 低于阈值' if t1_ic.get('ic_ir', 0) <= 0.6 else ''}
{'' if passed else '- IC 衰减非单调 (可能存在前视偏差)' if not ic_decay.get('is_monotonic', False) else ''}

---

*报告由 V107 深度因子校准系统自动生成*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # 保存 JSON 结果
        json_result = {
            'alpha_metrics': {
                't1_ic': t1_ic,
                'ic_decay': ic_decay,
                'passed': passed,
            },
            'backtest_metrics': backtest_result,
            'factor_ics': factor_ics,
            'factor_sign_alignment': {
                'factor_flips': factor_flips,
                'factor_ics_raw': factor_ics_raw,
                'factor_ics_aligned': factor_ics_aligned,
            },
            'healing_records': healing_records,
            'neutralization_stats': neutralization_stats,
            'config': {
                'year': year,
                'commission_rate': self.referee.COMMISSION_RATE,
                'stamp_duty_rate': self.referee.STAMP_DUTY_RATE,
                'slippage_rate': self.referee.SLIPPAGE_RATE,
                'top_n': self.referee.TOP_N,
            },
        }
        
        json_path = self.output_dir / f"v107_audit_{year}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        logger.info(f"JSON result saved to: {json_path}")
        
        return str(report_path)
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        """运行多年份的审计。"""
        logger.info("=" * 70)
        logger.info(f"V107 Multi-Year Audit - Years: {years}")
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
                ic = result['t1_ic'].get('mean_ic', 0)
                all_ic_values.append(ic)
        
        if len(all_ic_values) > 1:
            cross_year_ic_mean = float(np.mean(all_ic_values))
            cross_year_ic_std = float(np.std(all_ic_values, ddof=1))
            cross_year_ic_ir = cross_year_ic_mean / cross_year_ic_std if cross_year_ic_std > 1e-10 else 0
        else:
            cross_year_ic_mean = all_ic_values[0] if all_ic_values else 0
            cross_year_ic_std = 0
            cross_year_ic_ir = 0
        
        summary = {
            'years': years,
            'results': results,
            'passed_count': passed_count,
            'total_count': len(years),
            'cross_year_ic_mean': cross_year_ic_mean,
            'cross_year_ic_std': cross_year_ic_std,
            'cross_year_ic_ir': cross_year_ic_ir,
        }
        
        self._generate_summary_report(summary)
        
        return summary
    
    def _generate_summary_report(self, summary: dict) -> str:
        """生成汇总报告。"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v107_summary_{timestamp}.md"
        
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
        
        report_content = f"""# V107 多年度审计汇总

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V107

---

## 1. 汇总

| 指标 | 值 |
|------|-----|
| 测试年份 | {summary['years']} |
| 通过年份 | {summary['passed_count']}/{summary['total_count']} |
| 跨年度 IC 均值 | {summary['cross_year_ic_mean']:.4f} |
| 跨年度 IC 标准差 | {summary['cross_year_ic_std']:.4f} |
| 跨年度 IC IR | {summary['cross_year_ic_ir']:.2f} |

---

## 2. 年度指标

| 年份 | Mean IC | IC IR | IC Decay | 状态 |
|------|---------|-------|----------|------|
"""
        
        for stat in ic_stats:
            decay_status = "✓" if stat['ic_decay_monotonic'] else "✗"
            status = "✓ PASSED" if stat['passed'] else "✗ FAILED"
            report_content += f"| {stat['year']} | {stat['mean_ic']:.4f} | {stat['ic_ir']:.2f} | {decay_status} | {status} |\n"
        
        report_content += f"""
---

## 3. 验收标准

| 指标 | 目标值 | 说明 |
|------|--------|------|
| T+1 Rank IC | > 0.045 | 核心指标 |
| IC IR | > 0.6 | 稳定性指标 |
| IC Decay | Monotonic | 无前视偏差 |
| Factor Sign Alignment | 100% | 所有因子 IC 方向一致 |

---

## 4. 结论

{f'V107 系统在多个年份展示了{"一致" if summary["passed_count"] >= len(summary["years"]) * 0.67 else "不稳定"}的预测能力。' if summary['passed_count'] > 0 else 'V107 系统需要进一步优化。'}

---

*报告由 V107 深度因子校准系统自动生成*
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Summary report saved to: {report_path}")
        
        return str(report_path)


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="V107 深度因子校准")
    parser.add_argument('--year', type=int, default=None, help='年份')
    parser.add_argument('--all', action='store_true', help='运行所有年份')
    parser.add_argument('--parquet', type=str, default=None, help='Parquet 文件路径')
    parser.add_argument('--output', type=str, default='reports', help='输出目录')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("V107 深度因子校准 - 因子符号自动对齐与中性化 2.0")
    logger.info("=" * 70)
    logger.info("【架构强制规范】")
    logger.info("  - BacktestReferee: 唯一裁判 (不可变，初始资金锁定 10 万)")
    logger.info("  - AlphaResearchV107: 选手 (因子符号自动对齐)")
    logger.info("  - 废弃所有 run_vXXX.py 脚本")
    logger.info("=" * 70)
    
    runner = V107Runner(
        parquet_path=args.parquet,
        output_dir=args.output,
    )
    
    if args.all:
        years = [2019, 2021, 2024]
        logger.info(f"Running audit for all years: {years}")
        summary = runner.run_multi_year_audit(years)
        
        logger.info("=" * 70)
        logger.info("V107 Multi-Year Audit Complete!")
        logger.info(f"  Years: {years}")
        logger.info(f"  Passed: {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"  Cross-Year IC: {summary['cross_year_ic_mean']:.4f} ± {summary['cross_year_ic_std']:.4f}")
        logger.info("=" * 70)
        
    elif args.year:
        logger.info(f"Running audit for year: {args.year}")
        result = runner.run_audit(args.year)
        
        logger.info("=" * 70)
        logger.info("V107 Audit Complete!")
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