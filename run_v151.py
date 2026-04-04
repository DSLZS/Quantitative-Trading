#!/usr/bin/env python3
"""
V151 统一回测运行器 - Latency-Corrected-Alpha (LCA).

【裁判 - 选手机制】
- BacktestReferee: 唯一裁判 (不可变，初始资金锁定 10 万)
- AlphaResearchV151: 选手 (LCA + Rolling PAC + Lead-Signal)

【V151 核心改进】
1. LCA-EMA: α=0.8 (新信号 80%, 旧信号 20%) - 解决 IC 衰减反转
2. Rolling PAC: Rolling_IC_Sign(window=20) 严格防偷看未来
3. Lead-Signal: d(Factor)/dt 一阶差分提升反应速度
4. Volatility-Standardized IC: 冲击 IR > 0.55
5. DataHealer: NaN/Inf 自动修复 (ffill/中位数)

【目标指标】
- T+1 Rank IC > 0.05
- IC_IR > 0.55
- IC 衰减单调递减 (T+1 > T+3 > T+5)

使用示例:
    python run_v151.py --year 2024
    python run_v151.py --all
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger
import pandas as pd
import numpy as np

from engine.backtest_referee import BacktestReferee, get_backtest_referee
from alpha_research_v151 import AlphaResearchV151, get_alpha_research as get_alpha_research_v151

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# V151 核心参数
LCA_ALPHA = 0.8  # EMA 平滑系数 (V150: 0.4 → V151: 0.8)
ROLLING_WINDOW = 20  # Rolling PAC 窗口


class V151Runner:
    """
    V151 统一回测运行器 - Latency-Corrected-Alpha (LCA).
    """
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        output_dir: str = "reports",
    ) -> None:
        self.parquet_path = parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        db_url = os.getenv("DATABASE_URL")
        
        # 初始化 V151 Alpha 模块
        self.alpha_module = get_alpha_research_v151(
            ic_threshold=0.0001,
            n_factors=8,
            n_bins=10,
            enable_ensemble=True,
            enable_pac=True,
            enable_pin=True,
            enable_ema=True,
            enable_lead_signal=True,
            enable_volatility_weighting=True,
            enable_sector_neutral=True,
            auto_heal=True,
            db_url=db_url,
            lca_alpha=LCA_ALPHA,
            rolling_window=ROLLING_WINDOW,
        )
        
        # 初始化裁判
        self.referee = get_backtest_referee(self.alpha_module, output_dir=output_dir)
        self.referee.VERSION = "V151"
        
        logger.info("=" * 70)
        logger.info("V151Runner initialized - Latency-Corrected-Alpha (LCA)")
        logger.info("=" * 70)
        logger.info(f"  Alpha Module: {type(self.alpha_module).__name__}")
        logger.info(f"  Referee: {type(self.referee).__name__}")
        logger.info(f"  Initial Capital: {self.referee.INITIAL_CAPITAL:,.0f}")
        logger.info(f"  LCA-EMA α: {LCA_ALPHA} (New={LCA_ALPHA*100}%, Prev={(1-LCA_ALPHA)*100}%)")
        logger.info(f"  Rolling PAC Window: {ROLLING_WINDOW}")
        logger.info(f"  Lead-Signal: Enabled (d(Factor)/dt)")
        logger.info(f"  Volatility Weighting: Enabled (1/Std(IC))")
        logger.info(f"  DataHealer: Enabled (ffill/median)")
        logger.info("=" * 70)
    
    def load_data(self, year: int) -> pd.DataFrame:
        """加载指定年份的数据"""
        parquet_path = self.parquet_path or "data/parquet/stock_data_2024_2026.parquet"
        if Path(parquet_path).exists():
            logger.info(f"Loading V151 data from Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
            if 'trade_date' in df.columns:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df = df[df['trade_date'].dt.year == year]
                df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
            logger.info(f"Loaded {len(df)} rows for year {year}")
            return df
        
        logger.info(f"Attempting to load data for year {year} from database...")
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
        logger.info(f"V151 Audit - Year {year}")
        logger.info("=" * 70)
        
        df = self.load_data(year)
        if df.empty:
            logger.warning(f"No data loaded for year {year}")
            return {'year': year, 'error': 'No data loaded', 'passed': False}
        
        logger.info("[Preprocessing] Converting data types...")
        if 'trade_date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['trade_date']):
                df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['trade_date'] = df['trade_date'].dt.strftime('%Y-%m-%d')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info("[Referee] Running audit...")
        result = self.referee.run_audit(df)
        
        report_path = self.generate_v151_report(result, year)
        result['year'] = year
        result['custom_report_path'] = report_path
        
        return result
    
    def generate_v151_report(self, result: dict, year: int) -> str:
        """生成 V151 年度审计报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"v151_lca_audit_{year}_{timestamp}.md"
        
        t1_ic = result.get('t1_ic', {})
        ic_decay = result.get('ic_decay', {})
        backtest_result = result.get('backtest_result', {})
        passed = result.get('passed', False)
        
        # 获取 V151 特有统计
        factor_ics = self.alpha_module.get_factor_ics()
        selected_factors = self.alpha_module.get_selected_factors()
        ema_stats = self.alpha_module.get_ema_stats()
        pac_stats = self.alpha_module.get_pac_stats()
        lead_signal_stats = self.alpha_module.get_lead_signal_stats()
        volatility_weights = self.alpha_module.get_volatility_weights()
        
        # V150 对比
        v150_ic = 0.05
        v150_ir = 0.50
        
        # 构建因子 IC 表格
        factor_ic_info = ""
        if factor_ics:
            for factor_name, ic in sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True):
                selected = "✓" if factor_name in selected_factors else ""
                vol_weight = volatility_weights.get(factor_name, 1.0)
                factor_ic_info += f"| {factor_name} | {ic:.4f} | {vol_weight:.3f} | {selected} |\n"
        
        # 构建 PAC 统计
        pac_info = ""
        for factor, sign_history in list(pac_stats.items())[:5]:
            pac_info += f"| {factor} | {sign_history.get('current_sign', 0)} | {sign_history.get('rolling_ic', 0):.4f} |\n"
        
        # 构建 Lead-Signal 统计
        lead_info = ""
        for factor, stats in list(lead_signal_stats.items())[:5]:
            lead_info += f"| {factor} | {stats.get('delta_mean', 0):.4f} | {stats.get('delta_std', 0):.4f} |\n"
        
        t1_ic_mean = t1_ic.get('mean_ic', 0)
        t1_ic_ir = t1_ic.get('ic_ir', 0)
        ic_decay_pattern = ic_decay.get('decay_pattern', 'N/A')
        ic_decay_monotonic = ic_decay.get('is_monotonic', False)
        
        ir_improvement = (t1_ic_ir - v150_ir) / (abs(v150_ir) + 1e-10)
        target_met = t1_ic_ir >= 0.55
        
        report_content = f"""# V151 LCA Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Year**: {year}
**Architecture**: Referee-Player (裁判 - 选手)
**Version**: V151 Latency-Corrected-Alpha (LCA)

---

## 1. Executive Summary (执行摘要)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC | {t1_ic_mean:.4f} | > 0.05 | {'✓ PASSED' if t1_ic_mean > 0.05 else '✗ FAILED'} |
| IC IR | {t1_ic_ir:.2f} | > 0.55 | {'✓ PASSED' if t1_ic_ir > 0.55 else '✗ FAILED'} |
| IC Decay | {'Monotonic' if ic_decay_monotonic else 'Non-monotonic'} | Monotonic | {'✓ PASSED' if ic_decay_monotonic else '✗ FAILED'} |
| Lead-Signal Count | {len(lead_signal_stats)} | >= 1 | {'✓' if len(lead_signal_stats) >= 1 else '✗'} |

**Overall Assessment**: **{'PASSED ✓' if passed else 'FAILED ✗'}**

---

## 2. V151 Core Features (V151 核心特性)

### 2.1 LCA-EMA (Latency-Corrected-Alpha EMA)

| Parameter | V150 | V151 | Change |
|-----------|------|------|--------|
| EMA α | 0.4 | {LCA_ALPHA} | +{(LCA_ALPHA-0.4)*100:.0f}% |
| Signal Mix | 40% new | {LCA_ALPHA*100:.0f}% new | Faster response |
| Purpose | Smooth | Reduce latency | IC decay fix |

### 2.2 Rolling PAC (防偷看未来)

| Factor | Current Sign | Rolling IC |
|--------|--------------|------------|
{pac_info if pac_info else "*No PAC data*"}

### 2.3 Lead-Signal (一阶差分)

| Factor | Δ Mean | Δ Std | Purpose |
|--------|--------|-------|---------|
{lead_info if lead_info else "*No lead-signal data*"}

### 2.4 Volatility-Standardized IC

| Factor | IC | Vol Weight (1/Std) |
|--------|-----|-------------------|
{factor_ic_info if factor_ic_info else "*No factor data*"}

### 2.5 DataHealer (数据自愈)

| Metric | Value |
|--------|-------|
| NaN Repaired | {self.alpha_module.healing_log.get('nan_repaired', 0)} |
| Inf Repaired | {self.alpha_module.healing_log.get('inf_repaired', 0)} |
| Ffill Applied | {self.alpha_module.healing_log.get('ffill_applied', 0)} |
| Median Imputed | {self.alpha_module.healing_log.get('median_imputed', 0)} |

---

## 3. V151 vs V150 Comparison (IC 提升对比)

| Metric | V150 | V151 | Improvement |
|--------|------|------|-------------|
| T+1 IC | {v150_ic:.4f} | {t1_ic_mean:.4f} | {t1_ic_mean - v150_ic:+.4f} |
| IC IR | {v150_ir:.2f} | {t1_ic_ir:.2f} | {ir_improvement:+.2%} |
| EMA α | 0.4 | {LCA_ALPHA} | +{(LCA_ALPHA-0.4)*100:.0f}% |
| Lead-Signal | No | Yes | +New |

**IC vs V150**: {t1_ic_mean - v150_ic:+.4f} ({'✓' if t1_ic_mean > v150_ic else '✗'})
**IR vs V150**: {ir_improvement:+.2%} ({'✓' if t1_ic_ir > v150_ir else '✗'})

---

## 4. IC Decay Analysis (IC 衰减分析)

| Horizon | IC | Pattern |
|---------|-----|---------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | Baseline |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ Non-monotonic'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ Non-monotonic'} |

**Decay Pattern**: {ic_decay_pattern}
**Monotonic Check**: {'✓ PASSED' if ic_decay_monotonic else '✗ FAILED - IC 衰减扭曲 (T+5 > T+1)'}

### 4.1 IC Decay Diagnosis

{'**[DIAGNOSIS]** IC 衰减正常：T+1 > T+3 > T+5，信号实时性良好。' if ic_decay_monotonic else '**[DIAGNOSIS]** IC 衰减扭曲：T+5 IC 高于 T+1，说明信号反应太慢。V151 通过以下修复：'}
{'- LCA-EMA α 从 0.4 提升至 0.8，新信号占比从 40% 提升至 80%' if not ic_decay_monotonic else ''}
{'- Lead-Signal 一阶差分提升对价格拐点的反应速度' if not ic_decay_monotonic else ''}
{'- Volatility-Standardized IC 降低不稳定因子权重' if not ic_decay_monotonic else ''}

---

## 5. Backtest Performance (回测表现)

| Metric | Value |
|--------|-------|
| Initial Capital | {self.referee.INITIAL_CAPITAL:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {backtest_result.get('total_return', 0):.2%} |
| Annual Return | {backtest_result.get('annual_return', 0):.2%} |
| Sharpe Ratio | {backtest_result.get('sharpe_ratio', 0):.2f} |
| Max Drawdown | {backtest_result.get('max_drawdown', 0):.2%} |

---

## 6. Conclusion (结论)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > 0.05 | {t1_ic_mean:.4f} | {'✓' if t1_ic_mean > 0.05 else '✗'} |
| IC IR | > 0.55 | {t1_ic_ir:.2f} | {'✓' if t1_ic_ir > 0.55 else '✗'} |
| IC Decay | Monotonic | {ic_decay_pattern} | {'✓' if ic_decay_monotonic else '✗'} |
| Lead-Signal | Enabled | {len(lead_signal_stats)} factors | {'✓' if len(lead_signal_stats) >= 1 else '✗'} |

**{'PASSED ✓' if passed else 'FAILED ✗'}**

---

*Report generated by V151 Unified Main Entry (Latency-Corrected-Alpha)*
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
            'ema_stats': ema_stats,
            'pac_stats': pac_stats,
            'lead_signal_stats': lead_signal_stats,
            'volatility_weights': volatility_weights,
            'healing_log': self.alpha_module.healing_log,
            'v150_comparison': {
                'v150_ic': v150_ic,
                'v150_ir': v150_ir,
                'ic_improvement': t1_ic_mean - v150_ic,
                'ir_improvement': ir_improvement,
            },
            'config': {
                'year': year,
                'initial_capital': self.referee.INITIAL_CAPITAL,
                'lca_alpha': LCA_ALPHA,
                'rolling_window': ROLLING_WINDOW,
            },
        }
        
        json_path = self.output_dir / f"v151_lca_audit_{year}_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        return str(report_path)
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        """运行多年份的审计"""
        logger.info("=" * 70)
        logger.info(f"V151 Multi-Year Audit - Years: {years}")
        logger.info("=" * 70)
        
        results = []
        passed_count = 0
        all_ic_values = []
        all_ic_decay_patterns = []
        
        for year in years:
            result = self.run_audit(year)
            results.append(result)
            if result.get('passed', False):
                passed_count += 1
            if 't1_ic' in result:
                all_ic_values.append(result['t1_ic'].get('mean_ic', 0))
            if 'ic_decay' in result:
                all_ic_decay_patterns.append(result['ic_decay'].get('decay_pattern', 'N/A'))
        
        cross_year_ic_mean = float(np.mean(all_ic_values)) if all_ic_values else 0
        cross_year_ic_std = float(np.std(all_ic_values, ddof=1)) if len(all_ic_values) > 1 else 0
        cross_year_ic_ir = cross_year_ic_mean / cross_year_ic_std if cross_year_ic_std > 1e-10 else 0
        
        # IC 衰减分析
        ic_decay_analysis = {
            'pattern': all_ic_decay_patterns[0] if all_ic_decay_patterns else 'N/A',
            'is_monotonic': 'Monotonic' in (all_ic_decay_patterns[0] if all_ic_decay_patterns else ''),
        }
        
        v150_ir = 0.50
        ir_improvement = (cross_year_ic_ir - v150_ir) / (abs(v150_ir) + 1e-10)
        target_met = cross_year_ic_ir >= 0.55
        
        # 生成反思报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reflection_path = self.output_dir / f"v151_lca_reflection_{timestamp}.json"
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'version': 'V151',
            'strategy': 'LCA (Latency-Corrected-Alpha)',
            'core_improvements': {
                'lca_ema': f'α={LCA_ALPHA}, New={LCA_ALPHA*100}%, Prev={(1-LCA_ALPHA)*100}%',
                'rolling_pac': f'Rolling_IC_Sign(window={ROLLING_WINDOW}), strict no-look-ahead',
                'lead_signal': 'd(Factor)/dt first-order differentiation',
                'volatility_weighting': '1/Std(IC) weighting for stability',
                'data_healer': 'ffill/median auto-repair for NaN/Inf',
            },
            'selected_factors': self.alpha_module.get_selected_factors(),
            'factor_ics': self.alpha_module.get_factor_ics(),
            'ema_stats': self.alpha_module.get_ema_stats(),
            'pac_stats': self.alpha_module.get_pac_stats(),
            'lead_signal_stats': self.alpha_module.get_lead_signal_stats(),
            'summary': {
                'years': years,
                'passed_count': passed_count,
                'total_count': len(years),
                'cross_year_ic_mean': cross_year_ic_mean,
                'cross_year_ic_std': cross_year_ic_std,
                'cross_year_ic_ir': cross_year_ic_ir,
            },
            'ic_decay_analysis': ic_decay_analysis,
            'v150_vs_v151_comparison': {
                'v150_ir': v150_ir,
                'v151_ir': cross_year_ic_ir,
                'ir_improvement': ir_improvement,
                'target_ir': 0.55,
                'target_met': target_met,
            },
            'acceptance_criteria': {
                't1_ic_target': 0.05,
                't1_ic_actual': cross_year_ic_mean,
                'ic_ir_target': 0.55,
                'ic_ir_actual': cross_year_ic_ir,
                'ic_decay_target': 'T+1 > T+3 > T+5',
                'ic_decay_actual': ic_decay_analysis['pattern'],
                'ic_decay_monotonic': ic_decay_analysis['is_monotonic'],
            },
            'improvement_hypotheses': [] if (target_met and ic_decay_analysis['is_monotonic']) else [
                '假设 1：进一步提高 LCA α从 0.8 至 0.9，减少平滑延迟。',
                '假设 2：增加 Lead-Signal 权重，对 volume_price_contradiction 差分信号×1.5。',
                '假设 3：降低 Rolling PAC 窗口从 20 至 15，提升极性响应速度。',
            ],
        }
        
        with open(reflection_path, 'w', encoding='utf-8') as f:
            json.dump(reflection, f, indent=2, default=str)
        
        logger.info(f"Reflection saved to: {reflection_path}")
        
        summary = {
            'years': years,
            'results': results,
            'passed_count': passed_count,
            'total_count': len(years),
            'cross_year_ic_mean': cross_year_ic_mean,
            'cross_year_ic_std': cross_year_ic_std,
            'cross_year_ic_ir': cross_year_ic_ir,
            'ic_decay_analysis': ic_decay_analysis,
            'reflection_path': str(reflection_path),
        }
        
        return summary


def main():
    """主入口函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="V151 LCA Backtest Runner")
    parser.add_argument('--year', type=int, default=None, help='Year to run (e.g., 2024)')
    parser.add_argument('--all', action='store_true', help='Run all years (2021, 2024)')
    parser.add_argument('--parquet', type=str, default=None, help='Parquet file path')
    parser.add_argument('--output', type=str, default='reports', help='Output directory')
    
    args = parser.parse_args()
    
    runner = V151Runner(parquet_path=args.parquet, output_dir=args.output)
    
    if args.all:
        years = [2021, 2024]
        summary = runner.run_multi_year_audit(years)
        
        logger.info("=" * 70)
        logger.info("V151 Multi-Year Audit Complete!")
        logger.info(f"  Years: {years}")
        logger.info(f"  Passed: {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"  Cross-Year IC: {summary['cross_year_ic_mean']:.4f} ± {summary['cross_year_ic_std']:.4f}")
        logger.info(f"  Cross-Year IC IR: {summary['cross_year_ic_ir']:.2f} (V150: 0.50)")
        logger.info(f"  IC Decay Pattern: {summary['ic_decay_analysis']['pattern']}")
        logger.info(f"  IC Decay Monotonic: {'YES ✓' if summary['ic_decay_analysis']['is_monotonic'] else 'NO ✗'}")
        logger.info("=" * 70)
        
    elif args.year:
        result = runner.run_audit(args.year)
        
        logger.info("=" * 70)
        logger.info("V151 Audit Complete!")
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