#!/usr/bin/env python3
"""
V101 Unified Backtest Runner - Alpha Prediction System Audit.

【使用说明】
运行 2019/2021/2024 年回测，输出以 IC 为核心的详细审计报告。

使用示例:
    python run_v101.py --year 2019
    python run_v101.py --year 2021
    python run_v101.py --year 2024
    python run_v101.py --all  # 运行所有年份

【验收指标】
| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.05 | 核心指标：低于此值直接视为失败 |
| IC IR (稳定性) | > 0.6 | 跨年度预测能力的稳定性 |
| 因子贡献度 (Top Factor IC) | > 0.04 | 必须有至少一个核心因子具备独立战斗力 |
| 回测净收益 | 仅作输出参考 | 不作为优化目标 |
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger
import polars as pl

from data_loader import DataLoader, get_loader
from alpha_research import AlphaResearch, get_alpha_research
from backtest_accounting import BacktestAccounting, get_backtest_accounting

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class V101Runner:
    """
    V101 统一回测运行器。
    
    【运行流程】
    1. 加载数据（从 Parquet 或数据库）
    2. 运行 Alpha 分析（计算因子、IC 评估）
    3. 运行回测（计算收益、成本）
    4. 生成审计报告
    """
    
    def __init__(
        self,
        parquet_path: Optional[str] = None,
        initial_capital: float = 1_000_000.0,
        top_n: int = 50,
        output_dir: str = "reports",
    ) -> None:
        """
        初始化运行器。
        
        Args:
            parquet_path: Parquet 数据文件路径（可选）
            initial_capital: 初始资金
            top_n: 持有股票数量
            output_dir: 报告输出目录
        """
        self.parquet_path = parquet_path
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模块
        self.alpha_research = get_alpha_research()
        self.backtest_accounting = get_backtest_accounting(
            initial_capital=initial_capital,
            top_n=top_n,
            output_dir=output_dir,
        )
        
        logger.info("V101Runner initialized")
    
    def load_data(self, year: int) -> pl.DataFrame:
        """
        加载指定年份的数据。
        
        Args:
            year: 年份
            
        Returns:
            数据 DataFrame
        """
        # 优先从 Parquet 加载
        if self.parquet_path and Path(self.parquet_path).exists():
            logger.info(f"Loading data from Parquet: {self.parquet_path}")
            df = pl.read_parquet(self.parquet_path)
            
            # 按年份过滤
            if "trade_date" in df.columns:
                df = df.filter(
                    pl.col("trade_date").dt.year() == year
                )
            
            logger.info(f"Loaded {len(df)} rows for year {year}")
            return df
        
        # 否则尝试从数据库加载
        logger.info(f"Attempting to load data for year {year} from database...")
        
        # 这里需要从数据库加载数据
        # 由于数据库配置可能不同，这里提供一个示例实现
        try:
            from sqlalchemy import create_engine, text
            import os
            
            db_url = os.getenv("DATABASE_URL")
            if not db_url:
                raise ValueError("DATABASE_URL not configured")
            
            engine = create_engine(db_url)
            
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            
            query = text("""
                SELECT symbol, trade_date, open, high, low, close, pre_close,
                       change, pct_chg, volume, amount, turnover_rate, total_mv
                FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                ORDER BY symbol, trade_date
            """)
            
            df = pl.read_database_uri(query, uri=db_url, params={
                "start_date": start_date,
                "end_date": end_date,
            })
            
            logger.info(f"Loaded {len(df)} rows from database for year {year}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            # 返回空 DataFrame
            return pl.DataFrame()
    
    def run_audit(self, year: int) -> dict:
        """
        运行单一年份的审计。
        
        Args:
            year: 年份
            
        Returns:
            审计结果
        """
        logger.info("=" * 70)
        logger.info(f"V101 Audit - Year {year}")
        logger.info("=" * 70)
        
        # 1. 加载数据
        df = self.load_data(year)
        
        if df.is_empty():
            logger.warning(f"No data loaded for year {year}")
            return {
                "year": year,
                "error": "No data loaded",
                "passed": False,
            }
        
        # 2. 运行 Alpha 分析
        alpha_result = self.alpha_research.run_alpha_analysis(df)
        
        # 3. 运行回测
        backtest_result = self.backtest_accounting.run_full_audit(
            df=alpha_result["processed_df"],
            alpha_result=alpha_result,
            report_name=f"v101_audit_{year}",
        )
        
        # 4. 汇总结果
        result = {
            "year": year,
            "alpha_result": alpha_result,
            "backtest_result": backtest_result,
            "passed": alpha_result["passed"],
            "report_path": backtest_result["report_path"],
        }
        
        return result
    
    def run_multi_year_audit(self, years: list[int]) -> dict:
        """
        运行多年份的审计。
        
        Args:
            years: 年份列表
            
        Returns:
            汇总审计结果
        """
        logger.info("=" * 70)
        logger.info(f"V101 Multi-Year Audit - Years: {years}")
        logger.info("=" * 70)
        
        results = []
        passed_count = 0
        
        for year in years:
            result = self.run_audit(year)
            results.append(result)
            
            if result.get("passed", False):
                passed_count += 1
        
        # 汇总统计
        summary = {
            "years": years,
            "results": results,
            "passed_count": passed_count,
            "total_count": len(years),
            "pass_rate": passed_count / len(years) if years else 0,
        }
        
        # 生成汇总报告
        self._generate_summary_report(summary)
        
        return summary
    
    def _generate_summary_report(self, summary: dict) -> str:
        """生成汇总报告。"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"v101_summary_{timestamp}.md"
        
        results = summary.get("results", [])
        
        # 提取 IC 统计
        ic_stats = []
        for r in results:
            if "alpha_result" in r and "t1_ic" in r["alpha_result"]:
                t1_ic = r["alpha_result"]["t1_ic"]
                ic_stats.append({
                    "year": r["year"],
                    "mean_ic": t1_ic.get("mean_ic", 0),
                    "ic_ir": t1_ic.get("ic_ir", 0),
                    "passed": r.get("passed", False),
                })
        
        # 生成报告内容
        report_content = """# V101 Multi-Year Audit Summary

**Generated**: {timestamp}

---

## 1. Overall Summary (汇总)

| Metric | Value |
|--------|-------|
| Years Tested | {years} |
| Passed | {passed}/{total} |
| Pass Rate | {pass_rate:.1%} |

---

## 2. Year-by-Year IC Metrics (年度 IC 指标)

| Year | Mean IC | IC IR | Status |
|------|---------|-------|--------|
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            years=summary["years"],
            passed=summary["passed_count"],
            total=summary["total_count"],
            pass_rate=summary["pass_rate"],
        )
        
        for stat in ic_stats:
            status = "✓ PASSED" if stat["passed"] else "✗ FAILED"
            report_content += f"| {stat['year']} | {stat['mean_ic']:.4f} | {stat['ic_ir']:.2f} | {status} |\n"
        
        report_content += """
---

## 3. Acceptance Criteria (验收标准)

| Metric | Target | Description |
|--------|--------|-------------|
| T+1 Rank IC | > 0.05 | 核心指标：预测能力 |
| IC IR | > 0.6 | 稳定性指标 |
| Top Factor IC | > 0.04 | 核心因子独立战斗力 |

---

## 4. Conclusion (结论)

{conclusion}

---

*Report generated by V101 Runner*
""".format(
            conclusion="The V101 system has demonstrated consistent predictive power across multiple years." if summary["pass_rate"] >= 0.67 
                       else "The V101 system needs further optimization to achieve consistent predictive power."
        )
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Summary report saved to: {report_path}")
        
        return str(report_path)


def main():
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="V101 Unified Backtest Runner")
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to run audit (e.g., 2019, 2021, 2024)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run audit for all years (2019, 2021, 2024)"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default=None,
        help="Path to Parquet data file"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital (default: 1,000,000)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of stocks to hold (default: 50)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for reports"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("V101 Unified Backtest Runner")
    logger.info("=" * 70)
    
    # 初始化运行器
    runner = V101Runner(
        parquet_path=args.parquet,
        initial_capital=args.capital,
        top_n=args.top_n,
        output_dir=args.output,
    )
    
    # 确定运行年份
    if args.all:
        years = [2019, 2021, 2024]
        logger.info(f"Running audit for all years: {years}")
        summary = runner.run_multi_year_audit(years)
        
        logger.info("=" * 70)
        logger.info("V101 Multi-Year Audit Complete!")
        logger.info(f"  Years: {years}")
        logger.info(f"  Passed: {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"  Pass Rate: {summary['pass_rate']:.1%}")
        logger.info("=" * 70)
        
    elif args.year:
        logger.info(f"Running audit for year: {args.year}")
        result = runner.run_audit(args.year)
        
        logger.info("=" * 70)
        logger.info("V101 Audit Complete!")
        logger.info(f"  Year: {args.year}")
        logger.info(f"  Status: {'PASSED ✓' if result.get('passed', False) else 'FAILED ✗'}")
        logger.info(f"  Report: {result.get('report_path', 'N/A')}")
        logger.info("=" * 70)
        
    else:
        parser.print_help()
        logger.warning("Please specify --year or --all")
        sys.exit(1)


if __name__ == "__main__":
    main()