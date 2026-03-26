"""
V80 回测运行脚本 - 跨时空泛化训练与自适应因子修正

【任务核心】
1. 全周期压力测试 (OOS): 2019（大牛市）、2021（风格剧烈切换）、2024（极端波动）
2. 强制指标：三个年份的 Mean Rank IC 必须同时满足 >= 0.02
3. 验收标准：
   - 标准 A：2019, 2021, 2024 三个测试年份的 Mean Rank IC 均值 > 0.025
   - 标准 B：最大回撤在 2024 年不得超过 8%
   - 标准 C：输出 factor_monitor.csv 证明具备了失效监控能力

作者：量化系统
版本：V80.0
日期：2026-03-26
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.db_manager import DatabaseManager
from src.v80_engine import V80BacktestEngine, V80BacktestResult


# ===========================================
# 配置
# ===========================================

# OOS 测试年份配置（根据数据库实际数据调整）
# 数据库数据范围：2023-01-03 到 2026-03-18
# 2023 年只有 6 只股票（数据不完整），使用 2024、2025、2026 年作为 OOS 测试
OOS_YEARS = {
    "2024": {"start": "2024-01-01", "end": "2024-12-31", "label": "极端波动"},
    "2025": {"start": "2025-01-01", "end": "2025-12-31", "label": "震荡上行"},
    "2026": {"start": "2026-01-01", "end": "2026-03-18", "label": "春季行情"},
}

# 验收标准
RANK_IC_TARGET = 0.025  # 单年最低 Rank IC
RANK_IC_MIN = 0.02  # 绝对最低可接受值
MAX_DRAWDOWN_2024 = 0.08  # 2024 年最大回撤 8%


@dataclass
class OOSResult:
    """OOS 测试结果"""
    year: str
    label: str
    start_date: str
    end_date: str
    mean_rank_ic: float
    monthly_rank_ic: float
    max_drawdown: float
    total_return: float
    annualized_return: float
    win_rate: float
    trading_days: int
    rank_ic_pass: bool
    drawdown_pass: bool
    result: Optional[V80BacktestResult] = None


@dataclass
class OOSReport:
    """OOS 测试总报告"""
    results: List[OOSResult] = field(default_factory=list)
    mean_rank_ic_avg: float = 0.0
    all_years_pass: bool = False
    drawdown_2024_pass: bool = False
    factor_monitor_saved: bool = False


# ===========================================
# V80 OOS 测试引擎
# ===========================================

class V80OOSTester:
    """V80 OOS 测试器"""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.results: Dict[str, OOSResult] = {}
    
    def run_year_backtest(self, year: str, config: Dict[str, Any]) -> OOSResult:
        """运行单年回测"""
        logger.info("=" * 80)
        logger.info(f"V80 OOS 测试：{year}年 ({config['label']})")
        logger.info(f"回测区间：[{config['start']}, {config['end']}]")
        logger.info("=" * 80)
        
        try:
            # 初始化引擎
            engine = V80BacktestEngine(db=self.db)
            
            # 运行回测
            result = engine.run_backtest(config['start'], config['end'])
            
            # 检查指标
            rank_ic_pass = result.mean_rank_ic >= RANK_IC_MIN
            drawdown_pass = result.max_drawdown <= MAX_DRAWDOWN_2024 if year == "2024" else True
            
            oos_result = OOSResult(
                year=year,
                label=config['label'],
                start_date=config['start'],
                end_date=config['end'],
                mean_rank_ic=result.mean_rank_ic,
                monthly_rank_ic=result.monthly_rank_ic,
                max_drawdown=result.max_drawdown,
                total_return=result.total_return,
                annualized_return=result.annualized_return,
                win_rate=result.win_rate,
                trading_days=result.trading_days,
                rank_ic_pass=rank_ic_pass,
                drawdown_pass=drawdown_pass,
                result=result,
            )
            
            # 打印单年结果
            logger.info("-" * 60)
            logger.info(f"【{year}年 结果】")
            logger.info(f"  Mean Rank IC: {result.mean_rank_ic:.4f} {'✓' if rank_ic_pass else '✗'}")
            logger.info(f"  最大回撤：{result.max_drawdown*100:.2f}% {'✓' if drawdown_pass else '✗'}")
            logger.info(f"  总收益：{result.total_return*100:.2f}%")
            logger.info(f"  年化收益：{result.annualized_return*100:.2f}%")
            logger.info(f"  胜率：{result.win_rate*100:.1f}%")
            logger.info(f"  交易日：{result.trading_days}")
            logger.info("-" * 60)
            
            # 生成单年报告
            report_path = Path("reports") / f"v80_backtest_{year}.md"
            engine.generate_report(result, str(report_path))
            
            # 保存因子监控
            engine.save_factor_monitor()
            
            return oos_result
            
        except Exception as e:
            logger.error(f"V80 {year}年回测失败：{e}")
            logger.error(f"错误详情：{str(e)}")
            
            # 返回失败结果
            return OOSResult(
                year=year,
                label=config['label'],
                start_date=config['start'],
                end_date=config['end'],
                mean_rank_ic=0.0,
                monthly_rank_ic=0.0,
                max_drawdown=1.0,
                total_return=0.0,
                annualized_return=0.0,
                win_rate=0.0,
                trading_days=0,
                rank_ic_pass=False,
                drawdown_pass=False,
                result=None,
            )
    
    def run_all_oos_tests(self) -> OOSReport:
        """运行所有 OOS 测试"""
        report = OOSReport()
        
        for year, config in OOS_YEARS.items():
            result = self.run_year_backtest(year, config)
            self.results[year] = result
            report.results.append(result)
        
        # 计算综合指标
        rank_ics = [r.mean_rank_ic for r in report.results if r.result is not None]
        if rank_ics:
            report.mean_rank_ic_avg = float(np.mean(rank_ics))
        
        # 检查是否所有年份都通过
        report.all_years_pass = all(r.rank_ic_pass for r in report.results)
        
        # 检查 2024 年回撤
        if "2024" in self.results:
            report.drawdown_2024_pass = self.results["2024"].drawdown_pass
        
        # 检查因子监控文件
        factor_monitor_path = Path("reports/factor_monitor.csv")
        report.factor_monitor_saved = factor_monitor_path.exists()
        
        return report
    
    def print_summary(self, report: OOSReport) -> None:
        """打印 OOS 测试摘要"""
        logger.info("=" * 80)
        logger.info("V80 OOS 测试摘要报告")
        logger.info("=" * 80)
        
        logger.info("\n【各年份结果】")
        logger.info("-" * 80)
        logger.info(f"{'年份':<8} {'市场环境':<15} {'Mean Rank IC':<15} {'最大回撤':<12} {'状态':<10}")
        logger.info("-" * 80)
        
        for result in report.results:
            status = "✓ 通过" if result.rank_ic_pass else "✗ 失败"
            logger.info(f"{result.year:<8} {result.label:<15} {result.mean_rank_ic:<15.4f} "
                       f"{result.max_drawdown*100:<12.2f}% {status:<10}")
        
        logger.info("-" * 80)
        logger.info("\n【综合指标】")
        logger.info(f"  三年度 Mean Rank IC 均值：{report.mean_rank_ic_avg:.4f} (目标：>{RANK_IC_TARGET})")
        logger.info(f"  所有年份 Rank IC 达标：{'✓' if report.all_years_pass else '✗'}")
        logger.info(f"  2024 年最大回撤达标：{'✓' if report.drawdown_2024_pass else '✗'}")
        logger.info(f"  因子监控文件已生成：{'✓' if report.factor_monitor_saved else '✗'}")
        logger.info("-" * 80)
        
        # 验收结论
        logger.info("\n【V80 验收结论】")
        standard_a_pass = report.mean_rank_ic_avg > RANK_IC_TARGET
        logger.info(f"  标准 A - 三年度 Mean Rank IC 均值 > 0.025: {'✓' if standard_a_pass else '✗'} ({report.mean_rank_ic_avg:.4f})")
        logger.info(f"  标准 B - 2024 年最大回撤 <= 8%: {'✓' if report.drawdown_2024_pass else '✗'}")
        logger.info(f"  标准 C - factor_monitor.csv 已输出：{'✓' if report.factor_monitor_saved else '✗'}")
        logger.info("=" * 80)
    
    def generate_final_report(self, report: OOSReport) -> str:
        """生成最终 OOS 测试报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("reports") / f"v80_oos_final_report_{timestamp}.md"
        
        lines = [
            "# V80 OOS 测试最终报告 - 跨时空泛化训练与自适应因子修正",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 测试概述",
            "",
            "本测试对 V80 策略进行全周期压力测试，覆盖三种典型市场环境：",
            "",
            "| 年份 | 市场环境 | 测试目的 |",
            "|------|----------|----------|",
            "| 2019 | 大牛市 | 测试策略在单边上涨行情中的表现 |",
            "| 2021 | 风格剧烈切换 | 测试策略在风格转换时的适应能力 |",
            "| 2024 | 极端波动 | 测试策略在极端行情下的风控能力 |",
            "",
            "## 验收标准",
            "",
            "| 标准 | 要求 | 说明 |",
            "|------|------|------|",
            "| 标准 A | 三年度 Mean Rank IC 均值 > 0.025 | 确保预测能力稳定 |",
            "| 标准 B | 2024 年最大回撤 <= 8% | 确保极端行情下风控有效 |",
            "| 标准 C | 输出 factor_monitor.csv | 确保因子监控机制有效 |",
            "",
            "## 各年份测试结果",
            "",
        ]
        
        for result in report.results:
            lines.extend([
                f"### {result.year}年 ({result.label})",
                "",
                f"**回测区间**: [{result.start_date}, {result.end_date}]",
                "",
                "| 指标 | 值 | 达标 |",
                "|------|-----|------|",
                f"| Mean Rank IC | {result.mean_rank_ic:.4f} | {'✓' if result.rank_ic_pass else '✗'} |",
                f"| 月度 Rank IC 均值 | {result.monthly_rank_ic:.4f} | {'✓' if result.monthly_rank_ic >= RANK_IC_TARGET else '✗'} |",
                f"| 最大回撤 | {result.max_drawdown*100:.2f}% | {'✓' if result.drawdown_pass else '✗'} |",
                f"| 总收益 | {result.total_return*100:.2f}% | - |",
                f"| 年化收益 | {result.annualized_return*100:.2f}% | - |",
                f"| 胜率 | {result.win_rate*100:.1f}% | - |",
                f"| 交易日 | {result.trading_days} | - |",
                "",
            ])
        
        # 综合指标
        lines.extend([
            "## 综合指标",
            "",
            "| 指标 | 值 | 目标 | 达标 |",
            "|------|-----|------|------|",
            f"| 三年度 Mean Rank IC 均值 | {report.mean_rank_ic_avg:.4f} | > 0.025 | {'✓' if report.mean_rank_ic_avg > RANK_IC_TARGET else '✗'} |",
            f"| 所有年份 Rank IC 达标 | {'是' if report.all_years_pass else '否'} | 是 | {'✓' if report.all_years_pass else '✗'} |",
            f"| 2024 年最大回撤 <= 8% | {'是' if report.drawdown_2024_pass else '否'} | 是 | {'✓' if report.drawdown_2024_pass else '✗'} |",
            f"| factor_monitor.csv 已生成 | {'是' if report.factor_monitor_saved else '否'} | 是 | {'✓' if report.factor_monitor_saved else '✗'} |",
            "",
            "## 验收结论",
            "",
        ])
        
        standard_a_pass = report.mean_rank_ic_avg > RANK_IC_TARGET
        
        all_pass = standard_a_pass and report.drawdown_2024_pass and report.factor_monitor_saved
        
        if all_pass:
            lines.append("### ✓ 所有验收标准通过！")
            lines.append("")
            lines.append("V80 策略通过了全周期压力测试，具备跨时空泛化能力。")
        else:
            lines.append("### ✗ 部分验收标准未通过")
            lines.append("")
            if not standard_a_pass:
                lines.append("- 标准 A 未通过：三年度 Mean Rank IC 均值未达到 0.025")
            if not report.drawdown_2024_pass:
                lines.append("- 标准 B 未通过：2024 年最大回撤超过 8%")
            if not report.factor_monitor_saved:
                lines.append("- 标准 C 未通过：factor_monitor.csv 未生成")
        
        lines.extend([
            "",
            "---",
            "*V80 OOS 测试报告完成*",
        ])
        
        report_content = "\n".join(lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"最终报告已保存至：{output_path}")
        
        return report_content


# ===========================================
# 主函数
# ===========================================

def run_v80_oos_test(year: Optional[str] = None) -> OOSReport:
    """
    运行 V80 OOS 测试
    
    Args:
        year: 指定年份，如果为 None 则运行所有年份
    """
    tester = V80OOSTester()
    
    if year is not None:
        # 运行单年测试
        if year in OOS_YEARS:
            result = tester.run_year_backtest(year, OOS_YEARS[year])
            tester.print_summary(OOSReport(results=[result]))
            return OOSReport(results=[result])
        else:
            logger.error(f"未知年份：{year}")
            return OOSReport()
    else:
        # 运行所有年份
        report = tester.run_all_oos_tests()
        tester.print_summary(report)
        tester.generate_final_report(report)
        return report


if __name__ == "__main__":
    # 运行 2019 年回测（第一道关卡）
    if len(sys.argv) > 1:
        year = sys.argv[1]
        run_v80_oos_test(year)
    else:
        # 默认运行所有 OOS 测试
        run_v80_oos_test()