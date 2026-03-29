"""
V85 回测运行脚本 - 单因子消融实验与交互项核心化

【使用说明】
1. 确保数据库连接正常
2. 运行：python -m src.run_v85_backtest
3. 结果保存至：reports/v85_backtest_result.json
4. 日志输出至：logs/v85_backtest.log

【硬性指标】
- 指标 A：Vol_Price_Interaction 的单项 Rank IC 必须 > 0.04
- 指标 B：三年度融合后的 Mean Rank IC 必须转正且均值 > 0.035
- 指标 C：必须在总结中详细对比 V84 负值与 V85 正值的逻辑差异

作者：量化系统
版本：V85.0
日期：2026-03-29
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.v85_engine import (
    V85Engine,
    V85EngineConfig,
    run_v85_backtest,
    print_v85_report,
    V85_RANK_IC_OOS_YEARS,
)


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    
    # 控制台输出
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 文件日志
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/v85_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        sink=log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="DEBUG"
    )
    
    logger.info("=" * 60)
    logger.info("V85 回测脚本启动")
    logger.info(f"日志文件：{log_file}")
    logger.info("=" * 60)
    
    # 创建配置
    config = V85EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=V85_RANK_IC_OOS_YEARS,  # ["2019", "2021", "2024"]
    )
    
    logger.info("")
    logger.info("【回测配置】")
    logger.info(f"  回测区间：{config.start_date} ~ {config.end_date}")
    logger.info(f"  OOS 测试年份：{config.oos_years}")
    logger.info(f"  初始资金：{config.initial_capital:,.2f}")
    logger.info(f"  手续费率：{config.commission_rate:.2%}")
    logger.info(f"  最大持仓数：{config.max_positions}")
    logger.info("")
    
    # 运行回测
    result = run_v85_backtest(config)
    
    # 打印报告
    print_v85_report(result)
    
    # 保存 Markdown 报告
    save_markdown_report(result)
    
    # 保存 JSON 结果
    output_json_path = "reports/v85_backtest_result.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    serializable_result = {
        'oos_stats': result.get('oos_stats', {}),
        'hard_metrics': result.get('hard_metrics', {}),
        'ic_statistics': result.get('ic_statistics', {}),
        'factor_monthly_ics': result.get('factor_monthly_ics', {}),
        'monthly_rank_ic_stats': result.get('monthly_rank_ic_stats', {}),
        'year_trading_days': result.get('year_trading_days', {}),
        'ablation_report': result.get('ablation_report', ''),
        'comparison_report': result.get('comparison_report', ''),
        'status': result.get('status', {}),
        'data_integrity': result.get('data_integrity', {}),
        'ic_analysis': result.get('ic_analysis', {}),
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V85: 结果已保存至 {output_json_path}")
    logger.info(f"V85: 日志文件：{log_file}")
    
    return result


def save_markdown_report(result: dict):
    """保存 Markdown 格式报告"""
    output_path = "reports/v85_backtest_result.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    oos_stats = result.get('oos_stats', {})
    hard_metrics = result.get('hard_metrics', {})
    ic_statistics = result.get('ic_statistics', {})
    ablation_report = result.get('ablation_report', '')
    comparison_report = result.get('comparison_report', '')
    
    lines = [
        "# V85 回测报告 - 单因子消融实验与交互项核心化",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 执行摘要",
        "",
        "### V85 核心理念",
        "1. **Auto_Direction_Check（方向自修复）** - 基于过去 20 天 IC 符号自动调整权重方向",
        "2. **消融实验（Ablation Study）** - 独立输出三个因子的 Rank IC",
        "3. **Vol_Price_Interaction 核心化** - 70% 权重 + Sigmoid 非线性挤压",
        "4. **NaN 检测与修复** - 明确日志输出并自动修复",
        "",
        "## 硬性指标验证",
        "",
        f"- **指标 A** (Vol_Price_Interaction IC > 0.04): {'✓ 通过' if hard_metrics.get('metric_a_pass') else '✗ 未通过'}",
        f"- **指标 B** (三年度 Mean Rank IC > 0.035): {'✓ 通过' if hard_metrics.get('metric_b_pass') else '✗ 未通过'}",
        f"- **指标 C** (V84 vs V85 对比报告): {'✓ 通过' if hard_metrics.get('metric_c_pass') else '✗ 未通过'}",
        "",
        "## OOS 年度统计",
        "",
        "| 年份 | Mean Rank IC | Std Rank IC | 样本数 | 正占比 | 达标 |",
        "|------|-------------|-------------|--------|--------|------|",
    ]
    
    for year in V85_RANK_IC_OOS_YEARS:
        if year in oos_stats:
            stat = oos_stats[year]
            in_range = "✓" if stat.get('ic_in_target_range') else "✗"
            lines.append(
                f"| {year} | {stat['mean_rank_ic']:.4f} | {stat['std_rank_ic']:.4f} | "
                f"{stat['ic_count']} | {stat['positive_ratio']:.2%} | {in_range} |"
            )
    
    # 计算三年度平均
    valid_years = [y for y in V85_RANK_IC_OOS_YEARS if y in oos_stats and oos_stats[y].get('ic_count', 0) > 0]
    if valid_years:
        avg_ic = sum(oos_stats[y]['mean_rank_ic'] for y in valid_years) / len(valid_years)
        lines.append("")
        lines.append(f"**三年度平均 Mean Rank IC**: {avg_ic:.4f}")
    
    lines.extend([
        "",
        "## IC 统计",
        "",
        f"- Mean IC: {ic_statistics.get('mean_ic', 0.0):.4f}",
        f"- Mean Rank IC: {ic_statistics.get('mean_rank_ic', 0.0):.4f}",
        f"- IC IR: {ic_statistics.get('ic_ir', 0.0):.2f}",
        f"- Rank IC IR: {ic_statistics.get('rank_ic_ir', 0.0):.2f}",
        f"- 正 IC 占比：{ic_statistics.get('positive_ratio', 0.0):.2%}",
        f"- 统计天数：{ic_statistics.get('num_valid_days', 0)}",
        "",
        "## 消融实验报告",
        "",
        ablation_report if ablation_report else "无数据",
        "",
        "## V84 vs V85 对比报告",
        "",
        comparison_report if comparison_report else "无数据",
        "",
        "## 数据完整性",
        "",
        "| 年份 | 状态 | 交易天数 | 最小要求 |",
        "|------|------|----------|----------|",
    ])
    
    data_integrity = result.get('data_integrity', {})
    for year, info in data_integrity.items():
        status = "✓ 通过" if info.get('passed') else "✗ 失败"
        lines.append(f"| {year} | {status} | {info.get('trading_days', 0):,} | {info.get('min_required', 0):,} |")
    
    lines.extend([
        "",
        "## 附录",
        "",
        "### 配置参数",
        "- 初始资金：100,000",
        "- 手续费率：0.2%",
        "- 最大持仓数：10",
        "- 回测年份：2019, 2021, 2024",
        "",
        "---",
        "*报告由 V85 引擎自动生成*",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"V85: Markdown 报告已保存至 {output_path}")


if __name__ == "__main__":
    main()