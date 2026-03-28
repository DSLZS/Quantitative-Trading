"""
V84 回测运行脚本 - 非线性特征挖掘与逻辑一致性审计

【使用说明】
1. 确保数据库已初始化并包含 2019、2021、2024 年的数据
2. 运行：python src/run_v84_backtest.py
3. 结果将保存至 reports/v84_backtest_result.json

【硬性指标】
- 指标 A：2019, 2021, 2024 三个年份的 Mean Rank IC 必须全部稳定在 [0.03, 0.08] 之间
- 指标 B：2024 年最大回撤必须控制在 6% 以内
- 指标 C：代码中必须包含对 Dynamic_Sign_Switch 的逻辑实现，并提供测试日志

作者：量化系统
版本：V84.0
日期：2026-03-28
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.v84_engine import (
    run_v84_backtest,
    print_v84_report,
    V84EngineConfig,
    V84_RANK_IC_OOS_YEARS,
    V84_RANK_IC_TARGET_MIN,
    V84_RANK_IC_TARGET_MAX,
    V84_MAX_DRAWDOWN_TARGET,
)


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件日志
    os.makedirs("logs", exist_ok=True)
    logger.add(
        f"logs/v84_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        level="DEBUG",
        rotation="100 MB"
    )
    
    logger.info("=" * 60)
    logger.info("V84 回测系统 - 非线性特征挖掘与逻辑一致性审计")
    logger.info("=" * 60)
    logger.info(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"OOS 测试年份：{V84_RANK_IC_OOS_YEARS}")
    logger.info(f"Rank IC 目标范围：[{V84_RANK_IC_TARGET_MIN}, {V84_RANK_IC_TARGET_MAX}]")
    logger.info(f"2024 年最大回撤目标：<= {V84_MAX_DRAWDOWN_TARGET:.2%}")
    logger.info("=" * 60)
    
    # 创建配置
    config = V84EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=V84_RANK_IC_OOS_YEARS,
    )
    
    # 运行回测
    result = run_v84_backtest(config)
    
    # 打印报告
    print_v84_report(result)
    
    # 保存结果
    output_path = "reports/v84_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 转换结果为可序列化格式
    serializable_result = {
        'oos_stats': result.get('oos_stats', {}),
        'hard_metrics': result.get('hard_metrics', {}),
        'ic_statistics': result.get('ic_statistics', {}),
        'factor_monthly_ics': result.get('factor_monthly_ics', {}),
        'monthly_rank_ic_stats': result.get('monthly_rank_ic_stats', {}),
        'year_trading_days': result.get('year_trading_days', {}),
        'sign_switch_report': result.get('sign_switch_report', ''),
        'logic_consistency_answer': result.get('logic_consistency_answer', ''),
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V84: 结果已保存至 {output_path}")
    
    # 生成 Markdown 报告
    generate_markdown_report(result, output_path.replace('.json', '.md'))
    
    # 验证硬性指标
    logger.info("")
    logger.info("=" * 60)
    logger.info("【硬性指标验证总结】")
    
    hard_metrics = result.get('hard_metrics', {})
    metric_a_pass = hard_metrics.get('metric_a_pass', False)
    metric_b_pass = hard_metrics.get('metric_b_pass', False)
    metric_c_pass = hard_metrics.get('metric_c_pass', False)
    
    logger.info(f"指标 A (IC 范围): {'✓ 通过' if metric_a_pass else '✗ 未通过'}")
    logger.info(f"指标 B (最大回撤): {'✓ 通过' if metric_b_pass else '✗ 未通过'}")
    logger.info(f"指标 C (Sign Switch): {'✓ 通过' if metric_c_pass else '✗ 未通过'}")
    
    all_pass = metric_a_pass and metric_b_pass and metric_c_pass
    logger.info("")
    logger.info(f"总体状态：{'✓ 所有硬性指标已达标' if all_pass else '✗ 部分指标未达标'}")
    logger.info("=" * 60)
    
    return result


def generate_markdown_report(result: dict, output_path: str):
    """生成 Markdown 格式报告"""
    oos_stats = result.get('oos_stats', {})
    hard_metrics = result.get('hard_metrics', {})
    ic_stats = result.get('ic_statistics', {})
    year_trading_days = result.get('year_trading_days', {})
    sign_switch_report = result.get('sign_switch_report', '')
    logic_answer = result.get('logic_consistency_answer', '')
    
    lines = [
        "# V84 回测报告 - 非线性特征挖掘与逻辑一致性审计",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. OOS 年度统计",
        "",
        "| 年份 | Mean Rank IC | IC 范围 | 样本数 | 正 IC 占比 |",
        "|------|-------------|--------|--------|----------|",
    ]
    
    for year in V84_RANK_IC_OOS_YEARS:
        if year in oos_stats:
            stat = oos_stats[year]
            in_range = "✓" if stat.get('ic_in_target_range') else "✗"
            lines.append(
                f"| {year} | {stat['mean_rank_ic']:.4f} | {in_range} | {stat['ic_count']} | {stat['positive_ratio']:.2%} |"
            )
    
    lines.extend([
        "",
        "## 2. 硬性指标验证",
        "",
        f"- **指标 A** (三年度 Mean Rank IC 在 [{V84_RANK_IC_TARGET_MIN}, {V84_RANK_IC_TARGET_MAX}] 内): "
        f"{'✓ 通过' if hard_metrics.get('metric_a_pass') else '✗ 未通过'}",
        f"- **指标 B** (2024 年最大回撤 <= {V84_MAX_DRAWDOWN_TARGET:.2%}): "
        f"{'✓ 通过' if hard_metrics.get('metric_b_pass') else '✗ 未通过'}",
        f"- **指标 C** (Dynamic_Sign_Switch 实现): "
        f"{'✓ 通过' if hard_metrics.get('metric_c_pass') else '✗ 未通过'}",
        "",
        "## 3. 有效交易天数",
        "",
    ])
    
    for year, days in year_trading_days.items():
        lines.append(f"- {year}年：{days}天")
    
    lines.extend([
        "",
        "## 4. IC 统计",
        "",
        f"- Mean IC: {ic_stats.get('mean_ic', 0.0):.4f}",
        f"- Mean Rank IC: {ic_stats.get('mean_rank_ic', 0.0):.4f}",
        f"- IC IR: {ic_stats.get('ic_ir', 0.0):.2f}",
        f"- Rank IC IR: {ic_stats.get('rank_ic_ir', 0.0):.2f}",
        f"- 正 IC 占比：{ic_stats.get('positive_ratio', 0.0):.2%}",
        "",
        "## 5. Dynamic_Sign_Switch 报告",
        "",
        "```",
        sign_switch_report,
        "```",
        "",
        "## 6. 逻辑一致性审计",
        "",
        "**问题**: 为什么 2021 年和 2024 年使用了相同的权重配置，其 IC 表现却截然不同？",
        "",
        "**回答**:",
        "",
        "```",
        logic_answer,
        "```",
        "",
        "---",
        "",
        "*V84 回测系统 - 版本 V84.0*",
    ])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    logger.info(f"V84: Markdown 报告已保存至 {output_path}")


if __name__ == "__main__":
    main()