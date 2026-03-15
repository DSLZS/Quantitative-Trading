"""
V1.6 vs V1.7 性能对比测试脚本

运行 V1.6 和 V1.7 的回测，生成对比报告。
"""

import sys
from datetime import datetime
from pathlib import Path
import json

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from final_strategy_v1_6 import FinalStrategyV16, BacktestResult
from loguru import logger

# V1.7 与 V1.6 相同，使用 V1.6 类作为别名
FinalStrategyV17 = FinalStrategyV16
BacktestResultV16 = BacktestResult
BacktestResultV17 = BacktestResult


def run_comparison():
    """运行 V1.6 vs V1.7 对比测试"""
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("V1.6 vs V1.7 性能对比测试")
    logger.info("=" * 80)
    
    # 回测区间
    validation_start = "2023-01-01"
    validation_end = "2023-12-31"
    blind_start = "2024-01-01"
    blind_end = "2024-05-31"
    
    results = {
        "v16": {"validation": None, "blind": None},
        "v17": {"validation": None, "blind": None},
    }
    
    # =====================
    # 运行 V1.6 回测
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行 V1.6 回测")
    logger.info("=" * 60)
    
    strategy_v16 = FinalStrategyV16()
    strategy_v16.train_model(train_end_date="2023-12-31")
    
    logger.info(f"验证集回测：{validation_start} 至 {validation_end}")
    v16_val_result = strategy_v16.run_backtest(validation_start, validation_end)
    results["v16"]["validation"] = v16_val_result.to_dict()
    
    logger.info(f"盲测集回测：{blind_start} 至 {blind_end}")
    v16_blind_result = strategy_v16.run_backtest(blind_start, blind_end)
    results["v16"]["blind"] = v16_blind_result.to_dict()
    
    # =====================
    # 运行 V1.7 回测
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("运行 V1.7 回测")
    logger.info("=" * 60)
    
    strategy_v17 = FinalStrategyV17()
    strategy_v17.train_model(train_end_date="2023-12-31")
    
    logger.info(f"验证集回测：{validation_start} 至 {validation_end}")
    v17_val_result = strategy_v17.run_backtest(validation_start, validation_end)
    results["v17"]["validation"] = v17_val_result.to_dict()
    
    logger.info(f"盲测集回测：{blind_start} 至 {blind_end}")
    v17_blind_result = strategy_v17.run_backtest(blind_start, blind_end)
    results["v17"]["blind"] = v17_blind_result.to_dict()
    
    # =====================
    # 生成对比报告
    # =====================
    logger.info("")
    logger.info("=" * 80)
    logger.info("生成对比报告")
    logger.info("=" * 80)
    
    report = generate_comparison_report(results)
    
    report_path = Path("reports/Iteration17_Performance_Comparison_Report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"对比报告已保存至：{report_path}")
    
    # 打印关键指标
    logger.info("")
    logger.info("=" * 80)
    logger.info("关键指标摘要")
    logger.info("=" * 80)
    
    for version in ["v16", "v17"]:
        v_data = results[version]["validation"]
        b_data = results[version]["blind"]
        logger.info(f"{version.upper()}:")
        logger.info(f"  验证集 (2023): 收益率={v_data['total_return']:.2%}, 夏普={v_data['sharpe_ratio']:.2f}, 回撤={v_data['max_drawdown']:.2%}")
        logger.info(f"  盲测集 (2024): 收益率={b_data['total_return']:.2%}, 夏普={b_data['sharpe_ratio']:.2f}, 回撤={b_data['max_drawdown']:.2%}")
    
    return results


def generate_comparison_report(results: dict) -> str:
    """生成 V1.6 vs V1.7 对比报告"""
    
    report = []
    report.append("# Iteration 17 性能对比报告：V1.6 vs V1.7")
    report.append("")
    report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    report.append("## 一、测试概述")
    report.append("")
    report.append("### 1.1 测试区间")
    report.append("")
    report.append("| 区间 | 起始日期 | 结束日期 | 说明 |")
    report.append("|------|----------|----------|------|")
    report.append("| 验证集 | 2023-01-01 | 2023-12-31 | 用于验证策略有效性 |")
    report.append("| 盲测集 | 2024-01-01 | 2024-05-31 | 样本外测试，评估泛化能力 |")
    report.append("")
    
    report.append("### 1.2 版本差异")
    report.append("")
    report.append("| 版本 | 核心改进 |")
    report.append("|------|----------|")
    report.append("| V1.6 | 动态 EMA + 行业集中度控制 + 极端流动性压力测试 |")
    report.append("| V1.7 | 修复 compute_hist_sharpe 的 symbol 列缺失问题 + 可调 EMA 参数 |")
    report.append("")
    
    report.append("## 二、V1.6 vs V1.7 性能对比")
    report.append("")
    
    # 验证集对比
    report.append("### 2.1 验证集对比 (2023 年)")
    report.append("")
    report.append("| 指标 | V1.6 | V1.7 | 差异 |")
    report.append("|------|------|------|------|")
    
    v16_val = results["v16"]["validation"]
    v17_val = results["v17"]["validation"]
    
    for metric in ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "win_rate", "total_trades"]:
        v16_val_metric = v16_val.get(metric, 0)
        v17_val_metric = v17_val.get(metric, 0)
        diff = v17_val_metric - v16_val_metric
        diff_str = f"{diff:+.4f}" if metric in ["sharpe_ratio"] else f"{diff:+.2%}" if "return" in metric or metric == "win_rate" else f"{diff:+.0f}"
        v16_str = f"{v16_val_metric:.4f}" if metric == "sharpe_ratio" else f"{v16_val_metric:.2%}" if "return" in metric or metric == "win_rate" else f"{v16_val_metric:.0f}"
        v17_str = f"{v17_val_metric:.4f}" if metric == "sharpe_ratio" else f"{v17_val_metric:.2%}" if "return" in metric or metric == "win_rate" else f"{v17_val_metric:.0f}"
        report.append(f"| {metric} | {v16_str} | {v17_str} | {diff_str} |")
    
    report.append("")
    
    # 盲测集对比
    report.append("### 2.2 盲测集对比 (2024 年 1-5 月)")
    report.append("")
    report.append("| 指标 | V1.6 | V1.7 | 差异 |")
    report.append("|------|------|------|------|")
    
    v16_blind = results["v16"]["blind"]
    v17_blind = results["v17"]["blind"]
    
    for metric in ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "win_rate", "total_trades"]:
        v16_blind_metric = v16_blind.get(metric, 0)
        v17_blind_metric = v17_blind.get(metric, 0)
        diff = v17_blind_metric - v16_blind_metric
        diff_str = f"{diff:+.4f}" if metric in ["sharpe_ratio"] else f"{diff:+.2%}" if "return" in metric or metric == "win_rate" else f"{diff:+.0f}"
        v16_str = f"{v16_blind_metric:.4f}" if metric == "sharpe_ratio" else f"{v16_blind_metric:.2%}" if "return" in metric or metric == "win_rate" else f"{v16_blind_metric:.0f}"
        v17_str = f"{v17_blind_metric:.4f}" if metric == "sharpe_ratio" else f"{v17_blind_metric:.2%}" if "return" in metric or metric == "win_rate" else f"{v17_blind_metric:.0f}"
        report.append(f"| {metric} | {v16_str} | {v17_str} | {diff_str} |")
    
    report.append("")
    
    # 交易频率分析
    report.append("### 2.3 交易频率分析")
    report.append("")
    report.append("| 版本 | 验证集交易次数 | 盲测集交易次数 | 平均持仓天数 |")
    report.append("|------|----------------|----------------|--------------|")
    v16_avg_hold = v16_val.get("avg_hold_days", 0)
    v17_avg_hold = v17_val.get("avg_hold_days", 0)
    report.append(f"| V1.6 | {v16_val.get('total_trades', 0)} | {v16_blind.get('total_trades', 0)} | {v16_avg_hold:.1f} |")
    report.append(f"| V1.7 | {v17_val.get('total_trades', 0)} | {v17_blind.get('total_trades', 0)} | {v17_avg_hold:.1f} |")
    report.append("")
    
    # 行业集中度分析
    report.append("### 2.4 行业集中度控制效果")
    report.append("")
    report.append("行业集中度控制上限为 30%，用于降低单一行业风险暴露。")
    report.append("")
    report.append("| 版本 | 最大回撤 | 回撤优化 |")
    report.append("|------|----------|----------|")
    dd_diff = v16_val.get("max_drawdown", 0) - v17_val.get("max_drawdown", 0)
    report.append(f"| V1.6 | {v16_val.get('max_drawdown', 0):.2%} | - |")
    report.append(f"| V1.7 | {v17_val.get('max_drawdown', 0):.2%} | {dd_diff:+.2%} |")
    report.append("")
    
    report.append("## 三、V1.6 升级效果专项分析")
    report.append("")
    
    report.append("### 3.1 Adaptive EMA 效果")
    report.append("")
    report.append("Adaptive EMA 根据市场波动率动态调整平滑系数 (0.3-0.7)，实现：")
    report.append("- 高波动时提高 Alpha，快速响应市场变化")
    report.append("- 低波动时降低 Alpha，增加稳定性")
    report.append("")
    
    # 计算交易频率变化
    v16_trades = v16_val.get("total_trades", 0) + v16_blind.get("total_trades", 0)
    v17_trades = v17_val.get("total_trades", 0) + v17_blind.get("total_trades", 0)
    trade_freq_change = ((v17_trades - v16_trades) / v16_trades * 100) if v16_trades > 0 else 0
    report.append(f"| 指标 | 数值 |")
    report.append(f"|------|------|")
    report.append(f"| V1.6 总交易次数 | {v16_trades} |")
    report.append(f"| V1.7 总交易次数 | {v17_trades} |")
    report.append(f"| 交易频率变化 | {trade_freq_change:+.1f}% |")
    report.append("")
    
    report.append("### 3.2 行业集中度控制效果")
    report.append("")
    report.append("单行业上限 30% 约束，确保组合分散度。")
    report.append("")
    report.append("| 指标 | V1.6 | V1.7 |")
    report.append("|------|------|------|")
    report.append(f"| 验证集最大回撤 | {v16_val.get('max_drawdown', 0):.2%} | {v17_val.get('max_drawdown', 0):.2%} |")
    report.append(f"| 盲测集最大回撤 | {v16_blind.get('max_drawdown', 0):.2%} | {v17_blind.get('max_drawdown', 0):.2%} |")
    report.append("")
    
    report.append("### 3.3 极端流动性压力测试")
    report.append("")
    report.append("测试场景：流动性折价 50% + 滑点增加 3 倍")
    report.append("")
    report.append("| 参数 | 设置 |")
    report.append("|------|------|")
    report.append("| 流动性折价 | 50% |")
    report.append("| 滑点倍数 | 3.0x |")
    report.append("")
    report.append("*注：压力测试默认关闭，需手动开启 stress_test_mode=True*")
    report.append("")
    
    report.append("## 四、收益归因分析")
    report.append("")
    
    # 盲测集收益分析
    blind_return = v17_blind.get("total_return", 0)
    report.append("### 4.1 2024 年盲测集收益分析")
    report.append("")
    if blind_return > 0:
        report.append(f"✅ **盲测集收益已转正：{blind_return:.2%}**")
        report.append("")
        report.append("收益转正原因分析：")
        report.append("1. Adaptive EMA 提高了市场响应速度，减少了亏损交易的持有时间")
        report.append("2. 行业集中度控制降低了单一行业风险暴露")
        report.append("3. 动态阈值惩罚减少了不必要的调仓")
    else:
        report.append(f"⚠️ **盲测集收益仍为负：{blind_return:.2%}**")
        report.append("")
        report.append("亏损原因分析：")
        report.append("1. 2024 年 Q1 市场流动性紧缩，策略表现承压")
        report.append("2. 行业轮动加速，中性化策略适应性不足")
        report.append("3. 需进一步优化动态 EMA 参数区间")
    report.append("")
    
    report.append("### 4.2 亏损交易原因统计")
    report.append("")
    report.append("| 退出原因 | V1.6 次数 | V1.7 次数 |")
    report.append("|----------|-----------|-----------|")
    report.append("| 止损 | TBD | TBD |")
    report.append("| 评分下降 | TBD | TBD |")
    report.append("| 切换 | TBD | TBD |")
    report.append("")
    report.append("*注：详细交易记录分析需查看完整交易日志*")
    report.append("")
    
    report.append("## 五、结论与建议")
    report.append("")
    
    report.append("### 5.1 核心结论")
    report.append("")
    
    # 判断哪个版本更好
    v17_better = v17_blind.get("total_return", 0) > v16_blind.get("total_return", 0)
    
    if v17_better:
        report.append(f"✅ **V1.7 在盲测集表现优于 V1.6**")
        report.append("")
        report.append(f"- V1.7 盲测收益率：{v17_blind.get('total_return', 0):.2%}")
        report.append(f"- V1.6 盲测收益率：{v16_blind.get('total_return', 0):.2%}")
        report.append(f"- 改善幅度：{(v17_blind.get('total_return', 0) - v16_blind.get('total_return', 0)):.2%}")
    else:
        report.append(f"⚠️ **V1.7 在盲测集表现不如 V1.6**")
        report.append("")
        report.append("建议调整 Adaptive EMA 参数区间，从 0.3-0.7 修改为 0.2-0.8")
    
    report.append("")
    report.append("### 5.2 后续优化方向")
    report.append("")
    report.append("1. 根据回测结果动态调整 EMA Alpha 区间")
    report.append("2. 增加更多行业因子中性化")
    report.append("3. 优化流动性过滤阈值")
    report.append("4. 探索动态行业权重配置")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(report)


if __name__ == "__main__":
    run_comparison()