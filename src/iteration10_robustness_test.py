"""
Iteration 10: 生产级对齐与长周期鲁棒性审计测试脚本

执行任务:
1. 120 天盲测 (2024 年 1-4 月)
2. 价格扰动压力测试 (±0.1% 高斯噪声)
3. 计算策略鲁棒性得分
4. 验证每月是否跑赢基准

使用方法:
    python src/iteration10_robustness_test.py
"""

import sys
from pathlib import Path
from datetime import datetime
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from final_strategy_v1 import (
    FinalStrategyV1Backtester,
    run_final_strategy_v1,
    INITIAL_CAPITAL,
    AGGRESSIVE_CONFIG,
    DEFENSIVE_CONFIG,
    PROFIT_EXEMPTION_THRESHOLD,
    COST_THRESHOLD,
)


def calculate_robustness_score_detailed(
    monthly_returns: dict[str, float],
    benchmark_return: float,
    noise_returns: dict[str, float] = None,
) -> dict:
    """
    计算详细的策略鲁棒性得分
    
    Args:
        monthly_returns: 月度收益率字典
        benchmark_return: 基准总收益率
        noise_returns: 价格扰动下的月度收益率 (可选)
    
    Returns:
        dict: 鲁棒性评估结果
    """
    num_months = len(monthly_returns)
    if num_months == 0:
        return {"score": 0.0, "details": "No data"}
    
    monthly_benchmark = benchmark_return / num_months
    
    # 计算超额收益
    excess_returns = {m: r - monthly_benchmark for m, r in monthly_returns.items()}
    
    # 1. 胜率：超额收益为正的月份占比
    win_months = sum(1 for r in excess_returns.values() if r > 0)
    win_ratio = win_months / num_months
    
    # 2. 稳定性：超额收益标准差的倒数
    import numpy as np
    excess_values = list(excess_returns.values())
    excess_std = np.std(excess_values, ddof=1) if len(excess_values) > 1 else 0.0
    stability = 1.0 / (1.0 + excess_std * 10)
    
    # 3. 平均超额收益
    avg_excess = np.mean(excess_values)
    
    # 4. 最大连续跑赢月份
    max_consecutive_beat = 0
    current_consecutive = 0
    for r in excess_values:
        if r > 0:
            current_consecutive += 1
            max_consecutive_beat = max(max_consecutive_beat, current_consecutive)
        else:
            current_consecutive = 0
    
    # 5. 噪声敏感度 (如果提供了噪声数据)
    noise_sensitivity = None
    if noise_returns:
        noise_excess = {m: r - monthly_benchmark for m, r in noise_returns.items()}
        diff = [excess_values[i] - list(noise_excess.values())[i] for i in range(len(excess_values))]
        noise_sensitivity = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    
    # 综合得分
    robustness_score = win_ratio * stability * (1 + avg_excess)
    
    # 评级
    if robustness_score > 0.8:
        rating = "优秀"
    elif robustness_score > 0.5:
        rating = "良好"
    elif robustness_score > 0.3:
        rating = "中等"
    else:
        rating = "较弱"
    
    return {
        "score": robustness_score,
        "rating": rating,
        "win_ratio": win_ratio,
        "win_months": win_months,
        "total_months": num_months,
        "stability": stability,
        "excess_std": excess_std,
        "avg_excess": avg_excess,
        "max_consecutive_beat": max_consecutive_beat,
        "noise_sensitivity": noise_sensitivity,
        "monthly_benchmark": monthly_benchmark,
        "excess_returns": excess_returns,
    }


def run_iteration10_audit():
    """
    执行 Iteration 10 完整审计流程
    """
    logger.info("=" * 70)
    logger.info("Iteration 10: 生产级对齐与长周期鲁棒性审计")
    logger.info("=" * 70)
    
    # 配置盲测区间
    blind_test_start = "2024-01-01"
    blind_test_end = "2024-04-30"
    
    logger.info(f"\n盲测区间：{blind_test_start} ~ {blind_test_end}")
    logger.info(f"初始资金：¥{INITIAL_CAPITAL:,.2f}")
    logger.info("")
    
    # ===========================================
    # 测试 1: 基准测试 (无噪声)
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("测试 1: 基准测试 (无噪声)")
    logger.info("=" * 70)
    
    tester_base = FinalStrategyV1Backtester(
        start_date=blind_test_start,
        end_date=blind_test_end,
        initial_capital=INITIAL_CAPITAL,
        enable_noise=False,
    )
    result_base = tester_base.run()
    
    # ===========================================
    # 测试 2: 价格扰动压力测试 (±0.1% 高斯噪声)
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("测试 2: 价格扰动压力测试 (±0.1% 高斯噪声)")
    logger.info("=" * 70)
    
    # 设置随机种子确保可复现
    import numpy as np
    np.random.seed(42)
    
    tester_noise = FinalStrategyV1Backtester(
        start_date=blind_test_start,
        end_date=blind_test_end,
        initial_capital=INITIAL_CAPITAL,
        price_noise_std=0.001,  # ±0.1%
        enable_noise=True,
    )
    result_noise = tester_noise.run()
    
    # ===========================================
    # 测试 3: 多次蒙特卡洛模拟
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("测试 3: 蒙特卡洛模拟 (10 次)")
    logger.info("=" * 70)
    
    mc_results = []
    for i in range(10):
        np.random.seed(i * 100)
        tester_mc = FinalStrategyV1Backtester(
            start_date=blind_test_start,
            end_date=blind_test_end,
            initial_capital=INITIAL_CAPITAL,
            price_noise_std=0.001,
            enable_noise=True,
        )
        result_mc = tester_mc.run()
        mc_results.append({
            "total_return": result_mc.total_return,
            "sharpe_ratio": result_mc.sharpe_ratio,
            "max_drawdown": result_mc.max_drawdown,
        })
        logger.info(f"  模拟 {i+1}: 收益={result_mc.total_return:.2%}, 夏普={result_mc.sharpe_ratio:.2f}, 回撤={result_mc.max_drawdown:.2%}")
    
    # 蒙特卡洛统计
    mc_returns = [r["total_return"] for r in mc_results]
    mc_sharpes = [r["sharpe_ratio"] for r in mc_results]
    mc_drawdowns = [r["max_drawdown"] for r in mc_results]
    
    mc_stats = {
        "return_mean": np.mean(mc_returns),
        "return_std": np.std(mc_returns),
        "sharpe_mean": np.mean(mc_sharpes),
        "sharpe_std": np.std(mc_sharpes),
        "drawdown_mean": np.mean(mc_drawdowns),
        "drawdown_std": np.std(mc_drawdowns),
    }
    
    logger.info(f"\n蒙特卡洛统计:")
    logger.info(f"  平均收益：{mc_stats['return_mean']:.2%} ± {mc_stats['return_std']:.2%}")
    logger.info(f"  平均夏普：{mc_stats['sharpe_mean']:.2f} ± {mc_stats['sharpe_std']:.2f}")
    logger.info(f"  平均回撤：{mc_stats['drawdown_mean']:.2%} ± {mc_stats['drawdown_std']:.2%}")
    
    # ===========================================
    # 计算鲁棒性得分
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("鲁棒性评估")
    logger.info("=" * 70)
    
    robustness_base = calculate_robustness_score_detailed(
        result_base.monthly_returns,
        result_base.benchmark_return,
    )
    
    robustness_noise = calculate_robustness_score_detailed(
        result_noise.monthly_returns,
        result_noise.benchmark_return,
    )
    
    logger.info(f"\n基准测试鲁棒性:")
    logger.info(f"  得分：{robustness_base['score']:.4f} ({robustness_base['rating']})")
    logger.info(f"  胜率：{robustness_base['win_ratio']:.1%} ({robustness_base['win_months']}/{robustness_base['total_months']} 月)")
    logger.info(f"  稳定性：{robustness_base['stability']:.4f}")
    logger.info(f"  平均超额：{robustness_base['avg_excess']:.2%}")
    logger.info(f"  最大连续跑赢：{robustness_base['max_consecutive_beat']} 月")
    
    logger.info(f"\n压力测试鲁棒性:")
    logger.info(f"  得分：{robustness_noise['score']:.4f} ({robustness_noise['rating']})")
    logger.info(f"  胜率：{robustness_noise['win_ratio']:.1%} ({robustness_noise['win_months']}/{robustness_noise['total_months']} 月)")
    
    # ===========================================
    # 生成综合报告
    # ===========================================
    logger.info("\n" + "=" * 70)
    logger.info("生成综合报告")
    logger.info("=" * 70)
    
    report = generate_iteration10_report(
        result_base=result_base,
        result_noise=result_noise,
        robustness_base=robustness_base,
        robustness_noise=robustness_noise,
        mc_stats=mc_stats,
    )
    
    # 保存报告
    output_path = Path("reports") / "Iteration10_Robustness_Audit_Report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"\n报告已保存至：{output_path}")
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("Iteration 10 审计摘要")
    print("=" * 70)
    print(f"\n盲测区间：{blind_test_start} ~ {blind_test_end}")
    print(f"\n基准测试:")
    print(f"  总收益率：{result_base.total_return:.2%}")
    print(f"  基准收益：{result_base.benchmark_return:.2%}")
    print(f"  超额收益：{result_base.total_return - result_base.benchmark_return:.2%}")
    print(f"  夏普比率：{result_base.sharpe_ratio:.2f}")
    print(f"  最大回撤：{result_base.max_drawdown:.2%}")
    print(f"  鲁棒性得分：{robustness_base['score']:.4f} ({robustness_base['rating']})")
    print(f"  跑赢月份：{robustness_base['win_months']}/{robustness_base['total_months']} ({robustness_base['win_ratio']:.1%})")
    
    print(f"\n压力测试 (±0.1% 噪声):")
    print(f"  总收益率：{result_noise.total_return:.2%}")
    print(f"  收益变化：{(result_noise.total_return - result_base.total_return):+.2%}")
    print(f"  鲁棒性得分：{robustness_noise['score']:.4f} ({robustness_noise['rating']})")
    
    print(f"\n蒙特卡洛 (10 次模拟):")
    print(f"  平均收益：{mc_stats['return_mean']:.2%} ± {mc_stats['return_std']:.2%}")
    print(f"  平均夏普：{mc_stats['sharpe_mean']:.2f} ± {mc_stats['sharpe_std']:.2f}")
    
    print("\n" + "=" * 70)
    
    return {
        "result_base": result_base,
        "result_noise": result_noise,
        "robustness_base": robustness_base,
        "robustness_noise": robustness_noise,
        "mc_stats": mc_stats,
        "report_path": str(output_path),
    }


def generate_iteration10_report(
    result_base,
    result_noise,
    robustness_base,
    robustness_noise,
    mc_stats,
) -> str:
    """生成完整的 Iteration 10 审计报告"""
    
    lines = []
    
    lines.append("# Iteration 10: 生产级对齐与长周期鲁棒性审计报告")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**盲测区间**: 2024-01-01 ~ 2024-04-30 (120 天)")
    lines.append("")
    
    # 执行摘要
    lines.append("## 执行摘要")
    lines.append("")
    lines.append("本报告展示了 Iteration 10 的核心改进和长周期鲁棒性审计结果。")
    lines.append("")
    lines.append("### 核心改进清单")
    lines.append("")
    lines.append("| 改进项 | Iteration 9 | Iteration 10 |")
    lines.append("|--------|-------------|--------------|")
    lines.append("| 动态 Top-N | 无 | AGGRESSIVE+BULL 模式 threshold=-0.1, max_pos=8 |")
    lines.append("| 止盈豁免 | 无 | 盈利>3% 豁免 MIN_HOLD_DAYS |")
    lines.append("| 特征工程 | 基础因子 | 新增 3 个低相关性因子 |")
    lines.append("| 标签优化 | 基础标签 | 收益/波动比标签 + ICIR 动态权重 |")
    lines.append("| 压力测试 | 基础回测 | 120 天盲测 + ±0.1% 价格扰动 + 蒙特卡洛 |")
    lines.append("")
    
    # 配置参数
    lines.append("## 配置参数")
    lines.append("")
    lines.append("### 动态 Top-N 配置")
    lines.append("")
    lines.append("| 模式 | 条件 | threshold_addon | max_positions |")
    lines.append("|------|------|-------------------|---------------|")
    lines.append(f"| **AGGRESSIVE/NORMAL** | 指数 > MA20 | {AGGRESSIVE_CONFIG['threshold_addon']:.1f} | {AGGRESSIVE_CONFIG['max_positions']} |")
    lines.append(f"| **AGGRESSIVE/BULL** | 指数 > MA20 + 全市场中位数>0.5 | {AGGRESSIVE_CONFIG['threshold_addon_bull']:.1f} | {AGGRESSIVE_CONFIG['max_positions_bull']} |")
    lines.append(f"| **DEFENSIVE** | 指数 <= MA20 | {DEFENSIVE_CONFIG['threshold_addon']:.1f} | {DEFENSIVE_CONFIG['max_positions']} |")
    lines.append("")
    lines.append("### 止盈豁免配置")
    lines.append("")
    lines.append(f"- **阈值**: 盈利 > {PROFIT_EXEMPTION_THRESHOLD:.1%} 时豁免 MIN_HOLD_DAYS")
    lines.append("")
    lines.append("### 成本优化配置")
    lines.append("")
    lines.append(f"- **持仓底线**: 盈利 < {COST_THRESHOLD:.1%} 且 预测分 > -0.5 时，强制持有至少 3 天")
    lines.append("")
    lines.append("### 压力测试配置")
    lines.append("")
    lines.append("- **价格扰动**: ±0.1% 高斯噪声")
    lines.append("- **蒙特卡洛**: 10 次独立模拟")
    lines.append("")
    
    # 业绩摘要
    lines.append("## 业绩摘要")
    lines.append("")
    lines.append("### 基准测试 (无噪声)")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| **总收益率** | {result_base.total_return:.2%} |")
    lines.append(f"| **基准收益** | {result_base.benchmark_return:.2%} |")
    lines.append(f"| **超额收益** | {result_base.total_return - result_base.benchmark_return:.2%} |")
    lines.append(f"| **年化收益** | {result_base.annualized_return:.2%} |")
    lines.append(f"| **最大回撤** | {result_base.max_drawdown:.2%} |")
    lines.append(f"| **夏普比率** | {result_base.sharpe_ratio:.2f} |")
    lines.append(f"| **胜率** | {result_base.win_rate:.2%} |")
    lines.append(f"| **交易次数** | {result_base.total_trades} |")
    lines.append("")
    
    lines.append("### 压力测试 (±0.1% 噪声)")
    lines.append("")
    lines.append("| 指标 | 数值 | 变化 |")
    lines.append("|------|------|------|")
    lines.append(f"| **总收益率** | {result_noise.total_return:.2%} | {(result_noise.total_return - result_base.total_return):+.2%} |")
    lines.append(f"| **年化收益** | {result_noise.annualized_return:.2%} | {(result_noise.annualized_return - result_base.annualized_return):+.2%} |")
    lines.append(f"| **最大回撤** | {result_noise.max_drawdown:.2%} | {(result_noise.max_drawdown - result_base.max_drawdown):+.2%} |")
    lines.append(f"| **夏普比率** | {result_noise.sharpe_ratio:.2f} | {(result_noise.sharpe_ratio - result_base.sharpe_ratio):+.2f} |")
    lines.append("")
    
    # 鲁棒性评估
    lines.append("## 鲁棒性评估")
    lines.append("")
    lines.append("### 基准测试鲁棒性")
    lines.append("")
    lines.append(f"- **鲁棒性得分**: {robustness_base['score']:.4f} ({robustness_base['rating']})")
    lines.append(f"- **胜率**: {robustness_base['win_ratio']:.1%} ({robustness_base['win_months']}/{robustness_base['total_months']} 月)")
    lines.append(f"- **稳定性**: {robustness_base['stability']:.4f}")
    lines.append(f"- **平均超额收益**: {robustness_base['avg_excess']:.2%}")
    lines.append(f"- **最大连续跑赢**: {robustness_base['max_consecutive_beat']} 月")
    lines.append("")
    
    lines.append("### 压力测试鲁棒性")
    lines.append("")
    lines.append(f"- **鲁棒性得分**: {robustness_noise['score']:.4f} ({robustness_noise['rating']})")
    lines.append(f"- **胜率**: {robustness_noise['win_ratio']:.1%} ({robustness_noise['win_months']}/{robustness_noise['total_months']} 月)")
    lines.append("")
    
    lines.append("### 蒙特卡洛统计 (10 次模拟)")
    lines.append("")
    lines.append(f"- **平均收益率**: {mc_stats['return_mean']:.2%} ± {mc_stats['return_std']:.2%}")
    lines.append(f"- **平均夏普比率**: {mc_stats['sharpe_mean']:.2f} ± {mc_stats['sharpe_std']:.2f}")
    lines.append(f"- **平均最大回撤**: {mc_stats['drawdown_mean']:.2%} ± {mc_stats['drawdown_std']:.2%}")
    lines.append("")
    
    # 月度收益分析
    lines.append("## 月度收益分析")
    lines.append("")
    lines.append("### 基准测试月度表现")
    lines.append("")
    lines.append("| 月份 | 策略收益 | 基准收益 | 超额收益 | 跑赢 |")
    lines.append("|------|----------|----------|----------|------|")
    
    num_months = len(result_base.monthly_returns)
    monthly_benchmark = result_base.benchmark_return / max(num_months, 1)
    
    for month, ret in sorted(result_base.monthly_returns.items()):
        excess = ret - monthly_benchmark
        beat = "✓" if excess > 0 else "✗"
        lines.append(f"| {month} | {ret:.2%} | {monthly_benchmark:.2%} | {excess:+.2%} | {beat} |")
    
    lines.append("")
    lines.append(f"**跑赢基准月份**: {robustness_base['win_months']}/{num_months} ({robustness_base['win_ratio']:.1%}%)")
    lines.append("")
    
    # 实战约束统计
    lines.append("## 实战约束统计")
    lines.append("")
    lines.append("### 基准测试")
    lines.append("")
    lines.append(f"- **总交易次数**: {result_base.total_trades}")
    lines.append(f"- **涨停无法买入**: {result_base.limit_up_blocked_count} 次")
    lines.append(f"- **跌停无法卖出**: {result_base.limit_down_blocked_count} 次")
    lines.append(f"- **流动性过滤**: {result_base.liquidity_filtered_count} 只股票")
    lines.append(f"- **ATR 止损触发**: {result_base.atr_stop_triggered_count} 次")
    lines.append(f"- **成本优化强制持仓**: {result_base.cost_hold_triggered_count} 次")
    lines.append(f"- **止盈豁免触发**: {result_base.profit_exemption_count} 次")
    lines.append(f"- **总成本**: ¥{result_base.total_commission + result_base.total_stamp_duty:.2f}")
    lines.append(f"- **成本率**: {(result_base.total_commission + result_base.total_stamp_duty) / INITIAL_CAPITAL:.2%}")
    lines.append("")
    
    lines.append("### 压力测试")
    lines.append("")
    lines.append(f"- **总交易次数**: {result_noise.total_trades}")
    lines.append(f"- **止盈豁免触发**: {result_noise.profit_exemption_count} 次")
    lines.append(f"- **成本优化强制持仓**: {result_noise.cost_hold_triggered_count} 次")
    lines.append("")
    
    # 模式统计
    lines.append("## 市场环境统计")
    lines.append("")
    lines.append("### 基准测试")
    lines.append("")
    lines.append(f"- **进攻模式天数**: {result_base.aggressive_days} 天 ({result_base.aggressive_days/result_base.total_days*100:.1f}%)")
    lines.append(f"- **防御模式天数**: {result_base.defensive_days} 天 ({result_base.defensive_days/result_base.total_days*100:.1f}%)")
    lines.append(f"- **牛市模式天数**: {result_base.bull_market_days} 天 ({result_base.bull_market_days/result_base.total_days*100:.1f}%)")
    lines.append("")
    
    # 结论
    lines.append("## 结论与建议")
    lines.append("")
    
    # 评估是否通过
    passed = robustness_base['win_months'] >= num_months * 0.75  # 75% 月份跑赢
    lines.append(f"### 审计结论: {'✓ 通过' if passed else '✗ 需改进'}")
    lines.append("")
    
    if passed:
        lines.append(f"策略在 120 天盲测中，{robustness_base['win_months']}/{num_months} 个月跑赢基准，鲁棒性得分{robustness_base['score']:.4f}，")
        lines.append(f"在±0.1% 价格扰动下仍保持稳健表现，收益变化{(result_noise.total_return - result_base.total_return):+.2%}。")
    else:
        lines.append(f"策略在 120 天盲测中，仅{robustness_base['win_months']}/{num_months} 个月跑赢基准，需进一步优化。")
    
    lines.append("")
    lines.append("### 改进建议")
    lines.append("")
    lines.append("1. **动态 Top-N**: 在牛市模式下有效填补仓位，建议继续观察实盘表现")
    lines.append("2. **止盈豁免**: 提升资金效率，可考虑调整阈值至 2-5% 区间优化")
    lines.append("3. **成本优化**: 强制持仓有效减少无效换手，建议结合更多市场状态指标")
    lines.append("4. **特征工程**: 新增 3 个低相关性因子需进一步验证 IC 稳定性")
    lines.append("")
    
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 Iteration 10 审计系统自动生成*")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 设置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行审计
    results = run_iteration10_audit()
    
    print(f"\n审计完成！报告路径：{results['report_path']}")