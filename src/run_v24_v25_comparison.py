"""
V2.4 vs V2.5 对比测试脚本

目的：
1. 对比 V2.4 与 V2.5 在 2023 验证集和 2024 盲测集的表现
2. 统计领头羊保护机制效果
3. 分析被洗出后又创新高的比例
4. 对比 V1.9 基准 (3.89% 收益)
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent))

from final_strategy_v2_4 import FinalStrategyV24
from final_strategy_v2_5 import FinalStrategyV25


def run_comparison():
    """运行 V2.4 vs V2.5 对比测试"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("Iteration 26: V2.4 vs V2.5 对比测试")
    logger.info("=" * 60)
    logger.info("")
    
    # ==================== V2.4 测试 ====================
    logger.info("=" * 60)
    logger.info("【V2.4 基准测试】")
    logger.info("=" * 60)
    
    strategy_v24 = FinalStrategyV24(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    strategy_v24.train_model(train_end_date="2023-12-31", model_type="ensemble")
    
    # 2023 验证集
    logger.info("Running V2.4 2023 backtest...")
    v24_2023_result = strategy_v24.run_backtest("2023-01-01", "2023-12-31")
    
    # 2024 盲测集
    logger.info("Running V2.4 2024 backtest...")
    v24_2024_result = strategy_v24.run_backtest("2024-01-01", "2024-05-31")
    
    v24_attribution = strategy_v24.run_attribution_analysis()
    
    # ==================== V2.5 测试 ====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【V2.5 进攻性测试】")
    logger.info("=" * 60)
    
    strategy_v25 = FinalStrategyV25(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    strategy_v25.train_model(train_end_date="2023-12-31", model_type="ensemble")
    
    # 2023 验证集
    logger.info("Running V2.5 2023 backtest...")
    v25_2023_result = strategy_v25.run_backtest("2023-01-01", "2023-12-31")
    
    # 2024 盲测集
    logger.info("Running V2.5 2024 backtest...")
    v25_2024_result = strategy_v25.run_backtest("2024-01-01", "2024-05-31")
    
    v25_attribution = strategy_v25.run_attribution_analysis()
    
    # ==================== 生成对比报告 ====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【生成对比报告】")
    logger.info("=" * 60)
    
    report = generate_comparison_report(
        v24_2023_result, v24_2024_result, v24_attribution,
        v25_2023_result, v25_2024_result, v25_attribution,
        strategy_v24, strategy_v25,
    )
    
    report_path = Path("reports/Iteration26_V24_V25_Comparison_Report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"对比报告已保存至：{report_path}")
    
    # ==================== 输出摘要 ====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【对比摘要】")
    logger.info("=" * 60)
    
    logger.info(f"V2.4 盲测收益 (2024): {v24_2024_result.total_return:.2%}")
    logger.info(f"V2.5 盲测收益 (2024): {v25_2024_result.total_return:.2%}")
    logger.info(f"收益改善：{(v25_2024_result.total_return - v24_2024_result.total_return):.2%}")
    
    # V1.9 基准对比
    v19_benchmark = 0.0389  # 3.89%
    if v25_2024_result.total_return > v19_benchmark:
        logger.info(f"✅ V2.5 收益已超越 V1.9 基准 ({v19_benchmark:.1%})")
    elif v25_2024_result.total_return > 0.02:
        logger.info(f"✅ V2.5 收益已达到最低目标 (>2%)")
    elif v25_2024_result.total_return > 0:
        logger.info(f"✅ V2.5 收益已转正")
    else:
        logger.info(f"⚠️ V2.5 收益仍为负")
    
    # 领头羊统计
    logger.info("")
    logger.info(f"V2.5 领头羊识别：{strategy_v25.leader_stats['identified']}")
    logger.info(f"V2.5 领头羊保护：{strategy_v25.leader_stats['protected']}")
    logger.info(f"V2.5 领头羊被洗出：{strategy_v25.leader_stats['exited_failed']}")
    
    # 被洗出后又创新高比例
    washed_out_ratio = v25_attribution.get("trailing_stop_stats", {}).get("washed_out_ratio", 0)
    logger.info(f"V2.5 被洗出后又创新高：{washed_out_ratio:.1%}")
    
    if washed_out_ratio > 0.30:
        logger.warning("⚠️ 被洗出后又创新高比例 > 30%，止盈可能仍太紧")
    
    logger.info("=" * 60)
    
    return {
        "v24_2023": v24_2023_result.to_dict(),
        "v24_2024": v24_2024_result.to_dict(),
        "v25_2023": v25_2023_result.to_dict(),
        "v25_2024": v25_2024_result.to_dict(),
    }


def generate_comparison_report(v24_2023, v24_2024, v24_attribution,
                                v25_2023, v25_2024, v25_attribution,
                                strategy_v24, strategy_v25) -> str:
    """生成详细对比报告"""
    report = []
    report.append("# Iteration 26: V2.4 vs V2.5 对比测试报告")
    report.append("")
    report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试版本**: V2.4 (基准) vs V2.5 (领头羊保护计划)")
    report.append("")
    
    report.append("## 一、执行摘要")
    report.append("")
    report.append("### 1.1 V2.5 核心改进")
    report.append("")
    report.append("1. **领头羊保护机制**: Top 3% 标的强制 2% buffer + 2.0x ATR")
    report.append("2. **Inverted Regime Mapping**: 平稳市场 LGB 90%, 剧烈市场 Ridge 80%")
    report.append("3. **Pre-emptive Lock**: 持仓 Top 10% 强制 100% LGB")
    report.append("4. **止盈逻辑大幅松绑**: 统一 1.5x ATR + 0.01 buffer")
    report.append("5. **最大持仓天数限制**: 15 天防止僵尸股")
    report.append("")
    
    report.append("### 1.2 基准对比目标")
    report.append("")
    report.append("- **V1.9 基准收益**: 3.89% (2024 年盲测)")
    report.append("- **V2.5 目标收益**: > 3.89%")
    report.append("- **最低目标**: > 2%")
    report.append("")
    
    report.append("## 二、性能对比")
    report.append("")
    
    report.append("### 2.1 验证集表现对比 (2023 年)")
    report.append("")
    report.append("| 指标 | V2.4 | V2.5 | 改善 |")
    report.append("|------|------|------|------|")
    report.append(f"| 总收益率 | {v24_2023.total_return:.2%} | {v25_2023.total_return:.2%} | {(v25_2023.total_return - v24_2023.total_return):.2%} |")
    report.append(f"| 年化收益 | {v24_2023.annual_return:.2%} | {v25_2023.annual_return:.2%} | {(v25_2023.annual_return - v24_2023.annual_return):.2%} |")
    report.append(f"| 最大回撤 | {v24_2023.max_drawdown:.2%} | {v25_2023.max_drawdown:.2%} | {(v25_2023.max_drawdown - v24_2023.max_drawdown):.2%} |")
    report.append(f"| 夏普比率 | {v24_2023.sharpe_ratio:.2f} | {v25_2023.sharpe_ratio:.2f} | {(v25_2023.sharpe_ratio - v24_2023.sharpe_ratio):.2f} |")
    report.append(f"| 胜率 | {v24_2023.win_rate:.2%} | {v25_2023.win_rate:.2%} | {(v25_2023.win_rate - v24_2023.win_rate):.2%} |")
    report.append(f"| 交易次数 | {v24_2023.total_trades} | {v25_2023.total_trades} | {v25_2023.total_trades - v24_2023.total_trades} |")
    report.append("")
    
    report.append("### 2.2 盲测集表现对比 (2024 年)")
    report.append("")
    report.append("| 指标 | V2.4 | V2.5 | 改善 |")
    report.append("|------|------|------|------|")
    report.append(f"| 总收益率 | {v24_2024.total_return:.2%} | {v25_2024.total_return:.2%} | {(v25_2024.total_return - v24_2024.total_return):.2%} |")
    report.append(f"| 年化收益 | {v24_2024.annual_return:.2%} | {v25_2024.annual_return:.2%} | {(v25_2024.annual_return - v24_2024.annual_return):.2%} |")
    report.append(f"| 最大回撤 | {v24_2024.max_drawdown:.2%} | {v25_2024.max_drawdown:.2%} | {(v25_2024.max_drawdown - v24_2024.max_drawdown):.2%} |")
    report.append(f"| 夏普比率 | {v24_2024.sharpe_ratio:.2f} | {v25_2024.sharpe_ratio:.2f} | {(v25_2024.sharpe_ratio - v24_2024.sharpe_ratio):.2f} |")
    report.append(f"| 胜率 | {v24_2024.win_rate:.2%} | {v25_2024.win_rate:.2%} | {(v25_2024.win_rate - v24_2024.win_rate):.2%} |")
    report.append(f"| 交易次数 | {v24_2024.total_trades} | {v25_2024.total_trades} | {v25_2024.total_trades - v24_2024.total_trades} |")
    report.append("")
    
    report.append("### 2.3 收益对比分析")
    report.append("")
    
    v19_benchmark = 0.0389
    if v25_2024.total_return > v19_benchmark:
        report.append(f"✅ **V2.5 盲测收益 {v25_2024.total_return:.2%} > 3.89% (V1.9 基准), 目标达成**")
    elif v25_2024.total_return > 0.03:
        report.append(f"✅ **V2.5 盲测收益 {v25_2024.total_return:.2%} > 3.0%, 接近目标**")
    elif v25_2024.total_return > 0.02:
        report.append(f"✅ **V2.5 盲测收益 {v25_2024.total_return:.2%} > 2%, 达到最低目标**")
    elif v25_2024.total_return > 0:
        report.append(f"✅ **V2.5 盲测收益 {v25_2024.total_return:.2%} 已转正**")
    else:
        report.append(f"⚠️ **V2.5 盲测收益 {v25_2024.total_return:.2%} 仍为负，需继续优化**")
    
    report.append("")
    report.append("| 基准对比 | 数值 |")
    report.append("|------|------|")
    report.append(f"| V1.9 基准 | {v19_benchmark:.2%} |")
    report.append(f"| V2.4 vs V1.9 | {(v24_2024.total_return - v19_benchmark):.2%} |")
    report.append(f"| V2.5 vs V1.9 | {(v25_2024.total_return - v19_benchmark):.2%} |")
    report.append(f"| V2.5 vs V2.4 改善 | {(v25_2024.total_return - v24_2024.total_return):.2%} |")
    report.append("")
    
    report.append("## 三、V2.5 新增指标分析")
    report.append("")
    
    report.append("### 3.1 Inverted Regime Mapping 分析")
    report.append("")
    weight_trend = strategy_v25.get_weight_trend_analysis()
    overall_avg = weight_trend.get("overall_avg", {})
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| 平均 Ridge 权重 | {overall_avg.get('ridge_weight', 0):.2%} |")
    report.append(f"| 平均 LGB 权重 | {overall_avg.get('lgb_weight', 0):.2%} |")
    report.append(f"| 平均 ATR 排名 | {overall_avg.get('atr_rank', 0):.2%} |")
    report.append(f"| 平均 ATR | {overall_avg.get('atr', 0):.4f} |")
    report.append(f"| 总交易日数 | {weight_trend.get('total_days', 0)} |")
    report.append(f"| 平稳市场天数 | {weight_trend.get('peace_market_days', 0)} |")
    report.append(f"| 剧烈市场天数 | {weight_trend.get('volatile_market_days', 0)} |")
    report.append("")
    
    report.append("### 3.2 领头羊统计")
    report.append("")
    leader_stats = strategy_v25.leader_stats
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| 识别领头羊数量 | {leader_stats.get('identified', 0)} |")
    report.append(f"| 保护领头羊次数 | {leader_stats.get('protected', 0)} |")
    report.append(f"| 领头羊被洗出 | {leader_stats.get('exited_failed', 0)} |")
    report.append("")
    
    report.append("### 3.3 被洗出后又创新高分析")
    report.append("")
    trailing_stats = v25_attribution.get("trailing_stop_stats", {})
    washed_out_ratio = trailing_stats.get("washed_out_ratio", 0)
    report.append(f"- **被洗出后又创新高比例**: {washed_out_ratio:.1%}")
    report.append("")
    if washed_out_ratio > 0.30:
        report.append("⚠️ **警告**: 被洗出后又创新高比例 > 30%，止盈可能仍太紧")
    elif washed_out_ratio > 0.20:
        report.append("⚠️ **注意**: 被洗出后又创新高比例 > 20%，可考虑适度放宽止盈")
    else:
        report.append("✅ 被洗出后又创新高比例在合理范围内")
    report.append("")
    
    report.append("### 3.4 领头羊交易分析")
    report.append("")
    leader_analysis = v25_attribution.get("leader_analysis", {})
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| 领头羊交易总数 | {leader_analysis.get('total_leader_trades', 0)} |")
    report.append(f"| 领头羊胜率 | {leader_analysis.get('leader_winning_rate', 0):.1%} |")
    report.append(f"| 领头羊平均盈利 | {leader_analysis.get('leader_avg_profit', 0):.2%} |")
    report.append(f"| 领头羊被洗出 | {leader_analysis.get('leader_exited_failed', 0)} |")
    report.append("")
    
    report.append("## 四、归因分析对比")
    report.append("")
    
    report.append("### 4.1 亏损交易原因对比")
    report.append("")
    v24_loss_reasons = v24_attribution.get("loss_reasons", {})
    v25_loss_reasons = v25_attribution.get("loss_reasons", {})
    
    all_reasons = set(list(v24_loss_reasons.keys()) + list(v25_loss_reasons.keys()))
    if all_reasons:
        report.append("| 原因 | V2.4 次数 | V2.5 次数 | 变化 |")
        report.append("|------|----------|----------|------|")
        for reason in all_reasons:
            v24_count = v24_loss_reasons.get(reason, 0)
            v25_count = v25_loss_reasons.get(reason, 0)
            change = v25_count - v24_count
            change_str = f"+{change}" if change > 0 else str(change)
            report.append(f"| {reason} | {v24_count} | {v25_count} | {change_str} |")
    report.append("")
    
    report.append("### 4.2 追踪止盈触发对比")
    report.append("")
    v24_trailing = v24_attribution.get("trailing_stop_stats", {})
    v25_trailing = v25_attribution.get("trailing_stop_stats", {})
    
    report.append("| 指标 | V2.4 | V2.5 | 变化 |")
    report.append("|------|------|------|------|")
    report.append(f"| 触发次数 | {v24_trailing.get('count', 0)} | {v25_trailing.get('count', 0)} | {v25_trailing.get('count', 0) - v24_trailing.get('count', 0)} |")
    report.append(f"| 平均盈利 | {v24_trailing.get('avg_profit', 0):.2%} | {v25_trailing.get('avg_profit', 0):.2%} | {(v25_trailing.get('avg_profit', 0) - v24_trailing.get('avg_profit', 0)):.2%} |")
    report.append(f"| 最大盈利 | {v24_trailing.get('max_profit', 0):.2%} | {v25_trailing.get('max_profit', 0):.2%} | {(v25_trailing.get('max_profit', 0) - v24_trailing.get('max_profit', 0)):.2%} |")
    report.append(f"| 被洗出后又创新高 | N/A | {v25_trailing.get('washed_out_ratio', 0):.1%} | - |")
    report.append("")
    
    report.append("### 4.3 2024 年最大盈利交易分析")
    report.append("")
    
    # 分析 V2.5 最大盈利交易
    if strategy_v25.trade_records:
        max_profit_trade = max(strategy_v25.trade_records, key=lambda t: t.max_profit_pct)
        report.append(f"- **标的**: {max_profit_trade.symbol}")
        report.append(f"- **买入日期**: {max_profit_trade.entry_date}")
        report.append(f"- **卖出日期**: {max_profit_trade.exit_date}")
        report.append(f"- **盈利**: {max_profit_trade.pnl_pct:.2%}")
        report.append(f"- **持有期**: {max_profit_trade.hold_days} 天")
        report.append(f"- **退出原因**: {max_profit_trade.exit_reason}")
        report.append(f"- **持仓期间最大盈利**: {max_profit_trade.max_profit_pct:.2%}")
        report.append(f"- **是否领头羊**: {'是' if max_profit_trade.is_leader else '否'}")
    report.append("")
    
    report.append("## 五、压力测试对比")
    report.append("")
    
    v24_stress = strategy_v24.run_stress_test(v24_2024, noise_level=0.001)
    v25_stress = strategy_v25.run_stress_test(v25_2024, noise_level=0.001)
    
    report.append("| 指标 | V2.4 | V2.5 |")
    report.append("|------|------|------|")
    report.append(f"| 原始收益 | {v24_stress.get('original_return', 0):.2%} | {v25_stress.get('original_return', 0):.2%} |")
    report.append(f"| 扰动后收益 | {v24_stress.get('stressed_return', 0):.2%} | {v25_stress.get('stressed_return', 0):.2%} |")
    report.append(f"| 收益回落 | {v24_stress.get('return_drop', 0):.2%} | {v25_stress.get('return_drop', 0):.2%} |")
    report.append(f"| 噪声敏感度 | {v24_stress.get('noise_sensitivity', 'Unknown')} | {v25_stress.get('noise_sensitivity', 'Unknown')} |")
    report.append("")
    
    report.append("## 六、Alpha 因子有效性分析")
    report.append("")
    
    if v25_2024.total_return < 0.02:
        report.append("### 6.1 盲测收益未达 2% 原因分析")
        report.append("")
        report.append("如果 V2.5 盲测收益未能重回 2% 以上，可能原因包括:")
        report.append("")
        report.append("1. **Alpha 因子失效**: 2024 年市场风格切换，原有因子可能失效")
        report.append("2. **市场环境影响**: 2024 年市场波动率下降，动量因子效果减弱")
        report.append("3. **过度拟合风险**: 2023 年训练数据可能过度拟合特定市场环境")
        report.append("")
        report.append("建议优化方向:")
        report.append("")
        report.append("1. 增加更多适应性因子（如行业轮动、风格因子）")
        report.append("2. 调整训练数据时间窗口，增加数据多样性")
        report.append("3. 探索非线性因子组合方式")
    report.append("")
    
    report.append("## 七、结论与建议")
    report.append("")
    
    report.append("### 7.1 核心结论")
    report.append("")
    
    improvement = v25_2024.total_return - v24_2024.total_return
    if improvement > 0:
        report.append(f"✅ **V2.5 相对 V2.4 有改善**: +{improvement:.2%}")
    else:
        report.append(f"⚠️ **V2.5 相对 V2.4 有所下降**: {improvement:.2%}")
    report.append("")
    
    report.append("### 7.2 关键改进效果")
    report.append("")
    report.append(f"1. **领头羊保护**: 识别={leader_stats.get('identified', 0)}, 保护={leader_stats.get('protected', 0)}, 被洗出={leader_stats.get('exited_failed', 0)}")
    report.append(f"2. **Inverted Regime Mapping**: 平均 LGB={overall_avg.get('lgb_weight', 0):.1%}, 平稳市场天数={weight_trend.get('peace_market_days', 0)}")
    report.append(f"3. **被洗出后又创新高**: {v25_trailing.get('washed_out_ratio', 0):.1%}")
    report.append(f"4. **最大持仓天数**: 15 天限制生效，防止僵尸股")
    report.append("")
    
    report.append("### 7.3 后续优化方向")
    report.append("")
    report.append("1. 继续优化领头羊识别阈值（当前 Top 3%）")
    report.append("2. 探索更优的 Inverted Regime Mapping 参数")
    report.append("3. 增加行业轮动因子")
    report.append("4. 探索信号锁定门槛的自适应调整机制")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append(f"**报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试版本**: V2.4 vs V2.5")
    
    return "\n".join(report)


if __name__ == "__main__":
    run_comparison()