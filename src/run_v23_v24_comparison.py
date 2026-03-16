"""
V2.3 vs V2.4 对比测试脚本 - Iteration 25

【测试目标】
1. 基准对比：对比 V1.9 (3.89% 收益) 与 V2.4 的表现
2. 审计指标：
   - 统计"信号锁定"触发次数
   - 统计 2024 年最大的一笔盈利交易，分析其持有期和退出逻辑
   - 如果盲测收益未能重回 2% 以上，分析是否因为 alpha 因子本身在 2024 年失效
"""

import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    from final_strategy_v2_3 import FinalStrategyV23, run_walk_forward_validation as wf_v23, generate_audit_report
    from final_strategy_v2_4 import FinalStrategyV24, run_walk_forward_validation as wf_v24, generate_audit_report as generate_audit_report_v24
    from db_manager import DatabaseManager
except ImportError:
    from src.final_strategy_v2_3 import FinalStrategyV23, run_walk_forward_validation as wf_v23, generate_audit_report
    from src.final_strategy_v2_4 import FinalStrategyV24, run_walk_forward_validation as wf_v24, generate_audit_report as generate_audit_report_v24
    from src.db_manager import DatabaseManager


def run_comparison_test():
    """运行 V2.3 vs V2.4 对比测试"""
    
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("Iteration 25: V2.3 vs V2.4 对比测试")
    logger.info("=" * 80)
    
    # 初始化数据库
    db = DatabaseManager.get_instance()
    
    # =====================
    # V2.3 测试
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【V2.3 基准测试】")
    logger.info("=" * 60)
    
    strategy_v23 = FinalStrategyV23(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
        db=db,
    )
    
    # 训练模型
    strategy_v23.train_model(train_end_date="2023-12-31", model_type="ensemble")
    
    # 运行 Walk-Forward 验证
    wf_result_v23 = wf_v23(strategy_v23)
    
    # 归因分析
    attribution_v23 = strategy_v23.run_attribution_analysis()
    
    # 压力测试
    stress_v23 = strategy_v23.run_stress_test(
        type('BacktestResult', (), {'total_return': wf_result_v23['blind_test_result'].get('total_return', 0)})(),
        noise_level=0.001,
    )
    
    # =====================
    # V2.4 测试
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【V2.4 进攻性 Alpha 觉醒测试】")
    logger.info("=" * 60)
    
    strategy_v24 = FinalStrategyV24(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
        db=db,
    )
    
    # 训练模型
    strategy_v24.train_model(train_end_date="2023-12-31", model_type="ensemble")
    
    # 运行 Walk-Forward 验证
    wf_result_v24 = wf_v24(strategy_v24)
    
    # 归因分析
    attribution_v24 = strategy_v24.run_attribution_analysis()
    
    # 压力测试
    stress_v24 = strategy_v24.run_stress_test(
        type('BacktestResult', (), {'total_return': wf_result_v24['blind_test_result'].get('total_return', 0)})(),
        noise_level=0.001,
    )
    
    # =====================
    # 生成对比报告
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【生成对比报告】")
    logger.info("=" * 60)
    
    report = generate_comparison_report(
        wf_result_v23, wf_result_v24,
        attribution_v23, attribution_v24,
        stress_v23, stress_v24,
        strategy_v24,
    )
    
    # 保存报告
    report_path = Path("reports/Iteration25_V23_V24_Comparison_Report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"对比报告已保存至：{report_path}")
    
    # =====================
    # 输出摘要
    # =====================
    logger.info("")
    logger.info("=" * 60)
    logger.info("【对比摘要】")
    logger.info("=" * 60)
    
    v23_return = wf_result_v23['blind_test_result'].get('total_return', 0)
    v24_return = wf_result_v24['blind_test_result'].get('total_return', 0)
    
    logger.info(f"V2.3 盲测收益 (2024): {v23_return:.2%}")
    logger.info(f"V2.4 盲测收益 (2024): {v24_return:.2%}")
    logger.info(f"收益改善：{(v24_return - v23_return):.2%}")
    
    if v24_return > 0.0389:
        logger.info("✅ V2.4 收益 > 3.89% (V1.9 基准), 目标达成!")
    elif v24_return > 0.03:
        logger.info("✅ V2.4 收益 > 3.0%, 接近目标")
    elif v24_return > 0.02:
        logger.info("✅ V2.4 收益 > 2%, 达到最低目标")
    elif v24_return > 0:
        logger.info("✅ V2.4 收益已转正")
    else:
        logger.info("⚠️ V2.4 收益仍为负，需继续优化")
    
    logger.info("")
    logger.info(f"V2.4 信号锁定触发：{wf_result_v24.get('profit_relax_stats', {}).get('signal_locked', 0)}次")
    logger.info(f"V2.4 L1 放松触发：{wf_result_v24.get('profit_relax_stats', {}).get('l1_triggered', 0)}次")
    logger.info(f"V2.4 L2 放松触发：{wf_result_v24.get('profit_relax_stats', {}).get('l2_triggered', 0)}次")
    
    return wf_result_v23, wf_result_v24


def generate_comparison_report(
    wf_result_v23: dict, wf_result_v24: dict,
    attribution_v23: dict, attribution_v24: dict,
    stress_v23: dict, stress_v24: dict,
    strategy_v24,
) -> str:
    """生成 V2.3 vs V2.4 对比报告"""
    
    report = []
    report.append("# Iteration 25: V2.3 vs V2.4 对比测试报告")
    report.append("")
    report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试版本**: V2.3 (基准) vs V2.4 (进攻性 Alpha 觉醒)")
    report.append("")
    
    report.append("## 一、执行摘要")
    report.append("")
    report.append("### 1.1 V2.4 核心改进")
    report.append("")
    report.append("1. **信号锁定门槛下调**: 从 5% 降至 2.5%，让盈利单更快切换到 LGB 主导")
    report.append("2. **非线性权重映射 (Sigmoid-like)**: 低波动时 LGB=80%, 高波动时线性滑动")
    report.append("3. **追踪止盈深度放宽**: 基础因子 1.5，盈利>5% 时 2.0x + buffer 额外 0.5%")
    report.append("4. **因子层进化**: 新增 momentum_squeeze 因子，优化 relative_strength_sector")
    report.append("")
    
    report.append("### 1.2 基准对比目标")
    report.append("")
    report.append("- **V1.9 基准收益**: 3.89% (2024 年盲测)")
    report.append("- **V2.4 目标收益**: > 3.89%")
    report.append("- **最低目标**: > 2%")
    report.append("")
    
    report.append("## 二、性能对比")
    report.append("")
    
    vf_v23 = wf_result_v23.get("validation_result", {})
    bt_v23 = wf_result_v23.get("blind_test_result", {})
    vf_v24 = wf_result_v24.get("validation_result", {})
    bt_v24 = wf_result_v24.get("blind_test_result", {})
    
    report.append("### 2.1 验证集表现对比 (2023 年)")
    report.append("")
    report.append("| 指标 | V2.3 | V2.4 | 改善 |")
    report.append("|------|------|------|------|")
    report.append(f"| 总收益率 | {vf_v23.get('total_return', 0):.2%} | {vf_v24.get('total_return', 0):.2%} | {(vf_v24.get('total_return', 0) - vf_v23.get('total_return', 0)):.2%} |")
    report.append(f"| 年化收益 | {vf_v23.get('annual_return', 0):.2%} | {vf_v24.get('annual_return', 0):.2%} | {(vf_v24.get('annual_return', 0) - vf_v23.get('annual_return', 0)):.2%} |")
    report.append(f"| 最大回撤 | {vf_v23.get('max_drawdown', 0):.2%} | {vf_v24.get('max_drawdown', 0):.2%} | {(vf_v24.get('max_drawdown', 0) - vf_v23.get('max_drawdown', 0)):.2%} |")
    report.append(f"| 夏普比率 | {vf_v23.get('sharpe_ratio', 0):.2f} | {vf_v24.get('sharpe_ratio', 0):.2f} | {(vf_v24.get('sharpe_ratio', 0) - vf_v23.get('sharpe_ratio', 0)):.2f} |")
    report.append(f"| 胜率 | {vf_v23.get('win_rate', 0):.2%} | {vf_v24.get('win_rate', 0):.2%} | {(vf_v24.get('win_rate', 0) - vf_v23.get('win_rate', 0)):.2%} |")
    report.append(f"| 交易次数 | {vf_v23.get('total_trades', 0)} | {vf_v24.get('total_trades', 0)} | {vf_v24.get('total_trades', 0) - vf_v23.get('total_trades', 0)} |")
    report.append("")
    
    report.append("### 2.2 盲测集表现对比 (2024 年)")
    report.append("")
    report.append("| 指标 | V2.3 | V2.4 | 改善 |")
    report.append("|------|------|------|------|")
    report.append(f"| 总收益率 | {bt_v23.get('total_return', 0):.2%} | {bt_v24.get('total_return', 0):.2%} | {(bt_v24.get('total_return', 0) - bt_v23.get('total_return', 0)):.2%} |")
    report.append(f"| 年化收益 | {bt_v23.get('annual_return', 0):.2%} | {bt_v24.get('annual_return', 0):.2%} | {(bt_v24.get('annual_return', 0) - bt_v23.get('annual_return', 0)):.2%} |")
    report.append(f"| 最大回撤 | {bt_v23.get('max_drawdown', 0):.2%} | {bt_v24.get('max_drawdown', 0):.2%} | {(bt_v24.get('max_drawdown', 0) - bt_v23.get('max_drawdown', 0)):.2%} |")
    report.append(f"| 夏普比率 | {bt_v23.get('sharpe_ratio', 0):.2f} | {bt_v24.get('sharpe_ratio', 0):.2f} | {(bt_v24.get('sharpe_ratio', 0) - bt_v23.get('sharpe_ratio', 0)):.2f} |")
    report.append(f"| 胜率 | {bt_v23.get('win_rate', 0):.2%} | {bt_v24.get('win_rate', 0):.2%} | {(bt_v24.get('win_rate', 0) - bt_v23.get('win_rate', 0)):.2%} |")
    report.append(f"| 交易次数 | {bt_v23.get('total_trades', 0)} | {bt_v24.get('total_trades', 0)} | {bt_v24.get('total_trades', 0) - bt_v23.get('total_trades', 0)} |")
    report.append("")
    
    report.append("### 2.3 收益对比分析")
    report.append("")
    
    # V1.9 基准对比
    if bt_v24.get('total_return', 0) > 0.0389:
        report.append("✅ **V2.4 盲测集收益 > 3.89% (V1.9 基准), 目标达成**")
    elif bt_v24.get('total_return', 0) > 0.03:
        report.append("✅ **V2.4 盲测集收益 > 3.0%，接近目标**")
    elif bt_v24.get('total_return', 0) > 0.02:
        report.append("✅ **V2.4 盲测集收益 > 2%，达到最低目标**")
    elif bt_v24.get('total_return', 0) > 0:
        report.append("✅ **V2.4 盲测集收益已转正**")
    else:
        report.append("⚠️ **V2.4 盲测集收益仍为负，需继续优化**")
    
    report.append("")
    
    # 计算相对于 V1.9 的表现
    v19_benchmark = 0.0389
    v24_vs_v19 = bt_v24.get('total_return', 0) - v19_benchmark
    v23_vs_v19 = bt_v23.get('total_return', 0) - v19_benchmark
    
    report.append("| 基准对比 | 数值 |")
    report.append("|----------|------|")
    report.append(f"| V1.9 基准 | {v19_benchmark:.2%} |")
    report.append(f"| V2.3 vs V1.9 | {v23_vs_v19:.2%} |")
    report.append(f"| V2.4 vs V1.9 | {v24_vs_v19:.2%} |")
    report.append(f"| V2.4 vs V2.3 改善 | {(bt_v24.get('total_return', 0) - bt_v23.get('total_return', 0)):.2%} |")
    report.append("")
    
    report.append("## 三、V2.4 新增指标分析")
    report.append("")
    
    weight_trend = wf_result_v24.get("weight_trend", {})
    profit_relax_stats = wf_result_v24.get("profit_relax_stats", {})
    
    report.append("### 3.1 动态权重趋势分析")
    report.append("")
    overall_avg = weight_trend.get("overall_avg", {})
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| 平均 Ridge 权重 | {overall_avg.get('ridge_weight', 0):.2%} |")
    report.append(f"| 平均 LGB 权重 | {overall_avg.get('lgb_weight', 0):.2%} |")
    report.append(f"| 平均 ATR 排名 | {overall_avg.get('atr_rank', 0):.2%} |")
    report.append(f"| 平均 ATR | {overall_avg.get('atr', 0):.4f} |")
    report.append(f"| 总交易日数 | {weight_trend.get('total_days', 0)} |")
    report.append(f"| 低波动天数 | {weight_trend.get('low_vol_days', 0)} |")
    report.append(f"| 高波动天数 | {weight_trend.get('high_vol_days', 0)} |")
    report.append("")
    
    report.append("### 3.2 利润跑腾统计")
    report.append("")
    report.append("| 指标 | 数值 |")
    report.append("|------|------|")
    report.append(f"| L1 放松触发次数 (盈利>3%) | {profit_relax_stats.get('l1_triggered', 0)} |")
    report.append(f"| L2 放松触发次数 (盈利>5%) | {profit_relax_stats.get('l2_triggered', 0)} |")
    report.append(f"| 信号锁定触发次数 (盈利>2.5%) | {profit_relax_stats.get('signal_locked', 0)} |")
    report.append("")
    
    report.append("### 3.3 信号锁定效果分析")
    report.append("")
    
    signal_locked_count = profit_relax_stats.get('signal_locked', 0)
    total_trades_v24 = bt_v24.get('total_trades', 0)
    signal_lock_ratio = signal_locked_count / total_trades_v24 if total_trades_v24 > 0 else 0
    
    report.append(f"- **信号锁定触发比例**: {signal_lock_ratio:.1%}")
    report.append(f"- **信号锁定门槛**: 2.5% (V2.4) vs 5% (V2.3)")
    report.append("")
    
    if signal_lock_ratio > 0.3:
        report.append("✅ 信号锁定机制有效，超过 30% 的交易触发了 100% LGB 切换")
    elif signal_lock_ratio > 0.1:
        report.append("⚠️ 信号锁定触发比例较低，可能需要进一步降低门槛")
    else:
        report.append("⚠️ 信号锁定触发比例很低，建议检查盈利单质量")
    
    report.append("")
    
    report.append("## 四、归因分析对比")
    report.append("")
    
    loss_v23 = attribution_v23.get("loss_reasons", {})
    loss_v24 = attribution_v24.get("loss_reasons", {})
    trailing_v23 = attribution_v23.get("trailing_stop_stats", {})
    trailing_v24 = attribution_v24.get("trailing_stop_stats", {})
    
    report.append("### 4.1 亏损交易原因对比")
    report.append("")
    report.append("| 原因 | V2.3 次数 | V2.4 次数 | 变化 |")
    report.append("|------|----------|----------|------|")
    
    all_reasons = set(loss_v23.keys()) | set(loss_v24.keys())
    for reason in sorted(all_reasons):
        v23_count = loss_v23.get(reason, 0)
        v24_count = loss_v24.get(reason, 0)
        change = v24_count - v23_count
        change_str = f"+{change}" if change > 0 else str(change)
        report.append(f"| {reason} | {v23_count} | {v24_count} | {change_str} |")
    report.append("")
    
    report.append("### 4.2 追踪止盈触发对比")
    report.append("")
    report.append("| 指标 | V2.3 | V2.4 | 变化 |")
    report.append("|------|------|------|------|")
    report.append(f"| 触发次数 | {trailing_v23.get('count', 0)} | {trailing_v24.get('count', 0)} | {trailing_v24.get('count', 0) - trailing_v23.get('count', 0)} |")
    report.append(f"| 平均盈利 | {trailing_v23.get('avg_profit', 0):.2%} | {trailing_v24.get('avg_profit', 0):.2%} | {(trailing_v24.get('avg_profit', 0) - trailing_v23.get('avg_profit', 0)):.2%} |")
    report.append(f"| 最大盈利 | {trailing_v23.get('max_profit', 0):.2%} | {trailing_v24.get('max_profit', 0):.2%} | {(trailing_v24.get('max_profit', 0) - trailing_v23.get('max_profit', 0)):.2%} |")
    report.append("")
    
    report.append("### 4.3 2024 年最大盈利交易分析")
    report.append("")
    
    # 分析最大盈利交易
    if hasattr(strategy_v24, 'trade_records') and strategy_v24.trade_records:
        max_profit_trade = max(strategy_v24.trade_records, key=lambda t: t.pnl_pct)
        report.append(f"- **标的**: {max_profit_trade.symbol}")
        report.append(f"- **买入日期**: {max_profit_trade.entry_date}")
        report.append(f"- **卖出日期**: {max_profit_trade.exit_date}")
        report.append(f"- **盈利**: {max_profit_trade.pnl_pct:.2%}")
        report.append(f"- **持有期**: {max_profit_trade.hold_days} 天")
        report.append(f"- **退出原因**: {max_profit_trade.exit_reason}")
        report.append(f"- **持仓期间最大盈利**: {max_profit_trade.max_profit_pct:.2%}")
        report.append(f"- **触发放松级别**: L{max_profit_trade.relax_level_triggered}")
    else:
        report.append("⚠️ 无法获取交易记录详情")
    
    report.append("")
    
    report.append("## 五、压力测试对比")
    report.append("")
    report.append("| 指标 | V2.3 | V2.4 |")
    report.append("|------|------|------|")
    report.append(f"| 原始收益 | {stress_v23.get('original_return', 0):.2%} | {stress_v24.get('original_return', 0):.2%} |")
    report.append(f"| 扰动后收益 | {stress_v23.get('stressed_return', 0):.2%} | {stress_v24.get('stressed_return', 0):.2%} |")
    report.append(f"| 收益回落 | {stress_v23.get('return_drop', 0):.2%} | {stress_v24.get('return_drop', 0):.2%} |")
    report.append(f"| 噪声敏感度 | {stress_v23.get('noise_sensitivity', 'Unknown')} | {stress_v24.get('noise_sensitivity', 'Unknown')} |")
    report.append("")
    
    report.append("## 六、Alpha 因子有效性分析")
    report.append("")
    
    # 分析盲测收益是否达到 2%
    if bt_v24.get('total_return', 0) < 0.02:
        report.append("### 6.1 盲测收益未达 2% 原因分析")
        report.append("")
        report.append("如果 V2.4 盲测收益未能重回 2% 以上，可能原因包括:")
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
    else:
        report.append("✅ **V2.4 盲测收益达到 2% 以上，Alpha 因子在 2024 年仍然有效**")
        report.append("")
        report.append("因子有效性验证:")
        report.append("")
        report.append(f"- V2.4 盲测收益：{bt_v24.get('total_return', 0):.2%} > 2%")
        report.append(f"- 新增 momentum_squeeze 因子捕捉低波动上涨标的")
        report.append(f"- 优化 relative_strength_sector 因子，增加 Top 10% 偏好权重")
    
    report.append("")
    
    report.append("## 七、结论与建议")
    report.append("")
    report.append("### 7.1 核心结论")
    report.append("")
    
    improvement = bt_v24.get('total_return', 0) - bt_v23.get('total_return', 0)
    
    if improvement > 0.01:
        report.append(f"✅ **V2.4 相对 V2.3 改善显著**: +{improvement:.2%}")
    elif improvement > 0:
        report.append(f"✅ **V2.4 相对 V2.3 有所改善**: +{improvement:.2%}")
    else:
        report.append(f"⚠️ **V2.4 相对 V2.3 有所下降**: {improvement:.2%}")
    
    report.append("")
    report.append("### 7.2 关键改进效果")
    report.append("")
    report.append(f"1. **信号锁定门槛下调**: 触发{profit_relax_stats.get('signal_locked', 0)}次，占比{signal_lock_ratio:.1%}")
    report.append(f"2. **非线性权重映射**: 低波动天数={weight_trend.get('low_vol_days', 0)}, 平均 LGB 权重={overall_avg.get('lgb_weight', 0):.1%}")
    report.append(f"3. **追踪止盈深度放宽**: L1 触发={profit_relax_stats.get('l1_triggered', 0)}次，L2 触发={profit_relax_stats.get('l2_triggered', 0)}次")
    report.append(f"4. **新增因子**: momentum_squeeze 已纳入模型")
    report.append("")
    
    report.append("### 7.3 后续优化方向")
    report.append("")
    report.append("1. 继续优化动态权重映射参数，探索更优的 Sigmoid 形状")
    report.append("2. 增加更多行业轮动因子，提升相对强度因子效果")
    report.append("3. 探索信号锁定门槛的自适应调整机制")
    report.append("")
    
    report.append("---")
    report.append("")
    report.append(f"**报告生成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**测试版本**: V2.3 vs V2.4")
    
    return "\n".join(report)


if __name__ == "__main__":
    run_comparison_test()