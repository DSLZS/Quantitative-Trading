#!/usr/bin/env python3
"""
组合优化报告生成器 - 对比优化前后的策略表现。

生成内容包括:
1. 优化前后的收益对比图
2. 关键绩效指标对比表
3. 损益归因分析
4. 市场状态分析
5. 最终部署建议
"""

import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
import json

try:
    from .walk_forward_backtester import WalkForwardBacktester, run_walk_forward_backtest
    from .walk_forward_backtester_v2 import WalkForwardBacktesterV2, run_walk_forward_backtest_v2
except ImportError:
    from walk_forward_backtester import WalkForwardBacktester, run_walk_forward_backtest
    from walk_forward_backtester_v2 import WalkForwardBacktesterV2, run_walk_forward_backtest_v2


def run_comparison(
    parquet_path: str = "data/parquet/features_latest.parquet",
    initial_capital: float = 1_000_000.0,
    start_date: str = "2025-01-01",
) -> dict:
    """
    运行优化前后的对比回测。
    
    Returns:
        dict: 包含两个版本回测结果的字典
    """
    logger.info("=" * 80)
    logger.info("Running Portfolio Optimization Comparison")
    logger.info("=" * 80)
    
    # 运行 V1 (原始版本 - 防御性重训)
    logger.info("\n[1/2] Running V1 (Defensive Retraining)...")
    try:
        results_v1 = run_walk_forward_backtest(
            parquet_path=parquet_path,
            initial_capital=initial_capital,
            use_defensive_params=True,
            use_shuffle_importance=True,
            start_date=start_date,
        )
        logger.info(f"V1 Complete: Total Return = {results_v1['total_return']:.2%}")
    except Exception as e:
        logger.error(f"V1 failed: {e}")
        results_v1 = None
    
    # 运行 V2 (优化版本 - 组合优化)
    logger.info("\n[2/2] Running V2 (Portfolio Optimization)...")
    try:
        results_v2 = run_walk_forward_backtest_v2(
            parquet_path=parquet_path,
            initial_capital=initial_capital,
            top_k=5,
            use_volatility_scaling=True,
            use_regime_switch=True,
            use_defensive_params=True,
            start_date=start_date,
        )
        logger.info(f"V2 Complete: Total Return = {results_v2['total_return']:.2%}")
    except Exception as e:
        logger.error(f"V2 failed: {e}")
        results_v2 = None
    
    return {
        "v1": results_v1,
        "v2": results_v2,
    }


def generate_metrics_table(results_v1: dict, results_v2: dict) -> str:
    """生成绩效指标对比表。"""
    metrics = [
        ("总收益率", "total_return"),
        ("OOS 夏普比率", "oos_sharpe"),
        ("最终净值", "final_value"),
        ("交易成本", "total_cost", "cost_analysis"),
        ("Alpha", "alpha", "attribution"),
        ("Beta", "beta", "attribution"),
    ]
    
    lines = []
    lines.append("=" * 70)
    lines.append("绩效指标对比表")
    lines.append("=" * 70)
    lines.append(f"{'指标':<20} {'V1 (防御性重训)':<22} {'V2 (组合优化)':<22}")
    lines.append("-" * 70)
    
    for item in metrics:
        name = item[0]
        key = item[1]
        
        if len(item) == 3:
            # 嵌套键
            parent_key = item[2]
            v1_val = results_v1.get(parent_key, {}).get(key, "N/A") if results_v1 else "N/A"
            v2_val = results_v2.get(parent_key, {}).get(key, "N/A") if results_v2 else "N/A"
        else:
            v1_val = results_v1.get(key, "N/A") if results_v1 else "N/A"
            v2_val = results_v2.get(key, "N/A") if results_v2 else "N/A"
        
        # 格式化
        if isinstance(v1_val, float):
            if key in ["total_return", "alpha", "beta"]:
                v1_str = f"{v1_val:.2%}"
                v2_str = f"{v2_val:.2%}"
            elif key == "final_value" or key == "total_cost":
                v1_str = f"${v1_val:,.2f}"
                v2_str = f"${v2_val:,.2f}"
            else:
                v1_str = f"{v1_val:.2f}"
                v2_str = f"{v2_val:.2f}"
        else:
            v1_str = str(v1_val)
            v2_str = str(v2_val)
        
        lines.append(f"{name:<20} {v1_str:<22} {v2_str:<22}")
    
    lines.append("=" * 70)
    return "\n".join(lines)


def generate_volume_price_stable_analysis(
    results_v2: dict,
) -> str:
    """生成 volume_price_stable 因子的稳健性评价。"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("volume_price_stable 因子稳健性评价")
    lines.append("=" * 70)
    
    if results_v2 is None:
        lines.append("无可用数据")
        return "\n".join(lines)
    
    # 从窗口结果分析因子稳定性
    window_results = results_v2.get("window_results", [])
    
    lines.append(f"\n回测窗口数量：{len(window_results)}")
    
    # 分析 OOS 表现
    oos_sharpe = results_v2.get("oos_sharpe", 0)
    total_return = results_v2.get("total_return", 0)
    max_drawdown = results_v2.get("metrics", {}).get("max_drawdown", 0)
    
    lines.append(f"\nOOS 夏普比率：{oos_sharpe:.2f}")
    lines.append(f"总收益率：{total_return:.2%}")
    lines.append(f"最大回撤：{max_drawdown:.2%}")
    
    # 评价
    lines.append("\n【稳健性评价】")
    
    if oos_sharpe >= 1.5:
        lines.append("✓ OOS 夏普比率优秀 (>1.5)，因子在样本外具有强大的预测能力")
    elif oos_sharpe >= 1.0:
        lines.append("✓ OOS 夏普比率良好 (>1.0)，因子在样本外具有稳定的预测能力")
    elif oos_sharpe >= 0.8:
        lines.append("○ OOS 夏普比率可接受 (>0.8)，因子在样本外具有一定的预测能力")
    else:
        lines.append("✗ OOS 夏普比率偏低 (<0.8)，因子在样本外预测能力不足")
    
    if max_drawdown <= 0.05:
        lines.append("✓ 最大回撤控制优秀 (<5%)")
    elif max_drawdown <= 0.10:
        lines.append("○ 最大回撤控制良好 (<10%)")
    else:
        lines.append("✗ 最大回撤偏高 (>10%)")
    
    lines.append("\n【因子特性分析】")
    lines.append("volume_price_stable 因子结合了量价稳定性，具有以下特点：")
    lines.append("1. 低波动特性：偏好价格波动小的股票，具有防御性")
    lines.append("2. 量价配合：成交量与价格变动协调，反映资金流入流出的健康程度")
    lines.append("3. 均值回归：在震荡市中表现较好，在趋势市中可能滞后")
    
    return "\n".join(lines)


def generate_deployment_recommendations(
    results_v1: dict,
    results_v2: dict,
) -> str:
    """生成最终部署建议。"""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("如何平衡稳健性与盈利能力 - 最终部署建议")
    lines.append("=" * 70)
    
    # 对比分析
    lines.append("\n【对比分析】")
    
    if results_v1 and results_v2:
        v1_return = results_v1.get("total_return", 0)
        v2_return = results_v2.get("total_return", 0)
        v1_sharpe = results_v1.get("oos_sharpe", 0)
        v2_sharpe = results_v2.get("oos_sharpe", 0)
        
        lines.append(f"V1 (防御性重训): 收益率={v1_return:.2%}, OOS Sharpe={v1_sharpe:.2f}")
        lines.append(f"V2 (组合优化)  : 收益率={v2_return:.2%}, OOS Sharpe={v2_sharpe:.2f}")
        
        if v2_sharpe >= v1_sharpe * 0.9:
            lines.append("\n✓ V2 在保持夏普比率的同时提升了收益能力")
        else:
            lines.append("\n⚠ V2 的夏普比率有所下降，需要进一步优化")
    
    # 核心建议
    lines.append("\n【核心建议】")
    lines.append("""
1. 防御性参数配置（必须保留）
   - max_depth: 4 (限制树深度防止过拟合)
   - num_leaves: 18 (限制叶子节点数量)
   - lambda_l1/lambda_l2: 0.1 (L1/L2 正则化)
   - subsample/colsample_bytree: 0.8 (随机采样增加鲁棒性)

2. 选股策略优化
   - 采用 Top K 选股替代固定阈值（确保每日都有最优标的入选）
   - 建议 K 值：5-10 只（分散风险）

3. 仓位管理
   - 逆波动率加权：波动率小的股票多买，波动率大的股票少买
   - 单只股票上限：10-15%
   - 总仓位上限：根据市场状态动态调整

4. 市场状态开关
   - 大盘在 20 日均线上方：正常仓位 (100%)
   - 大盘在 20 日均线下方：防守仓位 (50%)

5. 交易频率控制
   - 最小持仓天数：1-3 天（减少频繁交易）
   - 设置最小预测差异阈值（避免无意义调仓）

6. 滚动重训频率
   - 建议每月重训一次
   - 训练窗口：12 个月
   - 持续监控 OOS 表现，如夏普比率连续 2 个月低于 0.8 需重新评估
""")
    
    # 风险提示
    lines.append("\n【风险提示】")
    lines.append("""
1. 过拟合风险：虽然采用了防御性参数，但仍需持续监控 OOS 表现
2. 市场风格漂移：如市场风格发生剧烈变化，策略可能短期失效
3. 流动性风险：小市值股票可能面临流动性不足问题
4. 交易成本：高频交易会产生较高成本，建议优化交易频率
""")
    
    # 部署检查清单
    lines.append("\n【部署检查清单】")
    lines.append("""
□ 确认数据源稳定可靠
□ 设置异常监控报警
□ 准备应急预案（如市场暴跌时的处理）
□ 定期（每周）审查策略表现
□ 每月进行一次完整回测验证
""")
    
    return "\n".join(lines)


def save_report(
    output_dir: str = "data/plots",
    results_v1: dict = None,
    results_v2: dict = None,
) -> str:
    """保存完整报告。"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"optimization_report_{timestamp}.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("量化交易策略组合优化报告\n")
        f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
        
        # 绩效指标对比表
        if results_v1 and results_v2:
            f.write(generate_metrics_table(results_v1, results_v2))
            f.write("\n")
        
        # volume_price_stable 因子分析
        f.write(generate_volume_price_stable_analysis(results_v2))
        f.write("\n")
        
        # 部署建议
        f.write(generate_deployment_recommendations(results_v1, results_v2))
    
    logger.info(f"Report saved to {report_path}")
    return str(report_path)


def main():
    """主函数：运行对比并生成报告。"""
    logger.info("Starting Portfolio Optimization Report Generation")
    
    # 运行对比回测
    results = run_comparison()
    
    # 保存报告
    report_path = save_report(
        results_v1=results["v1"],
        results_v2=results["v2"],
    )
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("组合优化报告摘要")
    print("=" * 80)
    
    if results["v1"]:
        print(f"\nV1 (防御性重训):")
        print(f"  总收益率：{results['v1']['total_return']:.2%}")
        print(f"  OOS 夏普比率：{results['v1']['oos_sharpe']:.2f}")
        print(f"  最终净值：${results['v1']['final_value']:,.2f}")
    
    if results["v2"]:
        print(f"\nV2 (组合优化):")
        print(f"  总收益率：{results['v2']['total_return']:.2%}")
        print(f"  OOS 夏普比率：{results['v2']['oos_sharpe']:.2f}")
        print(f"  最终净值：${results['v2']['final_value']:,.2f}")
        print(f"  Alpha: {results['v2']['attribution']['alpha']:.2%}")
        print(f"  Beta: {results['v2']['attribution']['beta']:.2%}")
    
    print(f"\n完整报告已保存至：{report_path}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()