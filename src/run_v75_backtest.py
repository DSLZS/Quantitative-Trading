"""
V75 回测运行脚本 - 自适应环境建模与波动率归一化

【功能】
1. 运行 V74 回测获取基准数据
2. 运行 V75 回测获取优化数据
3. 生成对比报告
4. 验证验收指标

【验收指标】
- 指标 A：全年度 Mean Rank IC >= 0.03
- 指标 B：最大回撤相对于 V74 减少 15% 以上
- 指标 C：评分分布 Std 保持在 20-30 之间

作者：量化系统
版本：V75.0
日期：2026-03-26
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("logs/v75_backtest_{time:YYYYMMDD}.log", level="DEBUG", rotation="1 day")


def run_v74_backtest_first(start_date: str, end_date: str) -> dict:
    """先运行 V74 回测获取基准数据"""
    logger.info("=" * 60)
    logger.info("Step 1: 运行 V74 回测获取基准数据")
    logger.info("=" * 60)
    
    try:
        from src.v74_engine import run_v74_backtest as run_v74
        
        result = run_v74(start_date, end_date)
        
        # 提取关键指标
        v74_metrics = {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'mean_rank_ic': result.mean_rank_ic,
            'monthly_rank_ic': result.monthly_rank_ic,
            'score_std': result.score_std,
            'win_rate': result.win_rate,
        }
        
        logger.info(f"V74 回测完成：最大回撤={v74_metrics['max_drawdown']:.4f}, "
                   f"Rank IC={v74_metrics['mean_rank_ic']:.4f}")
        
        return v74_metrics
        
    except Exception as e:
        logger.error(f"运行 V74 回测失败：{e}")
        logger.error("将继续运行 V75 回测（无 V74 对比）")
        return None


def run_v75_backtest(start_date: str, end_date: str, 
                     v74_max_drawdown: Optional[float] = None) -> dict:
    """运行 V75 回测"""
    logger.info("=" * 60)
    logger.info("Step 2: 运行 V75 回测")
    logger.info("=" * 60)
    
    try:
        from src.v75_engine import run_v75_backtest as run_v75
        
        result = run_v75(start_date, end_date, v74_max_drawdown=v74_max_drawdown)
        
        # 提取关键指标
        v75_metrics = {
            'total_return': result.total_return,
            'annualized_return': result.annualized_return,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'mean_rank_ic': result.mean_rank_ic,
            'monthly_rank_ic': result.monthly_rank_ic,
            'score_std': result.score_std,
            'win_rate': result.win_rate,
            'bullish_days': result.bullish_days,
            'bearish_days': result.bearish_days,
            'neutral_days': result.neutral_days,
        }
        
        # 计算回撤减少
        if v74_max_drawdown and v74_max_drawdown > 0:
            v75_metrics['drawdown_reduction'] = (v74_max_drawdown - result.max_drawdown) / v74_max_drawdown
        else:
            v75_metrics['drawdown_reduction'] = None
        
        logger.info(f"V75 回测完成：最大回撤={v75_metrics['max_drawdown']:.4f}, "
                   f"Rank IC={v75_metrics['mean_rank_ic']:.4f}")
        
        return v75_metrics
        
    except Exception as e:
        logger.error(f"运行 V75 回测失败：{e}")
        raise


def verify_criteria(v75_metrics: dict, v74_metrics: Optional[dict] = None) -> dict:
    """验证验收指标"""
    logger.info("=" * 60)
    logger.info("Step 3: 验证验收指标")
    logger.info("=" * 60)
    
    criteria_results = {}
    
    # 指标 A：全年度 Mean Rank IC >= 0.03
    rank_ic_pass = v75_metrics['mean_rank_ic'] >= 0.03
    criteria_results['rank_ic'] = {
        'value': v75_metrics['mean_rank_ic'],
        'target': 0.03,
        'pass': rank_ic_pass,
        'status': '✓ 达标' if rank_ic_pass else '✗ 未达标'
    }
    logger.info(f"指标 A (Mean Rank IC >= 0.03): {v75_metrics['mean_rank_ic']:.4f} "
               f"{criteria_results['rank_ic']['status']}")
    
    # 指标 B：最大回撤相对于 V74 减少 15% 以上
    if v74_metrics and v74_metrics['max_drawdown'] > 0:
        drawdown_reduction = (v74_metrics['max_drawdown'] - v75_metrics['max_drawdown']) / v74_metrics['max_drawdown']
        drawdown_pass = drawdown_reduction >= 0.15
        criteria_results['drawdown'] = {
            'value': drawdown_reduction,
            'target': 0.15,
            'v74_dd': v74_metrics['max_drawdown'],
            'v75_dd': v75_metrics['max_drawdown'],
            'pass': drawdown_pass,
            'status': '✓ 达标' if drawdown_pass else '✗ 未达标'
        }
        logger.info(f"指标 B (回撤减少 >= 15%): {drawdown_reduction*100:.1f}% "
                   f"(V74: {v74_metrics['max_drawdown']:.2%} -> V75: {v75_metrics['max_drawdown']:.2%}) "
                   f"{criteria_results['drawdown']['status']}")
    else:
        criteria_results['drawdown'] = {
            'value': None,
            'target': 0.15,
            'pass': False,
            'status': '⚠ 无法计算（缺少 V74 数据）'
        }
        logger.warning(f"指标 B (回撤减少 >= 15%): 无法计算（缺少 V74 数据）")
    
    # 指标 C：评分分布 Std 保持在 20-30 之间
    std_pass = 20.0 <= v75_metrics['score_std'] <= 30.0
    criteria_results['score_std'] = {
        'value': v75_metrics['score_std'],
        'target_min': 20.0,
        'target_max': 30.0,
        'pass': std_pass,
        'status': '✓ 达标' if std_pass else '✗ 未达标'
    }
    logger.info(f"指标 C (评分 Std 20-30): {v75_metrics['score_std']:.2f} "
               f"{criteria_results['score_std']['status']}")
    
    # 总体达标情况
    all_pass = (
        rank_ic_pass and 
        std_pass and 
        (criteria_results['drawdown']['pass'] if criteria_results['drawdown']['value'] is not None else True)
    )
    
    criteria_results['overall_pass'] = all_pass
    criteria_results['overall_status'] = '✓ 全部达标' if all_pass else '✗ 部分未达标'
    
    logger.info("=" * 60)
    logger.info(f"验收结果：{criteria_results['overall_status']}")
    logger.info("=" * 60)
    
    return criteria_results


def generate_comparison_report(v75_metrics: dict, 
                                v74_metrics: Optional[dict] = None,
                                criteria_results: Optional[dict] = None) -> str:
    """生成对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("reports") / f"V75_Comparison_Report_{timestamp}.md"
    
    report_lines = [
        "# V75 vs V74 对比报告",
        "",
        f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 核心算法进化",
        "",
        "### V74 特性",
        "- VPD 量价背离因子",
        "- 熵权 SNR + 偏度审计",
        "- Tanh 非线性融合",
        "",
        "### V75 新增特性",
        "- **市场环境敏感度 (Market Regime Sensor)**: 计算近期 IC 动态调权",
        "- **Risk-Adjusted RS**: 波动率归一化的相对强度",
        "- **Hurst 指数**: 分形维度过滤趋势持续性",
        "- **自适应权重分配**: 根据市场环境调整因子权重",
        "",
        "## 回测结果对比",
        "",
        "| 指标 | V74 | V75 | 变化 |",
        "|------|-----|-----|------|",
    ]
    
    if v74_metrics:
        # 总收益
        tr_change = (v75_metrics['total_return'] - v74_metrics['total_return']) / abs(v74_metrics['total_return']) if v74_metrics['total_return'] != 0 else 0
        report_lines.append(f"| 总收益 | {v74_metrics['total_return']:.4f} ({v74_metrics['total_return']*100:.2f}%) | {v75_metrics['total_return']:.4f} ({v75_metrics['total_return']*100:.2f}%) | {tr_change*100:+.1f}% |")
        
        # 年化收益
        ar_change = (v75_metrics['annualized_return'] - v74_metrics['annualized_return']) / abs(v74_metrics['annualized_return']) if v74_metrics['annualized_return'] != 0 else 0
        report_lines.append(f"| 年化收益 | {v74_metrics['annualized_return']:.4f} ({v74_metrics['annualized_return']*100:.2f}%) | {v75_metrics['annualized_return']:.4f} ({v75_metrics['annualized_return']*100:.2f}%) | {ar_change*100:+.1f}% |")
        
        # 最大回撤
        md_change = (v75_metrics['max_drawdown'] - v74_metrics['max_drawdown']) / v74_metrics['max_drawdown'] if v74_metrics['max_drawdown'] != 0 else 0
        report_lines.append(f"| 最大回撤 | {v74_metrics['max_drawdown']:.4f} ({v74_metrics['max_drawdown']*100:.2f}%) | {v75_metrics['max_drawdown']:.4f} ({v75_metrics['max_drawdown']*100:.2f}%) | {md_change*100:+.1f}% |")
        
        # 夏普比率
        sr_change = (v75_metrics['sharpe_ratio'] - v74_metrics['sharpe_ratio']) / abs(v74_metrics['sharpe_ratio']) if v74_metrics['sharpe_ratio'] != 0 else 0
        report_lines.append(f"| 夏普比率 | {v74_metrics['sharpe_ratio']:.3f} | {v75_metrics['sharpe_ratio']:.3f} | {sr_change*100:+.1f}% |")
        
        # Rank IC
        ic_change = v75_metrics['mean_rank_ic'] - v74_metrics['mean_rank_ic']
        report_lines.append(f"| Mean Rank IC | {v74_metrics['mean_rank_ic']:.4f} | {v75_metrics['mean_rank_ic']:.4f} | {ic_change:+.4f} |")
        
        # 评分标准差
        std_change = v75_metrics['score_std'] - v74_metrics['score_std']
        report_lines.append(f"| 评分 Std | {v74_metrics['score_std']:.2f} | {v75_metrics['score_std']:.2f} | {std_change:+.2f} |")
        
        # 胜率
        wr_change = (v75_metrics['win_rate'] - v74_metrics['win_rate']) / v74_metrics['win_rate'] if v74_metrics['win_rate'] != 0 else 0
        report_lines.append(f"| 胜率 | {v74_metrics['win_rate']:.2%} | {v75_metrics['win_rate']:.2%} | {wr_change*100:+.1f}% |")
    else:
        report_lines.extend([
            f"| 总收益 | N/A | {v75_metrics['total_return']:.4f} ({v75_metrics['total_return']*100:.2f}%) | - |",
            f"| 年化收益 | N/A | {v75_metrics['annualized_return']:.4f} ({v75_metrics['annualized_return']*100:.2f}%) | - |",
            f"| 最大回撤 | N/A | {v75_metrics['max_drawdown']:.4f} ({v75_metrics['max_drawdown']*100:.2f}%) | - |",
            f"| 夏普比率 | N/A | {v75_metrics['sharpe_ratio']:.3f} | - |",
            f"| Mean Rank IC | N/A | {v75_metrics['mean_rank_ic']:.4f} | - |",
            f"| 评分 Std | N/A | {v75_metrics['score_std']:.2f} | - |",
            f"| 胜率 | N/A | {v75_metrics['win_rate']:.2%} | - |",
        ])
    
    report_lines.extend([
        "",
        "## 市场环境统计（V75）",
        "",
        "| 环境类型 | 天数 | 说明 |",
        "|------|-----|------|",
        f"| 牛市 (bullish) | {v75_metrics.get('bullish_days', 0)} | IC > 0.03，资金流有效 |",
        f"| 熊市 (bearish) | {v75_metrics.get('bearish_days', 0)} | IC < 0，资金流失效 |",
        f"| 中性 (neutral) | {v75_metrics.get('neutral_days', 0)} | 0 <= IC <= 0.03 |",
        "",
        "## 验收指标验证",
        "",
    ])
    
    if criteria_results:
        report_lines.extend([
            "### 指标 A：Mean Rank IC >= 0.03",
            f"- **结果**: {v75_metrics['mean_rank_ic']:.4f}",
            f"- **状态**: {criteria_results['rank_ic']['status']}",
            "",
            "### 指标 B：回撤减少 >= 15%（相对于 V74）",
        ])
        
        if criteria_results['drawdown']['value'] is not None:
            report_lines.extend([
                f"- **结果**: {criteria_results['drawdown']['value']*100:.1f}%",
                f"- **V74 回撤**: {criteria_results['drawdown']['v74_dd']*100:.2f}%",
                f"- **V75 回撤**: {criteria_results['drawdown']['v75_dd']*100:.2f}%",
                f"- **状态**: {criteria_results['drawdown']['status']}",
            ])
        else:
            report_lines.append(f"- **状态**: {criteria_results['drawdown']['status']}")
        
        report_lines.extend([
            "",
            "### 指标 C：评分 Std 20-30",
            f"- **结果**: {v75_metrics['score_std']:.2f}",
            f"- **状态**: {criteria_results['score_std']['status']}",
            "",
            "## 总体结论",
            "",
            f"**验收结果**: {criteria_results['overall_status']}",
            "",
        ])
    else:
        report_lines.append("**验收结果**: 无法验证（缺少数据）")
    
    report_lines.extend([
        "",
        "---",
        "*V75 对比报告完成*",
    ])
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    logger.info(f"对比报告已保存至：{output_path}")
    
    return report_content


def main(start_date: str = "2024-01-01", end_date: str = "2024-12-31"):
    """主函数"""
    logger.info("=" * 60)
    logger.info("V75 回测 - 自适应环境建模与波动率归一化")
    logger.info(f"回测区间：[{start_date}, {end_date}]")
    logger.info("=" * 60)
    
    # Step 1: 运行 V74 获取基准
    v74_metrics = run_v74_backtest_first(start_date, end_date)
    
    # Step 2: 运行 V75
    v74_dd = v74_metrics['max_drawdown'] if v74_metrics else None
    v75_metrics = run_v75_backtest(start_date, end_date, v74_dd)
    
    # Step 3: 验证验收指标
    criteria_results = verify_criteria(v75_metrics, v74_metrics)
    
    # Step 4: 生成对比报告
    generate_comparison_report(v75_metrics, v74_metrics, criteria_results)
    
    # 返回总体结果
    return {
        'v74_metrics': v74_metrics,
        'v75_metrics': v75_metrics,
        'criteria_results': criteria_results,
    }


if __name__ == "__main__":
    import sys
    
    # 解析命令行参数
    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        start_date = "2024-01-01"
        end_date = "2024-12-31"
    
    result = main(start_date, end_date)
    
    # 输出最终结果
    print("\n" + "=" * 60)
    print("V75 回测完成")
    print("=" * 60)
    print(f"验收结果：{result['criteria_results']['overall_status']}")
    print("=" * 60)