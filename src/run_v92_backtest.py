"""
V92 回测运行脚本

【V92 核心任务】
1. 修复 V91 信号反转错误
2. 引入量价二阶导逻辑
3. IC 驱动因子权重动态调整
4. 简化中性化（回归 V90 稳健方法）

【V92 硬性指标】
- 指标 A：T+1 Rank IC ≥ 0.05，IC IR ≥ 0.6
- 指标 B：最大回撤 ≤ 10%
- 指标 C：年化换手率 300%-500%
- 指标 D：数学一致性检查（误差 < 0.1%）
- 指标 E：预测一致性 T+1 IC ≥ T+2 IC ≥ T+3 IC

作者：量化系统
版本：V92.0
日期：2026-03-30
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.v92_engine import V92Engine, V92EngineConfig, run_v92_backtest, print_v92_report
from loguru import logger


def main():
    """主函数"""
    print("=" * 70)
    print("V92 IC 驱动预测回测系统")
    print("=" * 70)
    print(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 配置 logger
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 创建配置
    config = V92EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        oos_years=["2019", "2021", "2024"],
        initial_capital=100000.00,
        max_positions=30,
        warmup_period=60,
        enable_divergence=True,  # 启用量价背离
        enable_ic_weighting=True,  # 启用 IC 权重
        enable_neutralization=True,  # 启用中性化
    )
    
    print(f"配置:")
    print(f"  - 回测区间：{config.start_date} 至 {config.end_date}")
    print(f"  - OOS 年份：{config.oos_years}")
    print(f"  - 初始资金：{config.initial_capital:,.2f}")
    print(f"  - 最大持仓：{config.max_positions}")
    print(f"  - 量价背离：{'启用' if config.enable_divergence else '禁用'}")
    print(f"  - IC 权重：{'启用' if config.enable_ic_weighting else '禁用'}")
    print(f"  - 中性化：{'启用' if config.enable_neutralization else '禁用'}")
    print("=" * 70)
    
    # 运行回测
    result = run_v92_backtest(config)
    
    # 打印报告
    print_v92_report(result)
    
    # 保存结果
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 JSON 结果
    json_path = os.path.join(output_dir, f"V92_backtest_result_{timestamp}.json")
    
    # 序列化结果
    serializable_result = {
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': result.get('ic_audit', {}),
        'consistency_check': result.get('consistency_check', {}),
        'trade_results': {
            'total_return': result.get('trade_results', {}).get('total_return', 0.0),
            'annualized_return': result.get('trade_results', {}).get('annualized_return', 0.0),
            'final_value': result.get('trade_results', {}).get('final_value', 0.0),
            'max_drawdown': result.get('trade_results', {}).get('max_drawdown', 0.0),
            'annualized_turnover': result.get('trade_results', {}).get('annualized_turnover', 0.0),
            'total_trading_days': result.get('trade_results', {}).get('total_trading_days', 0),
            'total_trades': result.get('trade_results', {}).get('total_trades', 0),
            'rebalance_count': result.get('trade_results', {}).get('rebalance_count', 0),
            'annual_returns': result.get('trade_results', {}).get('annual_returns', {}),
            'avg_annual_return': result.get('trade_results', {}).get('avg_annual_return', 0.0),
        },
        'audit_report': result.get('audit_report', ''),
        'trade_count': len(result.get('trade_records', [])),
        'snapshot_count': len(result.get('daily_snapshots', [])),
        'rebalance_count': len(result.get('rebalance_dates', [])),
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存至：{json_path}")
    
    # 保存 Markdown 报告
    md_path = os.path.join(output_dir, f"V92_IC_Driven_Audit_Report_{timestamp}.md")
    
    audit_report = result.get('audit_report', '')
    if audit_report:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(f"# V92 IC 驱动预测审计报告\n\n")
            f.write(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**策略版本**: V92.0\n\n")
            f.write(f"---\n\n")
            f.write(audit_report)
            f.write("\n")
    
    print(f"审计报告已保存至：{md_path}")
    
    # 检查是否所有指标通过
    consistency = result.get('consistency_check', {}).get('summary', {})
    all_passed = consistency.get('all_passed', False)
    
    if all_passed:
        print("\n" + "=" * 70)
        print("✓ 所有 V92 硬性指标通过！")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("✗ 部分 V92 硬性指标未通过，请检查报告")
        print("=" * 70)
    
    return result


if __name__ == "__main__":
    main()