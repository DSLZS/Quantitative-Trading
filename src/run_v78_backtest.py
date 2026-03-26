"""
V78 Backtest Runner - 残差动量增强与多时空尺度特征融合回测运行脚本

【使用说明】
1. 直接运行：python src/run_v78_backtest.py
2. 自定义回测区间：修改下方的 start_date 和 end_date 参数

【V78 核心特性】
- Residual Momentum: 个股收益率 - 行业收益率
- Short-term Reversal: 过去 5 日涨幅惩罚
- Dynamic Liquidity Audit: V-Shock 适度放量加分
- Multi-Timeframe RS: 0.7*RS_5 + 0.3*RS_20

【验收指标】
- 指标 A: Mean Rank IC >= 0.03
- 指标 B: 最大回撤 <= 8%
- 指标 C: 胜率 >= 45%

作者：量化系统
版本：V78.0
日期：2026-03-26
"""

import argparse
from datetime import datetime
from pathlib import Path

from src.v78_engine import run_v78_backtest, V78BacktestResult


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V78 回测运行器')
    parser.add_argument('--start', type=str, default='2024-01-01',
                        help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-12-31',
                        help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default=None,
                        help='报告输出路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("V78 回测运行器 - 残差动量增强与多时空尺度特征融合")
    print("=" * 60)
    print(f"回测区间：[{args.start}, {args.end}]")
    print(f"报告输出：{args.output or 'reports/v78_backtest_report_<timestamp>.md'}")
    print("=" * 60)
    
    try:
        # 运行回测
        result: V78BacktestResult = run_v78_backtest(
            start_date=args.start,
            end_date=args.end,
            output_path=args.output
        )
        
        # 打印验收结论
        print("\n" + "=" * 60)
        print("【V78 验收结论】")
        print("=" * 60)
        
        rank_ic_pass = result.mean_rank_ic >= 0.03
        drawdown_pass = result.max_drawdown <= 0.08
        win_rate_pass = result.win_rate >= 0.45
        
        print(f"指标 A - Mean Rank IC >= 0.03: {'✓' if rank_ic_pass else '✗'} ({result.mean_rank_ic:.4f})")
        print(f"指标 B - 最大回撤 <= 8%: {'✓' if drawdown_pass else '✗'} ({result.max_drawdown*100:.2f}%)")
        print(f"指标 C - 胜率 >= 45%: {'✓' if win_rate_pass else '✗'} ({result.win_rate*100:.2f}%)")
        print("=" * 60)
        
        # 综合结论
        all_pass = rank_ic_pass and drawdown_pass and win_rate_pass
        if all_pass:
            print("\n【结论】V78 所有验收指标达标！✓")
        else:
            print("\n【结论】V78 部分验收指标未达标，需要进一步优化。")
            if not rank_ic_pass:
                print("  - Rank IC 未达标，可能需要调整因子权重或增加新的 alpha 源")
            if not drawdown_pass:
                print("  - 回撤未达标，可能需要加强止损或降低仓位")
            if not win_rate_pass:
                print("  - 胜率未达标，可能需要优化选股逻辑或调整持仓周期")
        
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n回测执行失败：{e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()