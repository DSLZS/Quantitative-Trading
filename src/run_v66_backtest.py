"""
V66 回测运行脚本 - 机构资金踪迹模型 (IC 优化版)

【使用说明】
    python src/run_v66_backtest.py

【核心特性】
1. 拒绝降级逻辑：严禁 Fallback 到 price_only 模式
2. 数据熔断机制：stock_fund_flow 和 stock_industry_daily 少于 10,000 条则停止
3. 建表校验：自动执行 CREATE TABLE IF NOT EXISTS
4. IC 评估：输出《信号预测质量 IC 统计表》，要求 IC 均值 > 0.02
5. AE 指标：AE = (Win_Rate × P/L_Ratio) / Max_Drawdown × √Trade_Count

作者：量化系统
版本：V66.0
日期：2026-03-24
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger
from v66_engine import V66BacktestEngine, run_v66_backtest
from v66_core import V66_INITIAL_CAPITAL, V66_IC_TARGET_MEAN


def configure_logging():
    """配置日志输出"""
    logger.remove()
    
    # 控制台输出
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 文件输出
    log_file = f"reports/v66_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG"
    )


def run_v66_full_backtest(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    运行 V66 完整回测
    
    Parameters
    ----------
    config : Dict[str, Any], optional
        配置字典
        
    Returns
    -------
    Dict[str, Any]
        回测结果
    """
    logger.info("=" * 60)
    logger.info("V66 机构资金踪迹模型 - IC 优化版")
    logger.info("=" * 60)
    
    # 默认配置
    if config is None:
        config = {
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': V66_INITIAL_CAPITAL,
            'max_positions': 10,
        }
    
    logger.info(f"回测区间：[{config['start_date']}, {config['end_date']}]")
    logger.info(f"初始资金：{config['initial_capital']:,.2f}")
    logger.info(f"最大持仓：{config['max_positions']} 只")
    logger.info("=" * 60)
    
    try:
        # 运行回测
        metrics = run_v66_backtest(config)
        
        # 打印最终报告
        print_final_report(metrics)
        
        return metrics
        
    except ValueError as e:
        logger.error(f"V66: 数据熔断触发：{e}")
        raise
    except Exception as e:
        logger.error(f"V66: 回测失败：{e}")
        raise


def print_final_report(metrics: Dict[str, Any]):
    """打印最终报告"""
    print("\n" + "=" * 60)
    print("V66 回测最终报告")
    print("=" * 60)
    
    # 基础指标
    print("\n【基础指标】")
    print(f"  总收益率：  {metrics.get('total_return', 0)*100:.2f}%")
    print(f"  年化收益：  {metrics.get('annual_return', 0)*100:.2f}%")
    print(f"  波动率：    {metrics.get('volatility', 0)*100:.2f}%")
    print(f"  夏普比率：  {metrics.get('sharpe', 0):.2f}")
    print(f"  最大回撤：  {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    # 交易统计
    print("\n【交易统计】")
    print(f"  交易次数：  {metrics.get('n_trades', 0)}")
    print(f"  胜率：      {metrics.get('win_rate', 0)*100:.1f}%")
    print(f"  盈亏比：    {metrics.get('profit_loss_ratio', 0):.2f}")
    print(f"  AE 指标：    {metrics.get('ae_metric', 0):.2f}")
    
    # IC 统计
    print("\n【IC 统计 - 信号预测质量】")
    print(f"  Mean IC:      {metrics.get('ic_mean', 0):.4f} (目标：>{V66_IC_TARGET_MEAN})")
    print(f"  Mean Rank IC: {metrics.get('ic_rank_mean', 0):.4f}")
    print(f"  IC IR:        {metrics.get('ic_ir', 0):.2f}")
    print(f"  Positive Ratio: {metrics.get('ic_positive_ratio', 0)*100:.1f}%")
    
    # IC 达标检查
    ic_mean = metrics.get('ic_mean', 0)
    if ic_mean >= V66_IC_TARGET_MEAN:
        print(f"\n  ✓ IC 达标：Mean IC ({ic_mean:.4f}) >= 目标 ({V66_IC_TARGET_MEAN})")
    else:
        print(f"\n  ✗ IC 未达标：Mean IC ({ic_mean:.4f}) < 目标 ({V66_IC_TARGET_MEAN})")
    
    # 数据可信度
    print("\n【数据可信度统计】")
    dc_stats = metrics.get('data_credibility_stats', {})
    for credibility, count in dc_stats.items():
        print(f"  {credibility}: {count}")
    
    print("=" * 60)


def main():
    """主函数"""
    configure_logging()
    
    # 回测配置
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': V66_INITIAL_CAPITAL,
        'max_positions': 10,
    }
    
    try:
        # 运行回测
        metrics = run_v66_full_backtest(config)
        
        print("\n" + "=" * 60)
        print("V66 回测完成!")
        print("=" * 60)
        
        # 保存结果
        output_file = f"reports/v66_backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至：{output_file}")
        
    except ValueError as e:
        logger.error(f"V66: 数据熔断触发：{e}")
        print("\n" + "=" * 60)
        print("V66 回测失败 - 数据熔断触发")
        print("=" * 60)
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"V66: 回测失败：{e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)
        print("V66 回测失败")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()