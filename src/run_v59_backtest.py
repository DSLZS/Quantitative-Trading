"""
V59 回测运行脚本 - 打破负优化循环与逻辑彻底演变

【使用说明】
python src/run_v59_backtest.py

【核心功能】
1. 运行 V59 回测引擎
2. 执行 MasterLoop 迭代协议
3. 生成回测报告
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.v59_core import (
    V59FactorEngine, V59IndustryLoader,
    V59_INITIAL_CAPITAL, V59_MAX_ITERATION_ROUNDS
)
from src.v59_engine import V59BacktestEngine, MasterLoop


def setup_logger():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )


def load_test_data(start_date: str = "2025-01-01", end_date: str = "2025-12-31"):
    """
    加载测试数据
    
    注意：实际使用时需要从数据库或文件加载真实数据
    这里使用模拟数据进行测试
    """
    logger.info(f"Loading test data from {start_date} to {end_date}...")
    
    # 生成模拟股票数据
    symbols = [f"60000{i:02d}.SH" for i in range(1, 51)]  # 50 只股票
    symbols += [f"00000{i:02d}.SZ" for i in range(1, 51)]  # 50 只股票
    
    # 生成日期范围
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # 过滤周末
    trade_dates = []
    current = start_dt
    while current <= end_dt:
        if current.weekday() < 5:
            trade_dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    # 生成模拟价格数据
    import random
    random.seed(42)
    
    price_records = []
    for symbol in symbols[:20]:  # 使用 20 只股票减少数据量
        base_price = random.uniform(10, 100)
        for trade_date in trade_dates:
            open_price = base_price * (1 + random.uniform(-0.03, 0.03))
            high_price = max(open_price, base_price * (1 + random.uniform(0, 0.05)))
            low_price = min(open_price, base_price * (1 - random.uniform(0, 0.05)))
            close_price = base_price * (1 + random.uniform(-0.02, 0.02))
            volume = random.randint(100000, 10000000)
            
            price_records.append({
                'symbol': symbol,
                'trade_date': trade_date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
            
            base_price = close_price
    
    price_df = pl.DataFrame(price_records)
    
    # 计算因子数据
    factor_engine = V59FactorEngine()
    factor_df, factor_status = factor_engine.compute_all_factors(price_df)
    
    logger.info(f"Loaded {len(symbols[:20])} stocks, {len(trade_dates)} trade days")
    logger.info(f"Factor status: {factor_status}")
    
    return factor_df, price_df, None, None


def run_single_backtest(factor_df: pl.DataFrame, price_df: pl.DataFrame,
                        start_date: str, end_date: str) -> dict:
    """运行单次回测"""
    logger.info("\n" + "="*60)
    logger.info("Running V59 Single Backtest")
    logger.info("="*60)
    
    engine = V59BacktestEngine(initial_capital=V59_INITIAL_CAPITAL)
    
    result = engine.run_backtest(
        factor_data=factor_df,
        price_data=price_df,
        industry_data=None,
        index_data=None,
        start_date=start_date,
        end_date=end_date
    )
    
    if result.get('error'):
        logger.error(f"Backtest failed: {result['error']}")
        return {}
    
    backtest_result = result.get('result', {})
    
    logger.info("\n--- V59 Backtest Result ---")
    logger.info(f"Total Return: {backtest_result.get('total_return', 0):.2%}")
    logger.info(f"Max Drawdown: {backtest_result.get('max_drawdown', 0):.2%}")
    logger.info(f"Sharpe Ratio: {backtest_result.get('sharpe_ratio', 0):.2f}")
    logger.info(f"Win Rate: {backtest_result.get('win_rate', 0):.2%}")
    logger.info(f"Profit/Loss Ratio: {backtest_result.get('profit_loss_ratio', 0):.2f}")
    logger.info(f"Total Trades: {backtest_result.get('total_trades', 0)}")
    logger.info(f"Price Audit Violations: {backtest_result.get('price_audit_violations', 0)}")
    logger.info(f"Meets Target: {backtest_result.get('meets_target', False)}")
    logger.info(f"Target Analysis: {backtest_result.get('target_analysis', 'N/A')}")
    
    return backtest_result


def run_master_loop(factor_df: pl.DataFrame, price_df: pl.DataFrame,
                    start_date: str, end_date: str, max_iterations: int = 5):
    """运行 MasterLoop 迭代协议"""
    logger.info("\n" + "="*60)
    logger.info("Running V59 MasterLoop Iteration Protocol")
    logger.info(f"Max Iterations: {max_iterations}")
    logger.info("="*60)
    
    master_loop = MasterLoop(max_iterations=max_iterations)
    
    for i in range(max_iterations):
        logger.info(f"\n--- Iteration {i+1}/{max_iterations} ---")
        
        result = master_loop.run_iteration(
            factor_data=factor_df,
            price_data=price_df,
            industry_data=None,
            index_data=None,
            start_date=start_date,
            end_date=end_date
        )
        
        if result.get('meets_target', False):
            logger.info(f"\n✅ Target met at iteration {i+1}!")
            break
    
    # 获取最佳结果
    best_result = master_loop.get_best_result()
    if best_result:
        logger.info("\n--- Best Iteration Result ---")
        logger.info(f"Iteration: {best_result.iteration}")
        logger.info(f"Logic Path: {best_result.logic_path}")
        logger.info(f"Total Return: {best_result.metrics.get('total_return', 0):.2%}")
        logger.info(f"Profit/Loss Ratio: {best_result.metrics.get('profit_loss_ratio', 0):.2f}")
    
    # 对比逻辑路径
    comparison = master_loop.compare_logic_paths()
    if comparison:
        logger.info("\n--- Logic Path Comparison ---")
        for path, metrics in comparison.get('comparison', {}).items():
            logger.info(f"{path}: Return={metrics['avg_return']:.2%}, P/L={metrics['avg_pl_ratio']:.2f}, Stability={metrics['stability_score']:.2f}")
        logger.info(f"Best Logic Path: {comparison.get('best_logic_path', 'N/A')}")
    
    return master_loop


def generate_report(result: dict, output_path: str = "reports/V59_Backtest_Report.md"):
    """生成回测报告"""
    if not result:
        logger.warning("No result to generate report")
        return
    
    report_content = f"""# V59 回测报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**版本**: V59.0

---

## 一、核心指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {result.get('total_return', 0):.2%} |
| 年化收益 | {result.get('annual_return', 0):.2%} |
| 最大回撤 | {result.get('max_drawdown', 0):.2%} |
| 夏普比率 | {result.get('sharpe_ratio', 0):.2f} |
| 胜率 | {result.get('win_rate', 0):.2%} |
| 盈亏比 | {result.get('profit_loss_ratio', 0):.2f} |
| 交易次数 | {result.get('total_trades', 0)} |
| 盈利次数 | {result.get('winning_trades', 0)} |
| 亏损次数 | {result.get('losing_trades', 0)} |

---

## 二、V59 核心改进验证

### 2.1 ATR 动态止损

- 止损模式：{result.get('hard_stop_mode', 'atr_only')}
- ATR 倍数：3.0
- 固定止损：已废除

### 2.2 利润生长空间

- 保本止损激活阈值：10%
- 追踪止盈激活阈值：15%
- 追踪止盈回撤：3.0 * ATR

### 2.3 选股引擎 3.0

- 成交量突破倍数：2.0
- 行业趋势过滤：已启用
- MA20 突破要求：已启用

---

## 三、合规性审计

### 3.1 成交价审计

- 违规次数：{result.get('price_audit_violations', 0)}
- 审计状态：{"✅ 通过" if result.get('price_audit_violations', 0) == 0 else "❌ 存在违规"}

### 3.2 目标达成

- 是否达标：{result.get('meets_target', False)}
- 目标分析：{result.get('target_analysis', 'N/A')}

---

## 四、费用分析

| 费用类型 | 金额 |
|----------|------|
| 总佣金 | {result.get('total_commission', 0):,.2f} |
| 总滑点 | {result.get('total_slippage', 0):,.2f} |
| 总印花税 | {result.get('total_stamp_duty', 0):,.2f} |
| 总过户费 | {result.get('total_transfer_fee', 0):,.2f} |

---

## 五、交易质量

| 指标 | 数值 |
|------|------|
| 平均持仓天数 | {result.get('avg_holding_days', 0):.1f} |
| 平均盈利 | {result.get('avg_win', 0):,.2f} |
| 平均亏损 | {result.get('avg_loss', 0):,.2f} |
| 最大盈利 | {result.get('largest_win', 0):,.2f} |
| 最大亏损 | {result.get('largest_loss', 0):,.2f} |
| 连续盈利 | {result.get('consecutive_wins', 0)} |
| 连续亏损 | {result.get('consecutive_losses', 0)} |

---

## 六、V59 核心哲学

> 宁可让部分盈利单变成亏损单，也要换取大周期的盈利分布。
> 动态止损适应市场波动率，给利润足够的生长空间。

---

*报告由 V59 回测系统自动生成*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to {output_path}")


def main():
    """主函数"""
    setup_logger()
    
    logger.info("="*60)
    logger.info("V59 Backtest System - 打破负优化循环与逻辑彻底演变")
    logger.info("="*60)
    
    # 设置回测参数
    start_date = "2025-01-01"
    end_date = "2025-06-30"  # 使用 6 个月数据进行快速测试
    
    # 加载测试数据
    factor_df, price_df, industry_df, index_df = load_test_data(start_date, end_date)
    
    if factor_df is None or price_df is None:
        logger.error("Failed to load test data")
        return
    
    # 运行单次回测
    result = run_single_backtest(factor_df, price_df, start_date, end_date)
    
    # 运行 MasterLoop 迭代（最多 5 轮）
    master_loop = run_master_loop(factor_df, price_df, start_date, end_date, max_iterations=5)
    
    # 生成报告
    generate_report(result)
    
    logger.info("\n" + "="*60)
    logger.info("V59 Backtest Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()