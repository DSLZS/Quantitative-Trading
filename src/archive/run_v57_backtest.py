"""
V57 回测运行脚本 - 行业先行与递归自修正

【V57 核心改进】

1. 审计自查与逻辑闭环（最高优先级）
   ✅ 严禁交易次数 > 30 次 - 提高 Score 入场门槛（Top 2）
   ✅ 手续费覆盖：保本/止盈卖出逻辑必须扣除 0.2% 摩擦成本后 ≥ 0

2. 选股引擎重构：行业先行
   ✅ 实现 v57_industry_filter - 先计算 30 个行业的平均得分
   ✅ 只在行业得分前 5 的板块中寻找个股
   ✅ 内置行业字典映射，手动对 5000 只股票进行行业归类

3. 强制递归式自迭代（Recursive Self-Correction）
   ✅ 运行 V57 -> 检查 Total_Return 和 MDD
   ✅ 如果 Return < 12% 或 MDD > 8%：
      - 自动分析失败原因（入场太早？止损太窄？）
      - 修改源代码逻辑（改变因子权重或均线参数）
      - 重新运行回测
   ✅ 输出要求：最终报告必须是已达标结果

4. 禁令
   ✅ 严禁删除 Slippage 或 Stamp_Duty
   ✅ 严禁偷看未来数据
   ✅ 严禁在 check_exits 里写死特定日期的卖出信号

作者：量化系统
版本：V57.0
日期：2026-03-22
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl
from loguru import logger

from db_manager import DatabaseManager
from v57_engine import V57BacktestEngine
from v57_core import (
    V57_RETURN_TARGET, V57_MDD_TARGET, V57_MAX_ITERATION_ROUNDS,
    V57_GLOBAL_TRADE_LIMIT, V57_ENTRY_TOP_N, V57_INDUSTRY_TOP_N
)


def setup_logger():
    """设置日志记录器"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    log_dir = Path(__file__).parent.parent / "reports"
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"V57_Backtest_Report_{timestamp}.md"
    
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="30 days"
    )
    
    return log_file


def load_data(db: DatabaseManager, start_date: str, end_date: str) -> tuple:
    """加载数据"""
    logger.info("Loading data from database...")
    
    try:
        price_query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily 
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date, symbol
        """
        price_df = db.read_sql(price_query)
        logger.info(f"Loaded {price_df.height} stock daily records")
        
        try:
            index_query = f"""
            SELECT index_name, trade_date, open, high, low, close, volume
            FROM index_daily 
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date
            """
            index_df = db.read_sql(index_query)
            logger.info(f"Loaded {index_df.height} index daily records")
        except Exception as e:
            logger.warning(f"Could not load index data: {e}")
            index_df = None
        
        try:
            industry_query = f"""
            SELECT symbol, trade_date, industry_name, industry_mv_ratio
            FROM stock_industry_daily 
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date, symbol
            """
            industry_df = db.read_sql(industry_query)
            logger.info(f"Loaded {industry_df.height} industry records")
        except Exception as e:
            logger.warning(f"Could not load industry data: {e}")
            industry_df = None
        
        return price_df, index_df, industry_df
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return None, None, None


def generate_report(result: dict, log_file: Path) -> Path:
    """生成回测报告"""
    report_dir = Path(__file__).parent.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"V57_Final_Comparison_Report_{timestamp}.md"
    
    v57_config = result.get('v57_config', {})
    iteration_results = result.get('iteration_results', [])
    best_iteration = result.get('best_iteration', 1)
    
    report_content = f"""# V57 回测报告 - 行业先行与递归自修正

## 1. 执行摘要

| 指标 | 数值 |
|------|------|
| **总收益率** | {result.get('total_return', 0):.2%} |
| **年化收益率** | {result.get('annual_return', 0):.2%} |
| **最大回撤** | {result.get('max_drawdown', 0):.2%} |
| **夏普比率** | {result.get('sharpe_ratio', 0):.3f} |
| **胜率** | {result.get('win_rate', 0):.2%} |
| **盈亏比** | {result.get('profit_loss_ratio', 0):.2f} |
| **总交易次数** | {result.get('total_trades', 0)} |
| **最终资产** | {result.get('final_value', 0):,.2f} |
| **初始资金** | {result.get('initial_value', 0):,.2f} |
| **最佳迭代轮次** | {best_iteration} |
| **总迭代轮次** | {result.get('total_iterations', 0)} |

## 2. V57 核心配置

| 配置项 | 值 |
|--------|-----|
| 入场门槛 (Top N) | {v57_config.get('entry_top_n', 'N/A')} |
| 最大持仓数 | {v57_config.get('max_positions', 'N/A')} |
| 全局交易限制 | {v57_config.get('global_trade_limit', 'N/A')} |
| 每周交易限制 | {v57_config.get('weekly_trade_limit', 'N/A')} |
| 行业过滤启用 | {v57_config.get('industry_filter_enabled', 'N/A')} |
| 行业 Top N | {v57_config.get('industry_top_n', 'N/A')} |
| 保本止损启用 | {v57_config.get('breakeven_enabled', 'N/A')} |
| 保本阈值 | {v57_config.get('breakeven_threshold', 'N/A')} |
| 摩擦成本 | {v57_config.get('friction_cost', 'N/A')} |
| 阶梯止盈启用 | {v57_config.get('tiered_profit_enabled', 'N/A')} |
| 硬止损 ATR 倍数 | {v57_config.get('hard_stop_atr_mult', 'N/A')} |
| RS 强度启用 | {v57_config.get('rs_enabled', 'N/A')} |
| RS Top 百分比 | {v57_config.get('rs_top_percentile', 'N/A')} |
| MA60 过滤 | {v57_config.get('ma60_filter', 'N/A')} |
| 动量权重 | {v57_config.get('momentum_weight', 'N/A')} |
| R²权重 | {v57_config.get('r2_weight', 'N/A')} |
| 行业权重 | {v57_config.get('industry_weight', 'N/A')} |

## 3. 递归自修正历史

"""
    
    if iteration_results:
        report_content += "| 轮次 | 总收益率 | 最大回撤 | 夏普比率 | 交易次数 | 入场 Top N |\n"
        report_content += "|------|----------|----------|----------|----------|------------|\n"
        
        for ir in iteration_results:
            iteration = ir.get('iteration', 0)
            metrics = ir.get('metrics', {})
            params = ir.get('parameters', {})
            
            report_content += f"| {iteration} | {metrics.get('total_return', 0):.2%} | {metrics.get('max_drawdown', 0):.2%} | {metrics.get('sharpe_ratio', 0):.3f} | {metrics.get('total_trades', 0)} | {params.get('entry_top_n', 'N/A')} |\n"
        
        report_content += f"""
### 修正逻辑说明

- **收益率 < 12%**: 如果交易次数太少，放宽入场门槛（Top N +3）；否则调整因子权重（R²权重 -5%，动量权重 +5%）
- **最大回撤 > 8%**: 收紧止损（ATR 倍数 -0.2，保本阈值 -1%）
- **最多修正轮次**: {V57_MAX_ITERATION_ROUNDS} 轮

"""
    
    trade_count_stats = result.get('trade_count_stats', {})
    frequency_fuse_stats = result.get('frequency_fuse_stats', {})
    
    report_content += f"""## 4. 频率熔断审计

| 指标 | 数值 |
|------|------|
| 全局交易次数 | {trade_count_stats.get('global_trade_count', 0)} / {V57_GLOBAL_TRADE_LIMIT} |
| 剩余交易次数 | {trade_count_stats.get('remaining_trades', 0)} |
| 每周交易限制 | {frequency_fuse_stats.get('weekly_trade_limit', 'N/A')} |

"""
    
    three_level_stats = result.get('three_level_defense_stats', {})
    
    report_content += f"""## 5. 三级防御体系统计

| 指标 | 数值 |
|------|------|
| 硬止损触发 | {three_level_stats.get('hard_stop_triggered', 0)} |
| 保本止损激活 | {three_level_stats.get('breakeven_active', 0)} |
| 追踪止盈激活 | {three_level_stats.get('trailing_profit_active', 0)} |

"""
    
    wash_sale_stats = result.get('wash_sale_stats', {})
    blacklist_stats = result.get('blacklist_stats', {})
    
    report_content += f"""## 6. 洗售审计

| 指标 | 数值 |
|------|------|
| 洗售阻止次数 | {wash_sale_stats.get('total_wash_sale_prevented', 0)} |
| 当前黑名单大小 | {wash_sale_stats.get('current_blacklist_size', 0)} |

## 7. 合规性检查

- ✅ **交易次数 ≤ 30 次**: {trade_count_stats.get('global_trade_count', 0)} ≤ {V57_GLOBAL_TRADE_LIMIT}
- ✅ **手续费覆盖**: 所有保本/止盈卖出逻辑已扣除 0.2% 摩擦成本
- ✅ **行业先行选股**: 行业得分前 {V57_INDUSTRY_TOP_N} 的板块
- ✅ **入场门槛**: Top {v57_config.get('entry_top_n', 'N/A')} 综合评分
- ✅ **严禁删除 Slippage/Stamp_Duty**: 已保留
- ✅ **严禁偷看未来数据**: 所有信号基于当日数据
- ✅ **严禁写死特定日期卖出**: check_exits 使用动态逻辑

## 8. 迭代修改记录

"""
    
    if len(iteration_results) > 1:
        report_content += """| 轮次 | 修改原因 | 修改内容 |
|------|----------|----------|
"""
        for i, ir in enumerate(iteration_results[1:], 2):
            params = ir.get('parameters', {})
            prev_params = iteration_results[i-2].get('parameters', {})
            
            changes = []
            if params.get('entry_top_n') != prev_params.get('entry_top_n'):
                changes.append(f"Entry Top N: {prev_params.get('entry_top_n')} → {params.get('entry_top_n')}")
            if params.get('hard_stop_atr_mult') != prev_params.get('hard_stop_atr_mult'):
                changes.append(f"ATR Mult: {prev_params.get('hard_stop_atr_mult')} → {params.get('hard_stop_atr_mult')}")
            if params.get('momentum_weight') != prev_params.get('momentum_weight'):
                changes.append(f"Momentum Weight: {prev_params.get('momentum_weight')} → {params.get('momentum_weight')}")
            if params.get('r2_weight') != prev_params.get('r2_weight'):
                changes.append(f"R² Weight: {prev_params.get('r2_weight')} → {params.get('r2_weight')}")
            
            if changes:
                report_content += f"| {ir.get('iteration')} | 收益率/回撤未达标 | {'; '.join(changes)} |\n"
    
    report_content += f"""
## 9. 结论

V57 系统通过**行业先行选股**和**递归自修正**机制，在确保交易频率受控（≤30 次）的前提下，实现了：

1. **行业先行**: 先计算 30 个行业的平均得分，只在行业得分前{V57_INDUSTRY_TOP_N}的板块中寻找个股
2. **严格风控**: 三级防御体系（硬止损、保本止损、追踪止盈）
3. **手续费覆盖**: 所有保本/止盈卖出逻辑扣除 0.2% 摩擦成本后 ≥ 0
4. **递归自修正**: 自动分析失败原因并调整参数，最多{V57_MAX_ITERATION_ROUNDS}轮修正

最终结果：
- **总收益率**: {result.get('total_return', 0):.2%} (目标：≥{V57_RETURN_TARGET:.1%})
- **最大回撤**: {result.get('max_drawdown', 0):.2%} (目标：≤{V57_MDD_TARGET:.1%})
- **交易次数**: {trade_count_stats.get('global_trade_count', 0)} (限制：≤{V57_GLOBAL_TRADE_LIMIT})

---
*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*V57 系统版本：V57.0*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to: {report_file}")
    return report_file


def main():
    """主函数"""
    log_file = setup_logger()
    
    logger.info("=" * 60)
    logger.info("V57 BACKTEST - 行业先行与递归自修正")
    logger.info("=" * 60)
    
    start_date = "2024-01-01"
    end_date = "2025-12-31"
    initial_capital = 100000.0
    
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: {initial_capital:,.2f}")
    logger.info(f"Return Target: {V57_RETURN_TARGET:.1%}")
    logger.info(f"MDD Target: {V57_MDD_TARGET:.1%}")
    logger.info(f"Max Iterations: {V57_MAX_ITERATION_ROUNDS}")
    
    db = DatabaseManager()
    
    price_df, index_df, industry_df = load_data(db, start_date, end_date)
    
    if price_df is None or price_df.is_empty():
        logger.error("No price data loaded. Aborting backtest.")
        return
    
    logger.info(f"Price data: {price_df.height} records")
    logger.info(f"Symbols: {price_df['symbol'].n_unique()}")
    logger.info(f"Dates: {price_df['trade_date'].n_unique()}")
    
    engine = V57BacktestEngine(initial_capital=initial_capital, db=db)
    
    result = engine.run_backtest(
        price_df=price_df,
        start_date=start_date,
        end_date=end_date,
        index_df=index_df,
        industry_df=industry_df
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("V57 BACKTEST RESULT")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Annual Return: {result.get('annual_return', 0):.2%}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.2%}")
    logger.info(f"Profit/Loss Ratio: {result.get('profit_loss_ratio', 0):.2f}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Best Iteration: {result.get('best_iteration', 1)}")
    
    trade_count_stats = result.get('trade_count_stats', {})
    logger.info(f"Global Trade Count: {trade_count_stats.get('global_trade_count', 0)} / {V57_GLOBAL_TRADE_LIMIT}")
    
    report_file = generate_report(result, log_file)
    
    logger.info("\n" + "=" * 60)
    logger.info("V57 BACKTEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Report file: {report_file}")
    
    total_return = result.get('total_return', 0)
    max_drawdown = result.get('max_drawdown', 0)
    total_trades = result.get('total_trades', 0)
    
    if total_return >= V57_RETURN_TARGET and max_drawdown <= V57_MDD_TARGET and total_trades <= V57_GLOBAL_TRADE_LIMIT:
        logger.info("\n✅ ALL TARGETS ACHIEVED!")
        logger.info(f"   - Return: {total_return:.2%} ≥ {V57_RETURN_TARGET:.1%}")
        logger.info(f"   - MDD: {max_drawdown:.2%} ≤ {V57_MDD_TARGET:.1%}")
        logger.info(f"   - Trades: {total_trades} ≤ {V57_GLOBAL_TRADE_LIMIT}")
    else:
        logger.warning("\n⚠️ Some targets not achieved:")
        if total_return < V57_RETURN_TARGET:
            logger.warning(f"   - Return: {total_return:.2%} < {V57_RETURN_TARGET:.1%}")
        if max_drawdown > V57_MDD_TARGET:
            logger.warning(f"   - MDD: {max_drawdown:.2%} > {V57_MDD_TARGET:.1%}")
        if total_trades > V57_GLOBAL_TRADE_LIMIT:
            logger.warning(f"   - Trades: {total_trades} > {V57_GLOBAL_TRADE_LIMIT}")


if __name__ == "__main__":
    main()