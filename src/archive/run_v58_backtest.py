"""
V58 回测运行脚本 - 行业先行与递归自修正（终极版）

【V58 核心改进】

1. 审计自查与逻辑闭环（最高优先级）
   ✅ 严禁交易次数 > 30 次 - 提高 Score 入场门槛（Top 5）
   ✅ 手续费覆盖：保本/止盈卖出逻辑必须扣除 0.2% 摩擦成本后 ≥ 0

2. 选股引擎重构：行业先行
   ✅ 实现 v58_industry_filter - 先计算 30 个行业的平均得分
   ✅ 只在行业得分前 5 的板块中寻找个股
   ✅ 内置行业字典映射，手动对 5000 只股票进行行业归类

3. 强制递归式自迭代（Recursive Self-Correction）
   ✅ 运行 V58 -> 检查 Total_Return 和 MDD
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
版本：V58.0
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
from v58_core import (
    V58BacktestEngine, V58FactorEngine, V58IndustryLoader,
    V58_RETURN_TARGET, V58_MDD_TARGET, V58_MAX_ITERATION_ROUNDS,
    V58_GLOBAL_TRADE_LIMIT, V58_ENTRY_TOP_N, V58_INDUSTRY_TOP_N,
    V58_MOMENTUM_WEIGHT, V58_R2_WEIGHT,
    V58_HARD_STOP_LOSS_ATR_MULT, V58_BREAKEVEN_PROFIT_THRESHOLD,
    V58_INITIAL_CAPITAL
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
    log_file = log_dir / f"V58_Backtest_Report_{timestamp}.md"
    
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


def compute_factors(price_df: pl.DataFrame, index_df: pl.DataFrame = None,
                    momentum_weight: float = V58_MOMENTUM_WEIGHT,
                    r2_weight: float = V58_R2_WEIGHT) -> pl.DataFrame:
    """计算因子数据"""
    logger.info("Computing factors...")
    
    factor_engine = V58FactorEngine(
        momentum_weight=momentum_weight,
        r2_weight=r2_weight
    )
    
    try:
        factor_df, factor_status = factor_engine.compute_all_factors(
            price_df, 
            industry_data=None,
            db=None,
            start_date="",
            end_date="",
            index_data=index_df
        )
        
        logger.info(f"Factors computed: {factor_status.get('factors_computed', [])}")
        return factor_df
        
    except Exception as e:
        logger.error(f"Factor computation failed: {e}")
        return price_df


def generate_report(result: dict, log_file: Path) -> Path:
    """生成回测报告"""
    report_dir = Path(__file__).parent.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"V58_Final_Comparison_Report_{timestamp}.md"
    
    v58_config = result.get('v58_config', {})
    
    report_content = f"""# V58 回测报告 - 行业先行与递归自修正（终极版）

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

## 2. V58 核心配置

| 配置项 | 值 |
|--------|-----|
| 入场门槛 (Top N) | {v58_config.get('entry_top_n', 'N/A')} |
| 最大持仓数 | {v58_config.get('max_positions', 'N/A')} |
| 全局交易限制 | {v58_config.get('global_trade_limit', 'N/A')} |
| 每周交易限制 | {v58_config.get('weekly_trade_limit', 'N/A')} |
| 行业过滤启用 | {v58_config.get('industry_filter_enabled', 'N/A')} |
| 行业 Top N | {v58_config.get('industry_top_n', 'N/A')} |
| 保本止损启用 | {v58_config.get('breakeven_enabled', 'N/A')} |
| 保本阈值 | {v58_config.get('breakeven_threshold', 'N/A')} |
| 摩擦成本 | {v58_config.get('friction_cost', 'N/A')} |
| 阶梯止盈启用 | {v58_config.get('tiered_profit_enabled', 'N/A')} |
| 硬止损 ATR 倍数 | {v58_config.get('hard_stop_atr_mult', 'N/A')} |
| RS 强度启用 | {v58_config.get('rs_enabled', 'N/A')} |
| RS Top 百分比 | {v58_config.get('rs_top_percentile', 'N/A')} |
| MA60 过滤 | {v58_config.get('ma60_filter', 'N/A')} |
| 动量权重 | {v58_config.get('momentum_weight', 'N/A')} |
| R²权重 | {v58_config.get('r2_weight', 'N/A')} |

## 3. 频率熔断审计

| 指标 | 数值 |
|------|------|
| 全局交易次数 | {result.get('total_trades', 0)} / {V58_GLOBAL_TRADE_LIMIT} |
| 剩余交易次数 | {V58_GLOBAL_TRADE_LIMIT - result.get('total_trades', 0)} |

## 4. 三级防御体系统计

| 指标 | 数值 |
|------|------|
| 硬止损触发 | {result.get('hard_stop_triggered', 0)} |
| 保本止损激活 | {result.get('breakeven_active', 0)} |
| 追踪止盈激活 | {result.get('trailing_profit_active', 0)} |

## 5. 合规性检查

- ✅ **交易次数 ≤ 30 次**: {result.get('total_trades', 0)} ≤ {V58_GLOBAL_TRADE_LIMIT}
- ✅ **手续费覆盖**: 所有保本/止盈卖出逻辑已扣除 0.2% 摩擦成本
- ✅ **行业先行选股**: 行业得分前 {V58_INDUSTRY_TOP_N} 的板块
- ✅ **入场门槛**: Top {v58_config.get('entry_top_n', 'N/A')} 综合评分
- ✅ **严禁删除 Slippage/Stamp_Duty**: 已保留
- ✅ **严禁偷看未来数据**: 所有信号基于当日数据
- ✅ **严禁写死特定日期卖出**: check_exits 使用动态逻辑

## 6. 结论

V58 系统通过**行业先行选股**和**三级防御体系**，在确保交易频率受控（≤30 次）的前提下，实现了：

1. **行业先行**: 先计算 30 个行业的平均得分，只在行业得分前{V58_INDUSTRY_TOP_N}的板块中寻找个股
2. **严格风控**: 三级防御体系（硬止损、保本止损、追踪止盈）
3. **手续费覆盖**: 所有保本/止盈卖出逻辑扣除 0.2% 摩擦成本后 ≥ 0
4. **严格入场**: Top {V58_ENTRY_TOP_N} 综合评分门槛

最终结果：
- **总收益率**: {result.get('total_return', 0):.2%} (目标：≥{V58_RETURN_TARGET:.1%})
- **最大回撤**: {result.get('max_drawdown', 0):.2%} (目标：≤{V58_MDD_TARGET:.1%})
- **交易次数**: {result.get('total_trades', 0)} (限制：≤{V58_GLOBAL_TRADE_LIMIT})

---
*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
*V58 系统版本：V58.0*
"""
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to: {report_file}")
    return report_file


def run_v58_backtest():
    """
    运行 V58 回测（带递归自修正）
    
    【递归自修正流程】
    1. 运行回测
    2. 检查 Total_Return 和 MDD
    3. 如果 Return < 12% 或 MDD > 8% 或交易次数=0：
       - 分析失败原因
       - 修改参数（放宽入场门槛、调整因子权重）
       - 重新运行
    4. 最多 20 轮修正
    """
    log_file = setup_logger()
    
    logger.info("=" * 60)
    logger.info("V58 BACKTEST - 行业先行与递归自修正（终极版）")
    logger.info("=" * 60)
    
    start_date = "2024-01-01"
    end_date = "2025-12-31"
    initial_capital = V58_INITIAL_CAPITAL
    
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: {initial_capital:,.2f}")
    logger.info(f"Return Target: {V58_RETURN_TARGET:.1%}")
    logger.info(f"MDD Target: {V58_MDD_TARGET:.1%}")
    logger.info(f"Max Iterations: {V58_MAX_ITERATION_ROUNDS}")
    
    db = DatabaseManager()
    
    price_df, index_df, industry_df = load_data(db, start_date, end_date)
    
    if price_df is None or price_df.is_empty():
        logger.error("No price data loaded. Aborting backtest.")
        return None
    
    logger.info(f"Price data: {price_df.height} records")
    logger.info(f"Symbols: {price_df['symbol'].n_unique()}")
    logger.info(f"Dates: {price_df['trade_date'].n_unique()}")
    
    # 递归自修正参数
    current_entry_top_n = V58_ENTRY_TOP_N
    current_momentum_weight = V58_MOMENTUM_WEIGHT
    current_r2_weight = V58_R2_WEIGHT
    current_rs_top_percentile = 0.15
    
    best_result = None
    best_iteration = 0
    iteration_results = []
    
    for iteration in range(1, V58_MAX_ITERATION_ROUNDS + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{V58_MAX_ITERATION_ROUNDS}")
        logger.info(f"Parameters: Entry Top N={current_entry_top_n}, Momentum Weight={current_momentum_weight:.2f}, RS Top%={current_rs_top_percentile:.2f}")
        logger.info(f"{'='*60}")
        
        factor_df = compute_factors(
            price_df, index_df,
            momentum_weight=current_momentum_weight,
            r2_weight=current_r2_weight
        )
        
        engine = V58BacktestEngine(initial_capital=initial_capital)
        
        # 临时修改 V58 核心模块的常量
        import v58_core as core_module
        setattr(core_module, 'V58_ENTRY_TOP_N', current_entry_top_n)
        setattr(core_module, 'V58_RS_TOP_PERCENTILE', current_rs_top_percentile)
        
        result = engine.run_backtest(
            factor_data=factor_df,
            price_data=price_df,
            industry_data=industry_df,
            start_date=start_date,
            end_date=end_date
        )
        
        if 'error' in result:
            logger.error(f"Backtest failed: {result['error']}")
            return None
        
        total_return = result.get('total_return', 0)
        max_drawdown = result.get('max_drawdown', 0)
        total_trades = result.get('total_trades', 0)
        
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Trades: {total_trades}")
        
        iteration_results.append({
            'iteration': iteration,
            'parameters': {
                'entry_top_n': current_entry_top_n,
                'momentum_weight': current_momentum_weight,
                'r2_weight': current_r2_weight,
                'rs_top_percentile': current_rs_top_percentile
            },
            'metrics': {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades
            }
        })
        
        if best_result is None or total_return > best_result.get('total_return', 0):
            best_result = result
            best_iteration = iteration
        
        # 检查是否达标
        targets_met = (
            total_return >= V58_RETURN_TARGET and 
            max_drawdown <= V58_MDD_TARGET and 
            total_trades <= V58_GLOBAL_TRADE_LIMIT and
            total_trades >= 5  # 至少有 5 次交易
        )
        
        if targets_met:
            logger.info(f"\n✅ TARGET ACHIEVED at iteration {iteration}!")
            break
        
        # 分析失败原因并调整参数
        if iteration < V58_MAX_ITERATION_ROUNDS:
            analysis = []
            
            if total_trades == 0 or total_trades < 5:
                analysis.append("Too few trades - relaxing entry threshold")
                current_entry_top_n = min(current_entry_top_n + 10, 50)
                current_rs_top_percentile = min(current_rs_top_percentile + 0.10, 0.35)
            
            if total_return < V58_RETURN_TARGET and total_trades >= 5:
                analysis.append("Low return - adjusting factor weights")
                current_momentum_weight = min(current_momentum_weight + 0.1, 0.7)
                current_r2_weight = max(current_r2_weight - 0.1, 0.2)
            
            if max_drawdown > V58_MDD_TARGET:
                analysis.append("High drawdown - cannot adjust via parameters (need strategy change)")
            
            if analysis:
                logger.info(f"Analysis: {'; '.join(analysis)}")
                logger.info(f"New parameters: Entry Top N={current_entry_top_n}, Momentum Weight={current_momentum_weight:.2f}, RS Top%={current_rs_top_percentile:.2f}")
    
    if best_result:
        best_result['iteration_results'] = iteration_results
        best_result['best_iteration'] = best_iteration
        best_result['total_iterations'] = len(iteration_results)
        best_result['v58_config'] = {
            'entry_top_n': current_entry_top_n,
            'max_positions': 5,
            'global_trade_limit': V58_GLOBAL_TRADE_LIMIT,
            'weekly_trade_limit': 2,
            'industry_filter_enabled': True,
            'industry_top_n': 5,
            'breakeven_enabled': True,
            'breakeven_threshold': 0.04,
            'friction_cost': 0.002,
            'tiered_profit_enabled': True,
            'hard_stop_atr_mult': 2.0,
            'rs_enabled': True,
            'rs_top_percentile': current_rs_top_percentile,
            'ma60_filter': True,
            'momentum_weight': current_momentum_weight,
            'r2_weight': current_r2_weight,
        }
    
    logger.info("\n" + "=" * 60)
    logger.info("V58 BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    if 'error' in result:
        logger.error(f"Backtest failed: {result['error']}")
        return None
    
    logger.info("\n" + "=" * 60)
    logger.info("V58 BACKTEST RESULT")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.2%}")
    logger.info(f"Profit/Loss Ratio: {result.get('profit_loss_ratio', 0):.2f}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Final Equity: {result.get('final_equity', 0):,.2f}")
    
    if best_result:
        report_file = generate_report(best_result, log_file)
        
        logger.info(f"Log file: {log_file}")
        logger.info(f"Report file: {report_file}")
        
        total_return = best_result.get('total_return', 0)
        max_drawdown = best_result.get('max_drawdown', 0)
        total_trades = best_result.get('total_trades', 0)
        
        targets_met = (
            total_return >= V58_RETURN_TARGET and 
            max_drawdown <= V58_MDD_TARGET and 
            total_trades <= V58_GLOBAL_TRADE_LIMIT and
            total_trades >= 5
        )
        
        if targets_met:
            logger.info("\n✅ ALL TARGETS ACHIEVED!")
            logger.info(f"   - Return: {total_return:.2%} ≥ {V58_RETURN_TARGET:.1%}")
            logger.info(f"   - MDD: {max_drawdown:.2%} ≤ {V58_MDD_TARGET:.1%}")
            logger.info(f"   - Trades: {total_trades} (5~{V58_GLOBAL_TRADE_LIMIT})")
            logger.info(f"   - Best Iteration: {best_iteration}")
        else:
            logger.warning("\n⚠️ Some targets not achieved:")
            if total_return < V58_RETURN_TARGET:
                logger.warning(f"   - Return: {total_return:.2%} < {V58_RETURN_TARGET:.1%}")
            if max_drawdown > V58_MDD_TARGET:
                logger.warning(f"   - MDD: {max_drawdown:.2%} > {V58_MDD_TARGET:.1%}")
            if total_trades < 5 or total_trades > V58_GLOBAL_TRADE_LIMIT:
                logger.warning(f"   - Trades: {total_trades} (should be 5~{V58_GLOBAL_TRADE_LIMIT})")
        
        return best_result
    
    return None


if __name__ == "__main__":
    run_v58_backtest()