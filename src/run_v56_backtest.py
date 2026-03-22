"""
V56 Backtest Runner - 逻辑解封与真实趋势捕捉

【V56 核心改进】
1. 数据库容错：若 stock_industry_daily 表缺失，自动激活 IndustryLoader 模拟分类逻辑
2. RS 强度选股：只买入 RS 排名前 10% 且放量突破 20 日均线的股票
3. 保本止损：浮盈超过 4% 后，硬止损线上移至"买入成本价 + 0.5%"
4. 阶梯止盈：浮盈 10% 减仓 30%，浮盈 20% 减仓 40%
5. 强制多轮迭代：至少 5 轮独立参数扫描
6. 真实成交价：Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)

作者：量化系统
版本：V56.0
日期：2026-03-21
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

import polars as pl
from loguru import logger
from v56_engine import V56BacktestEngine
from db_manager import DatabaseManager


def setup_logger():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def generate_v56_report(result: dict, output_path: str):
    """生成 V56 回测报告"""
    
    report_lines = []
    
    # Header
    report_lines.append("# V56 回测报告 - 逻辑解封与真实趋势捕捉")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # V56 核心改进
    report_lines.append("## V56 核心改进")
    report_lines.append("")
    report_lines.append("1. **数据库容错**: 若 stock_industry_daily 表缺失，自动激活 IndustryLoader 模拟分类逻辑")
    report_lines.append("2. **RS 强度选股**: 只买入 RS 排名前 10% 且放量突破 20 日均线的股票")
    report_lines.append("3. **保本止损**: 浮盈超过 4% 后，硬止损线上移至'买入成本价 + 0.5%'")
    report_lines.append("4. **阶梯止盈**: 浮盈 10% 减仓 30%，浮盈 20% 减仓 40%")
    report_lines.append("5. **强制多轮迭代**: 至少 5 轮独立参数扫描")
    report_lines.append("6. **真实成交价**: Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)")
    report_lines.append("")
    
    # 回测概览
    report_lines.append("## 回测概览")
    report_lines.append("")
    report_lines.append("| 指标 | 数值 |")
    report_lines.append("|------|------|")
    report_lines.append(f"| 初始资金 | {result.get('initial_value', 0):,.2f} |")
    report_lines.append(f"| 最终价值 | {result.get('final_value', 0):,.2f} |")
    report_lines.append(f"| 总收益率 | {result.get('total_return', 0):.2%} |")
    report_lines.append(f"| 年化收益 | {result.get('annual_return', 0):.2%} |")
    report_lines.append(f"| 最大回撤 | {result.get('max_drawdown', 0):.2%} |")
    report_lines.append(f"| 夏普比率 | {result.get('sharpe_ratio', 0):.3f} |")
    report_lines.append(f"| 胜率 | {result.get('win_rate', 0):.2%} |")
    report_lines.append(f"| 盈亏比 | {result.get('profit_loss_ratio', 0):.2f} |")
    report_lines.append(f"| 总交易次数 | {result.get('total_trades', 0)} |")
    report_lines.append("")
    
    # 5 轮参数扫描结果
    report_lines.append("## 5 轮参数扫描结果 (Anti-Laziness Protocol)")
    report_lines.append("")
    
    iteration_results = result.get('iteration_results', [])
    if iteration_results:
        report_lines.append("| 轮次 | 参数配置 | 总收益 | 最大回撤 | 夏普比率 |")
        report_lines.append("|------|----------|--------|----------|----------|")
        
        for ir in iteration_results:
            round_num = ir.get('round', 0)
            desc = ir.get('description', '')
            params = ir.get('parameters', {})
            metrics = ir.get('metrics', {})
            
            param_str = f"ATR={params.get('atr_stop_mult', 0)}, BE={params.get('breakeven_threshold', 0):.0%}, RS={params.get('rs_top_percentile', 0):.0%}"
            
            report_lines.append(
                f"| {round_num} | {desc} | {metrics.get('total_return', 0):.2%} | "
                f"{metrics.get('max_drawdown', 0):.2%} | {metrics.get('sharpe_ratio', 0):.3f} |"
            )
        report_lines.append("")
        
        # 最优迭代
        best = result.get('best_iteration', {})
        if best:
            report_lines.append("### 最优迭代")
            report_lines.append("")
            report_lines.append(f"- **轮次**: {best.get('round')}")
            report_lines.append(f"- **配置**: {best.get('description')}")
            report_lines.append(f"- **参数**: ATR={best.get('parameters', {}).get('atr_stop_mult', 0)}, "
                              f"BE={best.get('parameters', {}).get('breakeven_threshold', 0):.0%}, "
                              f"RS={best.get('parameters', {}).get('rs_top_percentile', 0):.0%}")
            metrics = best.get('metrics', {})
            report_lines.append(f"- **总收益**: {metrics.get('total_return', 0):.2%}")
            report_lines.append(f"- **最大回撤**: {metrics.get('max_drawdown', 0):.2%}")
            report_lines.append(f"- **夏普比率**: {metrics.get('sharpe_ratio', 0):.3f}")
            report_lines.append("")
    
    # 防御体系统计
    report_lines.append("## 防御体系统计")
    report_lines.append("")
    
    defense_stats = result.get('three_level_defense_stats', {})
    if defense_stats:
        report_lines.append("| 退出类型 | 次数 |")
        report_lines.append("|----------|------|")
        report_lines.append(f"| 硬止损 | {defense_stats.get('hard_stop_count', 0)} |")
        report_lines.append(f"| 保本止损 | {defense_stats.get('breakeven_stop_count', 0)} |")
        report_lines.append(f"| 追踪止盈 | {defense_stats.get('trailing_profit_count', 0)} |")
        report_lines.append(f"| MA20 跌破 | {defense_stats.get('ma20_exit_count', 0)} |")
        report_lines.append(f"| 位次下跌 | {defense_stats.get('rank_drop_count', 0)} |")
        report_lines.append(f"| 时间止损 | {defense_stats.get('time_stop_count', 0)} |")
        report_lines.append(f"| 阶梯止盈 | {defense_stats.get('tiered_profit_count', 0)} |")
        report_lines.append("")
    
    # 频率熔断统计
    report_lines.append("## 频率熔断统计")
    report_lines.append("")
    
    freq_stats = result.get('frequency_fuse_stats', {})
    if freq_stats:
        report_lines.append(f"- **每周交易限制**: {freq_stats.get('weekly_trade_limit', 0)}")
        report_lines.append(f"- **全局交易限制**: {freq_stats.get('global_trade_limit', 0)}")
        report_lines.append(f"- **最大持仓数**: {freq_stats.get('max_positions', 0)}")
        report_lines.append("")
    
    # RS 强度选股统计
    report_lines.append("## RS 强度选股统计")
    report_lines.append("")
    
    rs_stats = result.get('rs_strength_stats', {})
    if rs_stats:
        report_lines.append(f"- **RS 选股 enabled**: {rs_stats.get('enabled', False)}")
        report_lines.append(f"- **RS 计算窗口**: {rs_stats.get('rs_window', 20)} 天")
        report_lines.append(f"- **前 N% 选股**: {rs_stats.get('top_percentile', 0):.0%}")
        report_lines.append(f"- **放量突破倍数**: {rs_stats.get('volume_breakout_mult', 1.5)}x")
        report_lines.append(f"- **MA20 突破 required**: {rs_stats.get('ma20_breakout_enabled', True)}")
        report_lines.append("")
    
    # 真实成交价审计
    report_lines.append("## 真实成交价审计")
    report_lines.append("")
    
    stop_audit_records = result.get('stop_audit_records', [])
    if stop_audit_records:
        valid_count = sum(1 for r in stop_audit_records if r.get('is_valid', False))
        report_lines.append(f"- **总审计记录**: {len(stop_audit_records)}")
        report_lines.append(f"- **有效记录**: {valid_count} ({valid_count/len(stop_audit_records)*100:.1f}%)")
        report_lines.append("")
        
        # 显示前 5 条审计记录
        report_lines.append("### 前 5 条审计记录")
        report_lines.append("")
        report_lines.append("| 日期 | 股票代码 | 触发价 | 次日开盘 | 成交价 | 滑点 |")
        report_lines.append("|------|----------|--------|----------|--------|------|")
        
        for record in stop_audit_records[:5]:
            report_lines.append(
                f"| {record.get('trade_date', '')} | {record.get('symbol', '')} | "
                f"{record.get('trigger_price', 0):.2f} | {record.get('next_open_price', 0):.2f} | "
                f"{record.get('execution_price', 0):.2f} | {record.get('slippage_applied', 0):.2%} |"
            )
        report_lines.append("")
    
    # V56 配置
    report_lines.append("## V56 配置参数")
    report_lines.append("")
    
    v56_config = result.get('v56_config', {})
    if v56_config:
        report_lines.append("| 参数 | 值 |")
        report_lines.append("|------|-----|")
        report_lines.append(f"| RS 选股 enabled | {v56_config.get('rs_enabled', False)} |")
        report_lines.append(f"| RS 前 N% | {v56_config.get('rs_top_percentile', 0):.0%} |")
        report_lines.append(f"| 保本止损 enabled | {v56_config.get('breakeven_enabled', False)} |")
        report_lines.append(f"| 保本盈利阈值 | {v56_config.get('breakeven_threshold', 0):.0%} |")
        report_lines.append(f"| 阶梯止盈 enabled | {v56_config.get('tiered_profit_enabled', False)} |")
        report_lines.append(f"| ATR 止损倍数 | {v56_config.get('hard_stop_atr_mult', 0)} |")
        report_lines.append(f"| 每周交易限制 | {v56_config.get('weekly_trade_limit', 0)} |")
        report_lines.append(f"| 全局交易限制 | {v56_config.get('global_trade_limit', 0)} |")
        report_lines.append("")
    
    # 结论
    report_lines.append("## 结论")
    report_lines.append("")
    
    total_return = result.get('total_return', 0)
    max_drawdown = result.get('max_drawdown', 0)
    
    if total_return > 0 and max_drawdown < 0.10:
        report_lines.append("✅ **V56 策略表现良好**：正收益且回撤控制在 10% 以内")
    elif total_return > 0:
        report_lines.append(f"⚠️ **V56 策略盈利但回撤较大**：收益率 {total_return:.2%}，最大回撤 {max_drawdown:.2%}")
    else:
        report_lines.append(f"❌ **V56 策略表现不佳**：收益率 {total_return:.2%}，需要进一步优化参数")
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append(f"*报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to: {output_path}")


def main():
    """主函数"""
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("V56 BACKTEST RUNNER - 逻辑解封与真实趋势捕捉")
    logger.info("=" * 60)
    
    # 初始化数据库
    try:
        db = DatabaseManager()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db = None
    
    # 设置回测参数
    start_date = "2024-01-01"
    end_date = "2025-12-31"
    initial_capital = 100000.0
    
    logger.info(f"Backtest period: {start_date} to {end_date}")
    logger.info(f"Initial capital: {initial_capital:,.2f}")
    
    # 加载数据
    logger.info("Loading price data...")
    try:
        price_df = db.read_sql(f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount
            FROM stock_daily 
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        """)
        logger.info(f"Loaded {len(price_df)} price records")
    except Exception as e:
        logger.error(f"Failed to load price data: {e}")
        price_df = pl.DataFrame()
    
    logger.info("Loading index data (沪深 300)...")
    try:
        index_df = db.read_sql(f"""
            SELECT index_name, trade_date, open, high, low, close, volume
            FROM index_daily 
            WHERE index_name = '沪深 300' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        """)
        logger.info(f"Loaded {len(index_df)} index records")
    except Exception as e:
        logger.error(f"Failed to load index data: {e}")
        index_df = pl.DataFrame()
    
    # 运行回测
    logger.info("Starting V56 backtest...")
    engine = V56BacktestEngine(initial_capital=initial_capital, db=db)
    result = engine.run_backtest(
        price_df=price_df,
        start_date=start_date,
        end_date=end_date,
        index_df=index_df
    )
    
    # 打印结果
    logger.info("=" * 60)
    logger.info("V56 BACKTEST RESULT")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result['total_return']:.2%}")
    logger.info(f"Annual Return: {result['annual_return']:.2%}")
    logger.info(f"Max Drawdown: {result['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {result['sharpe_ratio']:.3f}")
    logger.info(f"Win Rate: {result['win_rate']:.2%}")
    logger.info(f"Profit/Loss Ratio: {result['profit_loss_ratio']:.2f}")
    logger.info(f"Total Trades: {result['total_trades']}")
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(src_dir.parent / "reports" / f"V56_Backtest_Report_{timestamp}.md")
    generate_v56_report(result, report_path)
    
    logger.info("=" * 60)
    logger.info("V56 BACKTEST COMPLETE")
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()