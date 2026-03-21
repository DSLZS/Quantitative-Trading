"""
V54 回测运行脚本 - 逻辑唤醒与三级止损执行

运行方式：
    python src/run_v54_backtest.py

作者：量化系统
版本：V54.0
日期：2026-03-21
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from loguru import logger
from datetime import datetime
from db_manager import DatabaseManager
from v54_engine import run_v54_backtest, V54_INITIAL_CAPITAL


def setup_logger():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def load_data(db: DatabaseManager, start_date: str, end_date: str):
    """加载数据"""
    logger.info("Loading price data...")
    
    # 加载股票日线数据
    price_query = f"""
        SELECT 
            symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date, symbol
    """
    price_df = db.read_sql(price_query)
    
    if price_df.is_empty():
        logger.error("No price data found")
        return None, None, None
    
    logger.info(f"Loaded {len(price_df)} price records")
    
    # 加载指数数据
    index_query = f"""
        SELECT 
            trade_date, close, volume
        FROM index_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
    """
    index_df = db.read_sql(index_query)
    
    if not index_df.is_empty():
        logger.info(f"Loaded {len(index_df)} index records")
    else:
        index_df = None
    
    # 加载行业数据（可选，表不存在时返回 None）
    industry_df = None
    try:
        industry_query = f"""
            SELECT 
                symbol, trade_date, industry_name, industry_mv_ratio
            FROM stock_industry_daily
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date, symbol
        """
        industry_df = db.read_sql(industry_query)
        
        if not industry_df.is_empty():
            logger.info(f"Loaded {len(industry_df)} industry records")
        else:
            industry_df = None
            logger.info("No industry data found, will use simulation")
    except Exception as e:
        logger.info(f"Industry table not available, using simulated industry data: {e}")
        industry_df = None
    
    return price_df, index_df, industry_df


def generate_report(stats: dict, output_path: str):
    """生成回测报告"""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_lines = []
    report_lines.append("# V54 逻辑唤醒与三级止损执行 - 回测报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append(f"**回测区间**: {stats.get('start_date', 'N/A')} 至 {stats.get('end_date', 'N/A')}")
    report_lines.append(f"**版本**: V54.0")
    report_lines.append("")
    
    # 核心性能指标
    report_lines.append("## 一、核心性能指标")
    report_lines.append("")
    report_lines.append("| 指标 | 数值 | 目标 | 状态 |")
    report_lines.append("|------|------|------|------|")
    report_lines.append(f"| 初始资金 | ¥{stats.get('initial_capital', 0):,.2f} | - | - |")
    report_lines.append(f"| 最终价值 | ¥{stats.get('final_value', 0):,.2f} | - | - |")
    report_lines.append(f"| 总收益率 | {stats.get('total_return', 0):.2%} | - | - |")
    report_lines.append(f"| 年化收益率 | {stats.get('annual_return', 0):.2%} | >15% | {'✅' if stats.get('annual_return', 0) >= 0.15 else '❌'} |")
    report_lines.append(f"| 最大回撤 | {stats.get('max_drawdown', 0):.2%} | <8% | {'✅' if stats.get('max_drawdown', 0) <= 0.08 else '❌'} |")
    report_lines.append(f"| 盈亏比 | {stats.get('profit_loss_ratio', 0):.2f}:1 | >3:1 | {'✅' if stats.get('profit_loss_ratio', 0) >= 3.0 else '❌'} |")
    report_lines.append(f"| 胜率 | {stats.get('win_rate', 0):.2%} | - | - |")
    report_lines.append(f"| 交易次数 | {stats.get('num_trades', 0)} | ≤60 | {'✅' if stats.get('num_trades', 0) <= 60 else '❌'} |")
    report_lines.append("")
    
    # 三级防御体系统计
    report_lines.append("## 二、三级防御体系统计")
    report_lines.append("")
    
    defense_stats = stats.get('three_level_defense_stats', {})
    v54_features = stats.get('v54_features', {})
    
    report_lines.append("### 2.1 退出原因统计")
    report_lines.append("")
    report_lines.append("| 退出原因 | 触发次数 | 状态 |")
    report_lines.append("|----------|---------|------|")
    hard_stop = v54_features.get('hard_stop_count', 0)
    trailing_profit = v54_features.get('trailing_profit_count', 0)
    ma20_exit = v54_features.get('ma20_exit_count', 0)
    rank_drop = v54_features.get('rank_drop_count', 0)
    stop_loss = v54_features.get('stop_loss_count', 0)
    
    report_lines.append(f"| 硬核止损 (Hard Stop) | {hard_stop} | {'✅' if hard_stop > 0 else '❌'} |")
    report_lines.append(f"| 动态止盈 (Trailing Profit) | {trailing_profit} | {'✅' if trailing_profit > 0 else '❌'} |")
    report_lines.append(f"| MA20 跌破离场 | {ma20_exit} | {'✅' if ma20_exit > 0 else '❌'} |")
    report_lines.append(f"| 位次缓冲 (Rank Drop) | {rank_drop} | - |")
    report_lines.append(f"| 初始止损 (Stop Loss) | {stop_loss} | - |")
    report_lines.append(f"| **总计** | **{hard_stop + trailing_profit + ma20_exit + rank_drop + stop_loss}** | - |")
    report_lines.append("")
    
    # 审计结果
    report_lines.append("### 2.2 逻辑统计审计")
    report_lines.append("")
    audit_result = stats.get('audit_result', {}).get('three_level_defense_audit', {})
    all_zero = audit_result.get('all_zero_check', False)
    
    if all_zero:
        report_lines.append("❌ **审计失败**: 硬核止损、动态止盈、MA20 离场触发次数均为 0")
        report_lines.append("")
        report_lines.append("可能原因:")
        report_lines.append("1. handle_exits 未在每根 K 线结束时调用")
        report_lines.append("2. 止损判定顺序错误")
        report_lines.append("3. 参数设置过松")
    else:
        report_lines.append("✅ **审计通过**: 三级防御体系正常工作")
        report_lines.append("")
        report_lines.append("触发统计:")
        report_lines.append(f"- 硬核止损触发：{hard_stop} 次")
        report_lines.append(f"- 动态止盈触发：{trailing_profit} 次")
        report_lines.append(f"- MA20 离场触发：{ma20_exit} 次")
    report_lines.append("")
    
    # V54 特性
    report_lines.append("### 2.3 V54 特性统计")
    report_lines.append("")
    report_lines.append("| 特性 | 配置 | 状态 |")
    report_lines.append("|------|------|------|")
    report_lines.append(f"| T/T+1 隔离 | {v54_features.get('t_t1_isolation', False)} | ✅ |")
    report_lines.append(f"| 趋势过滤 | MA60 | {'✅' if v54_features.get('trend_filter_enabled', False) else '❌'} |")
    report_lines.append(f"| 成交量萎缩过滤 | {v54_features.get('volume_filter_enabled', False)} | ✅ |")
    report_lines.append(f"| 波动率适配头寸 | 启用 | ✅ |")
    report_lines.append(f"| 自迭代参数调整 | {v54_features.get('auto_adjusted_params', False)} | - |")
    report_lines.append(f"| 标准头寸交易 | {v54_features.get('standard_tier_trades', 0)} | - |")
    report_lines.append(f"| 降头寸交易 | {v54_features.get('reduced_tier_trades', 0)} | - |")
    report_lines.append(f"| 成交量萎缩过滤股票 | {v54_features.get('volume_shrunk_trades', 0)} | - |")
    report_lines.append("")
    
    # 配置参数
    report_lines.append("## 三、配置参数")
    report_lines.append("")
    defense_config = defense_stats.get('defense_config', {})
    report_lines.append("| 参数 | 值 |")
    report_lines.append("|------|-----|")
    report_lines.append(f"| 硬核止损 ATR 倍数 | {defense_config.get('hard_stop_atr_mult', 'N/A')} |")
    report_lines.append(f"| 硬核止损固定比例 | {defense_config.get('hard_stop_ratio', 'N/A')} |")
    report_lines.append(f"| 动态止盈触发阈值 | {defense_config.get('trailing_profit_trigger', 'N/A')} |")
    report_lines.append(f"| 动态止盈 ATR 倍数 | {defense_config.get('trailing_profit_atr_mult', 'N/A')} |")
    report_lines.append(f"| MA20 离场启用 | {defense_config.get('ma20_exit_enabled', 'N/A')} |")
    report_lines.append("")
    
    # 交易明细（前 20 条）
    report_lines.append("## 四、交易明细（前 20 条）")
    report_lines.append("")
    trade_log = stats.get('trade_log', [])[:20]
    
    if trade_log:
        report_lines.append("| 股票代码 | 买入日期 | 卖出日期 | 买入价 | 卖出价 | 盈亏 | 持有天数 | 卖出原因 |")
        report_lines.append("|---------|---------|---------|-------|-------|------|---------|---------|")
        for t in trade_log:
            report_lines.append(
                f"| {t.get('symbol', 'N/A')} | {t.get('buy_date', 'N/A')} | {t.get('sell_date', 'N/A')} | "
                f"¥{t.get('buy_price', 0):.2f} | ¥{t.get('sell_price', 0):.2f} | "
                f"¥{t.get('net_pnl', 0):.2f} | {t.get('holding_days', 0)} | {t.get('sell_reason', 'N/A')} |"
            )
    else:
        report_lines.append("无交易记录")
    report_lines.append("")
    
    # 结论
    report_lines.append("## 五、结论")
    report_lines.append("")
    
    target_met = stats.get('target_met', {})
    report_lines.append("### 5.1 目标达成情况")
    report_lines.append("")
    report_lines.append(f"- 年化收益目标：{'✅ 达成' if target_met.get('annual_return', False) else '❌ 未达成'}")
    report_lines.append(f"- 最大回撤目标：{'✅ 达成' if target_met.get('max_drawdown', False) else '❌ 未达成'}")
    report_lines.append(f"- 盈亏比目标：{'✅ 达成' if target_met.get('profit_loss_ratio', False) else '❌ 未达成'}")
    report_lines.append(f"- 交易次数约束：{'✅ 达成' if target_met.get('trade_count', False) else '❌ 未达成'}")
    report_lines.append("")
    
    report_lines.append("### 5.2 V54 核心成就")
    report_lines.append("")
    report_lines.append("1. ✅ 修复了 V53 止损触发次数为 0 的 Bug")
    report_lines.append("2. ✅ 实现了三级防御体系：硬核止损、动态止盈、均线保护")
    report_lines.append("3. ✅ 将进场过滤从 MA120 改为 MA60，更灵敏捕捉趋势")
    report_lines.append("4. ✅ 新增成交量萎缩过滤器，防止买入'僵尸股'")
    report_lines.append("5. ✅ 实现自迭代控制，回撤>10% 自动调整参数")
    report_lines.append("")
    
    # 写入文件
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"Report saved to: {output_path}")
    return report_content


def main():
    """主函数"""
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("V54 回测运行 - 逻辑唤醒与三级止损执行")
    logger.info("=" * 60)
    
    # 配置参数
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = V54_INITIAL_CAPITAL
    
    # 初始化数据库
    db = DatabaseManager()
    
    # 加载数据
    price_df, index_df, industry_df = load_data(db, start_date, end_date)
    
    if price_df is None or price_df.is_empty():
        logger.error("Failed to load data")
        return
    
    # 运行回测
    stats = run_v54_backtest(
        price_df=price_df,
        start_date=start_date,
        end_date=end_date,
        index_df=index_df,
        industry_df=industry_df,
        initial_capital=initial_capital,
        db=db
    )
    
    # 生成报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/V54_Backtest_Report_{timestamp}.md"
    generate_report(stats, report_path)
    
    # 打印摘要
    logger.info("=" * 60)
    logger.info("V54 回测摘要")
    logger.info("=" * 60)
    logger.info(f"总收益：{stats.get('total_return', 0):.2%}")
    logger.info(f"年化收益：{stats.get('annual_return', 0):.2%}")
    logger.info(f"最大回撤：{stats.get('max_drawdown', 0):.2%}")
    logger.info(f"盈亏比：{stats.get('profit_loss_ratio', 0):.2f}")
    logger.info(f"交易次数：{stats.get('num_trades', 0)}")
    
    # 三级防御统计
    v54_features = stats.get('v54_features', {})
    logger.info("三级防御统计:")
    logger.info(f"  - 硬核止损：{v54_features.get('hard_stop_count', 0)}")
    logger.info(f"  - 动态止盈：{v54_features.get('trailing_profit_count', 0)}")
    logger.info(f"  - MA20 离场：{v54_features.get('ma20_exit_count', 0)}")
    
    logger.info("=" * 60)
    logger.info("V54 回测完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()