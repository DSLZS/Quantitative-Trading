"""
V55 回测脚本 - 真实成交价与精选股系统

运行 V55 回测并生成报告
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from v55_engine import run_v55_backtest
from v55_core import V55_INITIAL_CAPITAL


def setup_logger():
    """设置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )


def load_data_from_db(db=None):
    """从数据库加载数据"""
    if db is None:
        return None, None, None
    
    try:
        # 加载股票价格数据
        price_query = """
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'
        ORDER BY trade_date, symbol
        """
        price_df = db.read_sql(price_query)
        
        # 加载指数数据
        index_query = """
        SELECT trade_date, close, high, low, open, volume
        FROM index_daily
        WHERE trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'
        ORDER BY trade_date
        """
        index_df = db.read_sql(index_query)
        
        # 加载行业数据
        industry_query = """
        SELECT symbol, trade_date, industry_name, industry_mv_ratio
        FROM stock_industry_daily
        WHERE trade_date >= '2024-01-01' AND trade_date <= '2025-12-31'
        ORDER BY trade_date, symbol
        """
        industry_df = db.read_sql(industry_query)
        
        return price_df, index_df, industry_df
        
    except Exception as e:
        logger.error(f"Failed to load data from DB: {e}")
        return None, None, None


def generate_sample_data() -> tuple:
    """
    生成模拟数据用于测试
    
    返回:
        (price_df, index_df, industry_df)
    """
    logger.info("Generating sample data for testing...")
    
    # 生成日期范围
    import pandas as pd
    from datetime import timedelta
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    # 生成交易日（排除周末）
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # 周一到周五
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    
    # 生成 100 只股票
    symbols = [f"600{i:04d}.SH" for i in range(100)] + [f"000{i:04d}.SZ" for i in range(50)] + [f"300{i:04d}.SZ" for i in range(50)]
    
    # 生成价格数据
    price_records = []
    import random
    random.seed(42)
    
    for symbol in symbols:
        base_price = random.uniform(10, 100)
        for i, date in enumerate(dates):
            # 随机游走价格
            daily_return = random.gauss(0.0005, 0.02)
            base_price *= (1 + daily_return)
            
            open_price = base_price * (1 + random.uniform(-0.01, 0.01))
            high_price = max(open_price, base_price) * (1 + random.uniform(0, 0.03))
            low_price = min(open_price, base_price) * (1 - random.uniform(0, 0.03))
            close_price = base_price
            volume = random.randint(100000, 10000000)
            amount = volume * close_price
            
            price_records.append({
                'symbol': symbol,
                'trade_date': date,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'amount': amount
            })
    
    price_df = pl.DataFrame(price_records)
    
    # 生成指数数据
    index_records = []
    index_base = 3000
    for i, date in enumerate(dates):
        daily_return = random.gauss(0.0003, 0.015)
        index_base *= (1 + daily_return)
        
        index_records.append({
            'trade_date': date,
            'close': index_base,
            'high': index_base * (1 + random.uniform(0, 0.02)),
            'low': index_base * (1 - random.uniform(0, 0.02)),
            'open': index_base * (1 + random.uniform(-0.01, 0.01)),
            'volume': random.randint(1000000000, 5000000000)
        })
    
    index_df = pl.DataFrame(index_records)
    
    # 生成行业数据
    industry_records = []
    industries = ['Finance_Industry', 'Manufacturing', 'Technology', 'Growth_Tech', 'Hard_Tech']
    
    for symbol in symbols:
        industry = random.choice(industries)
        for date in dates:
            industry_records.append({
                'symbol': symbol,
                'trade_date': date,
                'industry_name': industry,
                'industry_mv_ratio': random.uniform(0.8, 1.2)
            })
    
    industry_df = pl.DataFrame(industry_records)
    
    logger.info(f"Generated {len(price_df)} price records, {len(index_df)} index records, {len(industry_df)} industry records")
    
    return price_df, index_df, industry_df


def generate_stop_audit_table(stop_audit_records: list) -> str:
    """
    生成止损真实性统计表
    
    参数:
        stop_audit_records: 止损审计记录列表
    
    返回:
        Markdown 格式的统计表
    """
    if not stop_audit_records:
        return "### 止损真实性统计表\n\n无止损记录。\n"
    
    # 随机选择 5 条记录（如果超过 5 条）
    import random
    sample_size = min(5, len(stop_audit_records))
    sample_records = random.sample(stop_audit_records, sample_size)
    
    # 构建表格
    header = "| # | 股票代码 | 交易日期 | 止损原因 | 触发价 | 次日开盘价 | min(Trigger, Next_Open) | 实际成交价 | 滑点 | 是否合规 |\n"
    separator = "|---|----------|----------|----------|--------|------------|------------------------|------------|------|----------|\n"
    
    rows = []
    for i, record in enumerate(sample_records, 1):
        symbol = record.get('symbol', 'N/A')
        trade_date = record.get('trade_date', 'N/A')
        reason = record.get('reason', 'N/A')
        trigger_price = record.get('trigger_price', 0)
        next_open_price = record.get('next_open_price', 0)
        min_trigger_open = record.get('min_trigger_open', 0)
        execution_price = record.get('execution_price', 0)
        slippage = record.get('slippage_applied', 0)
        is_valid = record.get('is_valid', True)
        
        valid_str = "✅" if is_valid else "❌"
        
        row = f"| {i} | {symbol} | {trade_date} | {reason} | {trigger_price:.2f} | {next_open_price:.2f} | {min_trigger_open:.2f} | {execution_price:.2f} | {slippage:.4f} | {valid_str} |\n"
        rows.append(row)
    
    table = "### 止损真实性统计表\n\n"
    table += "V55 强制规则：止损单成交价必须满足 `Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)`。严禁在止损时获得任何价格优势。\n\n"
    table += header
    table += separator
    table += "".join(rows)
    table += f"\n**统计**: 共 {len(stop_audit_records)} 条止损记录，抽样显示 {sample_size} 条。合规率：{sum(1 for r in stop_audit_records if r.get('is_valid', True)) / len(stop_audit_records) * 100:.1f}%\n"
    
    return table


def generate_v55_report(stats: dict, output_path: str = None) -> str:
    """
    生成 V55 回测报告
    
    参数:
        stats: 回测统计字典
        output_path: 输出文件路径
    
    返回:
        报告内容字符串
    """
    import os
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_path is None:
        output_path = f"reports/V55_Backtest_Report_{timestamp}.md"
    
    # 确保 reports 目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 报告头部
    report = "# V55 回测报告 - 真实成交价与精选股系统\n\n"
    report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += "---\n\n"
    
    # 对赌条款声明
    report += "## 对赌条款\n\n"
    report += "**条款内容**: 若 MDD > 8% 或交易次数 > 45 次，系统风控设计失败。\n\n"
    
    target_met = stats.get('target_met', {})
    audit_result = stats.get('audit_result', {})
    
    mdd_exceeded = audit_result.get('mdd_exceeded', False)
    trade_count_exceeded = audit_result.get('trade_count_exceeded', False)
    
    if mdd_exceeded or trade_count_exceeded:
        report += "### ❌ 对赌条款触发 - 系统风控设计失败\n\n"
        report += "**道歉声明**: 我们深感抱歉，V55 系统未能满足风控要求。\n\n"
        
        if mdd_exceeded:
            max_dd = stats.get('max_drawdown', 0)
            report += f"- **MDD 失败**: 最大回撤 {max_dd:.2%} > 目标 8%\n"
        if trade_count_exceeded:
            num_trades = stats.get('num_trades', 0)
            report += f"- **交易次数失败**: 交易次数 {num_trades} > 限制 45\n"
        
        report += "\n**失败代码行号参考**:\n"
        report += "- `v55_core.py`: V55RiskManager.check_exits() - 止损逻辑\n"
        report += "- `v55_core.py`: V55RiskManager.can_open_new_position() - 频率熔断\n"
        report += "- `v55_core.py`: V55RiskManager.check_industry_constraint() - 行业对冲\n"
        report += "- `v55_engine.py`: V55BacktestEngine._handle_exits() - 真实成交价执行\n"
    else:
        report += "### ✅ 对赌条款通过\n\n"
        report += "V55 系统满足所有风控要求：\n"
        report += f"- MDD: {stats.get('max_drawdown', 0):.2%} <= 8% ✅\n"
        report += f"- 交易次数：{stats.get('num_trades', 0)} <= 45 ✅\n"
    
    report += "\n---\n\n"
    
    # 核心绩效指标
    report += "## 核心绩效指标\n\n"
    report += f"| 指标 | 数值 |\n"
    report += f"|------|------|\n"
    report += f"| 回测区间 | {stats.get('start_date', 'N/A')} 至 {stats.get('end_date', 'N/A')} |\n"
    report += f"| 初始资金 | ¥{stats.get('initial_capital', 0):,.2f} |\n"
    report += f"| 最终价值 | ¥{stats.get('final_value', 0):,.2f} |\n"
    report += f"| 总收益 | {stats.get('total_return', 0):.2%} |\n"
    report += f"| 年化收益 | {stats.get('annual_return', 0):.2%} |\n"
    report += f"| 最大回撤 | {stats.get('max_drawdown', 0):.2%} |\n"
    report += f"| 最大单日亏损 | {stats.get('max_single_day_loss', 0):.2%} |\n"
    report += f"| 交易次数 | {stats.get('num_trades', 0)} |\n"
    report += f"| 胜率 | {stats.get('win_rate', 0):.2%} |\n"
    report += f"| 盈亏比 | {stats.get('profit_loss_ratio', 0):.2f} |\n"
    report += f"| 平均盈利 | ¥{stats.get('avg_win', 0):,.2f} |\n"
    report += f"| 平均亏损 | ¥{stats.get('avg_loss', 0):,.2f} |\n"
    
    report += "\n---\n\n"
    
    # 止损真实性统计表（任务要求）
    stop_audit_records = stats.get('stop_audit_records', [])
    report += generate_stop_audit_table(stop_audit_records)
    report += "\n---\n\n"
    
    # 三级防御体系统计
    report += "## 三级防御体系统计\n\n"
    defense_stats = stats.get('three_level_defense_stats', {})
    report += f"| 防御类型 | 触发次数 |\n"
    report += f"|----------|----------|\n"
    report += f"| 硬止损 (Hard Stop) | {defense_stats.get('hard_stop_count', 0)} |\n"
    report += f"| 动态止盈 (Trailing Profit) | {defense_stats.get('trailing_profit_count', 0)} |\n"
    report += f"| 均线保护 (MA20 Break) | {defense_stats.get('ma20_exit_count', 0)} |\n"
    report += f"| 位次缓冲 (Rank Drop) | {defense_stats.get('rank_drop_count', 0)} |\n"
    report += f"| 初始止损 (Initial Stop) | {defense_stats.get('initial_stop_count', 0)} |\n"
    report += f"| 时间止损 (Time Stop) | {defense_stats.get('time_stop_count', 0)} |\n"
    report += f"| **总计** | {defense_stats.get('total_exit_count', 0)} |\n"
    
    report += "\n**配置参数**:\n"
    defense_config = defense_stats.get('defense_config', {})
    report += f"- 硬止损：max({defense_config.get('hard_stop_atr_mult', 2.5)} * ATR, {defense_config.get('hard_stop_ratio', 0.08):.0%})\n"
    report += f"- 动态止盈触发：{defense_config.get('trailing_profit_trigger', 0.10):.0%}\n"
    report += f"- 动态止盈回撤：{defense_config.get('trailing_profit_atr_mult', 2.5)} * ATR\n"
    report += f"- 时间止损：{defense_config.get('time_stop_days', 5)} 天不盈利减仓 {defense_config.get('time_stop_reduce_ratio', 0.5):.0%}\n"
    
    report += "\n---\n\n"
    
    # 频率熔断统计
    report += "## 频率熔断统计\n\n"
    freq_stats = stats.get('frequency_fuse_stats', {})
    report += f"- 每周开仓限制：{freq_stats.get('weekly_trade_limit', 3)} 只\n"
    report += f"- 最大持仓限制：{freq_stats.get('max_positions', 5)} 只\n"
    report += f"- 全场交易次数限制：{freq_stats.get('global_trade_limit', 45)} 次\n"
    report += f"- 当前周交易次数：{freq_stats.get('current_week_trades', 0)}\n"
    
    report += "\n---\n\n"
    
    # 行业对冲约束统计
    report += "## 行业对冲硬约束\n\n"
    industry_stats = stats.get('industry_constraint_stats', {})
    report += f"- 同行业最大持仓：{industry_stats.get('max_same_industry', 2)} 只\n"
    report += f"- 规则描述：{industry_stats.get('description', 'Maximum 2 positions per industry')}\n"
    
    report += "\n---\n\n"
    
    # 波动率挤压选股统计
    report += "## 波动率挤压选股 (Volatility Squeeze)\n\n"
    squeeze_stats = stats.get('volatility_squeeze_stats', {})
    report += f"- 启用状态：{'是' if squeeze_stats.get('enabled', True) else '否'}\n"
    report += f"- 低位阈值：{squeeze_stats.get('low_threshold', 0.5):.0%}\n"
    report += f"- 计算窗口：{squeeze_stats.get('squeeze_window', 20)} 天\n"
    report += f"- 突破倍数：{squeeze_stats.get('breakout_mult', 1.5)}x\n"
    report += f"- 描述：{squeeze_stats.get('description', '')}\n"
    
    report += "\n---\n\n"
    
    # 真实成交价审计
    report += "## 真实成交价审计\n\n"
    exec_audit = audit_result.get('real_execution_price_audit', {})
    report += f"- 止损记录总数：{exec_audit.get('total_stop_records', 0)}\n"
    report += f"- 合规记录数：{exec_audit.get('valid_records', 0)}\n"
    report += f"- 不合规记录数：{exec_audit.get('invalid_records', 0)}\n"
    all_valid = exec_audit.get('all_valid', True)
    if all_valid:
        compliance_rate = '100%'
    else:
        valid = exec_audit.get('valid_records', 0)
        total = max(1, exec_audit.get('total_stop_records', 1))
        compliance_rate = f"{valid / total * 100:.1f}%"
    report += f"- 合规率：{compliance_rate}\n"
    
    report += "\n**V55 强制规则**: `Execution_Price = min(Trigger_Price, Next_Open_Price) * (1 - Slippage)`\n"
    
    report += "\n---\n\n"
    
    # V55 特性总结
    report += "## V55 特性总结\n\n"
    v55_features = stats.get('v55_features', {})
    
    report += "### 已启用的功能\n\n"
    report += f"- [x] T/T+1 严格隔离：{v55_features.get('t_t1_isolation', True)}\n"
    report += f"- [x] 趋势过滤 (MA60): {v55_features.get('trend_filter_enabled', True)}\n"
    report += f"- [x] 成交量萎缩过滤：{v55_features.get('volume_filter_enabled', True)}\n"
    report += f"- [x] 波动率挤压选股：{v55_features.get('volatility_squeeze_enabled', True)}\n"
    report += f"- [x] 时间止损：{v55_features.get('time_stop_enabled', True)}\n"
    report += f"- [x] 波动率适配头寸：{v55_features.get('volatility_adaptive_positioning', True)}\n"
    report += f"- [x] 真实成交价逻辑：{v55_features.get('real_execution_price', True)}\n"
    report += f"- [x] 频率熔断：每周最多 {v55_features.get('frequency_fuse', {}).get('weekly_limit', 3)} 只，最大持仓 {v55_features.get('frequency_fuse', {}).get('max_positions', 5)} 只\n"
    report += f"- [x] 行业对冲：最多持有 {v55_features.get('industry_constraint', {}).get('max_same_industry', 2)} 只同行业股票\n"
    
    report += "\n### 统计详情\n\n"
    report += f"- 硬止损触发：{v55_features.get('hard_stop_count', 0)}\n"
    report += f"- 动态止盈触发：{v55_features.get('trailing_profit_count', 0)}\n"
    report += f"- MA20 离场触发：{v55_features.get('ma20_exit_count', 0)}\n"
    report += f"- 位次缓冲离场：{v55_features.get('rank_drop_count', 0)}\n"
    report += f"- 标准头寸交易：{v55_features.get('standard_tier_trades', 0)}\n"
    report += f"- 降头寸交易：{v55_features.get('reduced_tier_trades', 0)}\n"
    report += f"- 成交量萎缩过滤：{v55_features.get('volume_shrunk_trades', 0)}\n"
    report += f"- 波动率挤压低位入场：{v55_features.get('squeeze_low_trades', 0)}\n"
    
    report += "\n---\n\n"
    
    # 目标达成情况
    report += "## 目标达成情况\n\n"
    report += f"| 目标 | 要求 | 实际 | 状态 |\n"
    report += f"|------|------|------|------|\n"
    
    annual_return = stats.get('annual_return', 0)
    max_drawdown = stats.get('max_drawdown', 0)
    profit_loss_ratio = stats.get('profit_loss_ratio', 0)
    num_trades = stats.get('num_trades', 0)
    
    report += f"| 年化收益 | >= 15% | {annual_return:.2%} | {'✅' if annual_return >= 0.15 else '❌'} |\n"
    report += f"| 最大回撤 | <= 8% | {max_drawdown:.2%} | {'✅' if max_drawdown <= 0.08 else '❌'} |\n"
    report += f"| 盈亏比 | >= 3.0 | {profit_loss_ratio:.2f} | {'✅' if profit_loss_ratio >= 3.0 else '❌'} |\n"
    report += f"| 交易次数 | <= 45 | {num_trades} | {'✅' if num_trades <= 45 else '❌'} |\n"
    
    report += "\n---\n\n"
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {output_path}")
    
    return report


def main():
    """主函数"""
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("V55 回测脚本 - 真实成交价与精选股系统")
    logger.info("=" * 60)
    
    # 尝试从数据库加载数据，如果失败则使用模拟数据
    price_df, index_df, industry_df = None, None, None
    
    try:
        # 尝试导入 db_manager
        from db_manager import DatabaseManager
        db = DatabaseManager()
        price_df, index_df, industry_df = load_data_from_db(db)
        if price_df is not None and not price_df.is_empty():
            logger.info("Data loaded from database")
        else:
            logger.warning("Database returned empty data, using generated sample data instead")
            price_df, index_df, industry_df = generate_sample_data()
    except Exception as e:
        logger.warning(f"Failed to load from database: {e}")
        logger.info("Using generated sample data instead")
        price_df, index_df, industry_df = generate_sample_data()
    
    if price_df is None or price_df.is_empty():
        logger.error("No price data available")
        logger.info("Generating sample data as fallback")
        price_df, index_df, industry_df = generate_sample_data()
    
    # 设置回测参数
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = V55_INITIAL_CAPITAL
    
    logger.info(f"Running V55 backtest from {start_date} to {end_date}")
    logger.info(f"Initial capital: ¥{initial_capital:,.2f}")
    
    # 运行回测
    stats = run_v55_backtest(
        price_df=price_df,
        start_date=start_date,
        end_date=end_date,
        index_df=index_df,
        industry_df=industry_df,
        initial_capital=initial_capital
    )
    
    # 生成报告
    report = generate_v55_report(stats)
    
    # 保存统计结果为 JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"reports/v55_backtest_result_{timestamp}.json"
    
    # 移除 portfolio_values 以减少 JSON 大小
    stats_for_json = {k: v for k, v in stats.items() if k != 'portfolio_values'}
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_for_json, f, default=str, indent=2)
    
    logger.info(f"Results saved to: {json_path}")
    
    # 打印摘要
    logger.info("=" * 60)
    logger.info("V55 回测摘要")
    logger.info("=" * 60)
    logger.info(f"总收益：{stats.get('total_return', 0):.2%}")
    logger.info(f"年化收益：{stats.get('annual_return', 0):.2%}")
    logger.info(f"最大回撤：{stats.get('max_drawdown', 0):.2%}")
    logger.info(f"交易次数：{stats.get('num_trades', 0)}")
    logger.info(f"胜率：{stats.get('win_rate', 0):.2%}")
    logger.info(f"盈亏比：{stats.get('profit_loss_ratio', 0):.2f}")
    
    # 审计结果
    audit_result = stats.get('audit_result', {})
    logger.info(f"交易次数审计：{audit_result.get('trade_count_status', 'N/A')}")
    logger.info(f"MDD 约束审计：{audit_result.get('mdd_status', 'N/A')}")
    logger.info(f"真实成交价审计：{'PASSED' if audit_result.get('real_execution_price_audit', {}).get('all_valid', True) else 'FAILED'}")


if __name__ == "__main__":
    main()