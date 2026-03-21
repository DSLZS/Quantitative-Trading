"""
V51 回测脚本 - 严正审计与回撤攻坚战

运行 V51 回测并生成对比报告
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import polars as pl
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager
from v51_engine import run_v51_backtest, V51_INITIAL_CAPITAL


def load_data_from_db(db: DatabaseManager, start_date: str, end_date: str):
    """从数据库加载数据"""
    logger.info(f"从数据库加载数据：{start_date} 至 {end_date}")
    
    # 加载股票价格数据
    price_query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date, symbol
    """
    price_df = db.read_sql(price_query)
    logger.info(f"加载股票价格数据：{len(price_df)} 条记录")
    
    # 加载指数数据
    index_query = f"""
        SELECT trade_date, close, high, low
        FROM index_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
    """
    try:
        index_df = db.read_sql(index_query)
        logger.info(f"加载指数数据：{len(index_df)} 条记录")
    except Exception as e:
        logger.warning(f"指数数据加载失败：{e}")
        index_df = None
    
    # 加载行业数据
    industry_query = f"""
        SELECT symbol, trade_date, industry_name, industry_mv_ratio
        FROM stock_industry_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date, symbol
    """
    try:
        industry_df = db.read_sql(industry_query)
        logger.info(f"加载行业数据：{len(industry_df)} 条记录")
    except Exception as e:
        logger.warning(f"行业数据加载失败：{e}")
        industry_df = None
    
    return price_df, index_df, industry_df


def generate_report(stats: dict, output_path: str):
    """生成回测报告"""
    report_lines = []
    report_lines.append("# V51 回测报告 - 严正审计与回撤攻坚战")
    report_lines.append("")
    report_lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # 基本信息
    report_lines.append("## 基本信息")
    report_lines.append(f"- 版本：{stats.get('version', 'V51.0')}")
    report_lines.append(f"- 回测区间：{stats.get('start_date', '')} 至 {stats.get('end_date', '')}")
    report_lines.append(f"- 初始资金：{stats.get('initial_capital', 0):,.2f}")
    report_lines.append(f"- 最终净值：{stats.get('final_value', 0):,.2f}")
    report_lines.append("")
    
    # 业绩指标
    report_lines.append("## 业绩指标")
    report_lines.append(f"- 总收益率：{stats.get('total_return', 0):.2%}")
    report_lines.append(f"- 年化收益率：{stats.get('annual_return', 0):.2%}")
    report_lines.append(f"- 最大回撤：{stats.get('max_drawdown', 0):.2%}")
    report_lines.append(f"- 盈亏比：{stats.get('profit_loss_ratio', 0):.2f}")
    report_lines.append(f"- 胜率：{stats.get('win_rate', 0):.2%}")
    report_lines.append(f"- 交易次数：{stats.get('num_trades', 0)}")
    report_lines.append(f"- 盈利交易：{stats.get('profitable_trades', 0)}")
    report_lines.append(f"- 亏损交易：{stats.get('losing_trades', 0)}")
    report_lines.append("")
    
    # 目标对比
    targets = stats.get('performance_targets', {})
    target_met = stats.get('target_met', {})
    report_lines.append("## 目标达成情况")
    report_lines.append(f"- 年化收益目标 ({targets.get('annual_return_target', 0):.0%}): {'✅ 达成' if target_met.get('annual_return') else '❌ 未达成'}")
    report_lines.append(f"- 最大回撤目标 ({targets.get('max_drawdown_target', 0):.0%}): {'✅ 达成' if target_met.get('max_drawdown') else '❌ 未达成'}")
    report_lines.append(f"- 盈亏比目标 ({targets.get('profit_loss_ratio_target', 0):.1f}): {'✅ 达成' if target_met.get('profit_loss_ratio') else '❌ 未达成'}")
    report_lines.append(f"- 交易次数目标 ({targets.get('min_trades_target', 0)}-{targets.get('max_trades_target', 0)}): {'✅ 达成' if target_met.get('trade_count') else '❌ 未达成'}")
    report_lines.append("")
    
    # V51 特性
    features = stats.get('v51_features', {})
    report_lines.append("## V51 特性")
    report_lines.append(f"- T/T+1 隔离：{'✅ 启用' if features.get('t_t1_isolation') else '❌ 未启用'}")
    report_lines.append(f"- ATR 移动止损：{'✅ 启用' if features.get('atr_trailing_stop') else '❌ 未启用'}")
    report_lines.append(f"- 利润回撤保护：{'✅ 启用' if features.get('hwm_profit_protection') else '❌ 未启用'}")
    report_lines.append(f"- 利润回撤阈值：{features.get('hwm_threshold', 0):.0%}")
    report_lines.append(f"- 利润回撤比例：{features.get('hwm_drawdown_ratio', 0):.0%}")
    report_lines.append(f"- 单日回撤限制：{features.get('single_day_dd_limit', 0):.2%}")
    report_lines.append(f"- 周回撤限制：{features.get('weekly_dd_limit', 0):.0%}")
    report_lines.append(f"- 减仓比例：{features.get('cut_position_ratio', 0):.0%}")
    report_lines.append(f"- 行业适配：{'✅ 启用' if features.get('industry_adaptation') else '❌ 未启用'}")
    report_lines.append(f"- 高位放量压制：{'✅ 启用' if features.get('volume_suppression') else '❌ 未启用'}")
    report_lines.append("")
    
    # 风控统计
    drawdown_stats = stats.get('drawdown_stats', {})
    report_lines.append("## 风控状态")
    report_lines.append(f"- 当前单日回撤：{drawdown_stats.get('daily_drawdown', 0):.2%}")
    report_lines.append(f"- 当前周回撤：{drawdown_stats.get('weekly_drawdown', 0):.2%}")
    report_lines.append(f"- 单日回撤触发：{'⚠️ 是' if drawdown_stats.get('single_day_triggered') else '✅ 否'}")
    report_lines.append(f"- 周回撤触发：{'⚠️ 是' if drawdown_stats.get('weekly_triggered') else '✅ 否'}")
    report_lines.append(f"- 减仓激活：{'⚠️ 是' if drawdown_stats.get('cut_position_active') else '✅ 否'}")
    report_lines.append("")
    
    # 洗售统计
    wash_sale_stats = stats.get('wash_sale_stats', {})
    report_lines.append("## 洗售审计")
    report_lines.append(f"- 拦截次数：{wash_sale_stats.get('total_blocked', 0)}")
    report_lines.append("")
    
    # 黑名单统计
    blacklist_stats = stats.get('blacklist_stats', {})
    report_lines.append("## 止损黑名单")
    report_lines.append(f"- 当前黑名单数量：{blacklist_stats.get('total_blacklisted', 0)}")
    report_lines.append("")
    
    # 声明
    mdd = stats.get('max_drawdown', 0)
    if mdd > 0.04:
        report_lines.append("## ⚠️ 优化失败声明")
        report_lines.append(f"**我未能控制住回撤，原因在于：**")
        report_lines.append(f"- 最大回撤 {mdd:.2%} 超过目标 4%")
        report_lines.append("- 可能原因：ATR 止损倍数过松、利润回撤保护阈值过高、组合级风控触发不及时")
    else:
        report_lines.append("## ✅ 回撤控制成功")
        report_lines.append(f"- 最大回撤 {mdd:.2%} 控制在目标 4% 以内")
    
    report_content = "\n".join(report_lines)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"报告已保存至：{output_path}")
    return report_content


def main():
    """主函数"""
    # 配置
    start_date = "2024-01-01"
    end_date = "2026-03-21"
    initial_capital = V51_INITIAL_CAPITAL
    
    # 设置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add("logs/v51_backtest_{time:YYYYMMDD_HHmmss}.log", level="DEBUG")
    
    logger.info("=" * 60)
    logger.info("V51 回测脚本 - 严正审计与回撤攻坚战")
    logger.info("=" * 60)
    
    # 初始化数据库
    db = DatabaseManager()
    
    # 加载数据
    price_df, index_df, industry_df = load_data_from_db(db, start_date, end_date)
    
    if price_df.is_empty():
        logger.error("价格数据为空，无法运行回测")
        return
    
    # 运行回测
    stats = run_v51_backtest(
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
    report_path = f"reports/V51_Backtest_Report_{timestamp}.md"
    generate_report(stats, report_path)
    
    # 保存 JSON 结果
    json_path = f"reports/V51_backtest_result_{timestamp}.json"
    
    # 简化 JSON 输出（移除 portfolio_values 以减小文件大小）
    stats_for_json = stats.copy()
    if 'portfolio_values' in stats_for_json:
        stats_for_json['portfolio_values'] = stats_for_json['portfolio_values'][-100:]  # 只保留最后 100 条
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_for_json, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"JSON 结果已保存至：{json_path}")
    
    # 打印摘要
    logger.info("=" * 60)
    logger.info("V51 回测摘要")
    logger.info("=" * 60)
    logger.info(f"总收益率：{stats.get('total_return', 0):.2%}")
    logger.info(f"年化收益率：{stats.get('annual_return', 0):.2%}")
    logger.info(f"最大回撤：{stats.get('max_drawdown', 0):.2%}")
    logger.info(f"盈亏比：{stats.get('profit_loss_ratio', 0):.2f}")
    logger.info(f"交易次数：{stats.get('num_trades', 0)}")
    
    # 目标达成检查
    target_met = stats.get('target_met', {})
    mdd = stats.get('max_drawdown', 0)
    
    logger.info("=" * 60)
    if mdd > 0.04:
        logger.error("⚠️ 优化失败声明：我未能控制住回撤")
        logger.error(f"   最大回撤 {mdd:.2%} 超过目标 4%")
    else:
        logger.success("✅ 回撤控制成功")
        logger.success(f"   最大回撤 {mdd:.2%} 控制在目标 4% 以内")
    
    if target_met.get('annual_return'):
        logger.success("✅ 年化收益目标达成")
    else:
        logger.warning("❌ 年化收益目标未达成")
    
    if target_met.get('profit_loss_ratio'):
        logger.success("✅ 盈亏比目标达成")
    else:
        logger.warning("❌ 盈亏比目标未达成")


if __name__ == "__main__":
    main()