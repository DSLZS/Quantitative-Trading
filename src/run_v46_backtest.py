"""
V46 Backtest Runner - 极简主义与长线守望

【使用说明】
1. 确保数据库已连接
2. 确保 stock_daily, index_daily 表有数据
3. 运行：python run_v46_backtest.py

【V46 目标】
- 年化收益 > 15%
- 回撤 < 4%
- 交易次数 [20, 35]

作者：量化系统
版本：V46.0
日期：2026-03-19
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from v46_engine import V46Engine, FIXED_INITIAL_CAPITAL
from v46_core import (
    TRADE_COUNT_FAIL_THRESHOLD,
    ANNUAL_RETURN_TARGET,
    MAX_DRAWDOWN_TARGET,
)


def load_v44_data_for_comparison() -> dict | None:
    """
    加载 V44 数据用于对比
    
    尝试从最新的 V44 报告中读取数据
    """
    try:
        reports_dir = Path("reports")
        v44_reports = sorted(reports_dir.glob("V44*Report*.md"))
        
        if not v44_reports:
            logger.warning("No V44 reports found for comparison")
            return None
        
        latest_v44_report = v44_reports[-1]
        logger.info(f"Loading V44 data from: {latest_v44_report}")
        
        # 尝试从 JSON 文件读取（如果有）
        json_path = latest_v44_report.with_suffix('.json')
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('v44_data', data)
        
        # 默认返回 None，让 V46 自行判断
        return None
        
    except Exception as e:
        logger.warning(f"Failed to load V44 data: {e}")
        return None


def run_v46_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = FIXED_INITIAL_CAPITAL,
) -> dict:
    """
    运行 V46 回测
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
    
    Returns:
        回测结果字典
    """
    logger.info("=" * 80)
    logger.info("V46 BACKTEST - MINIMALISM & LONG-TERM HOLDING")
    logger.info("=" * 80)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"Initial Capital: {initial_capital:,.2f}")
    logger.info("=" * 80)
    
    # 加载 V44 数据用于对比
    v44_data = load_v44_data_for_comparison()
    if v44_data:
        logger.info(f"V44 data loaded for comparison: Return={v44_data.get('total_return', 0):.2%}")
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V46 REQUIRES database connection - exiting")
        raise
    
    # 创建引擎
    engine = V46Engine(config={
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'v44_data': v44_data,
    }, db=db)
    
    # 运行回测
    result = engine.run_backtest()
    
    # 生成报告
    report = engine.generate_markdown_report(result)
    
    # 保存报告
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"V46_Backtest_Report_{timestamp}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {output_path}")
    
    # 保存 JSON 结果
    json_path = output_dir / f"V46_Backtest_Result_{timestamp}.json"
    json_data = {
        'version': 'V46',
        'timestamp': timestamp,
        'period': {'start': start_date, 'end': end_date},
        'initial_capital': initial_capital,
        'final_value': result.get('final_value', initial_capital),
        'total_return': result.get('total_return', 0),
        'max_drawdown': result.get('max_drawdown', 0),
        'sharpe_ratio': result.get('sharpe_ratio', 0),
        'total_trades': result.get('total_trades', 0),
        'buy_trades': result.get('buy_trades', 0),
        'sell_trades': result.get('sell_trades', 0),
        'win_rate': result.get('win_rate', 0),
        'avg_holding_days': result.get('avg_holding_days', 0),
        'total_fees': result.get('total_fees', 0),
        'wash_sale_stats': result.get('wash_sale_stats', {}),
        'market_regime_stats': result.get('market_regime_stats', {}),
        'v44_data': v44_data,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON result saved to: {json_path}")
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("V46 BACKTEST SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Return:      {result.get('total_return', 0):.2%} (Target: >{ANNUAL_RETURN_TARGET:.0%})")
    logger.info(f"Max Drawdown:      {result.get('max_drawdown', 0):.2%} (Target: <{MAX_DRAWDOWN_TARGET:.0%})")
    logger.info(f"Sharpe Ratio:      {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Total Trades:      {result.get('total_trades', 0)} (Target: [20, 35])")
    logger.info(f"Win Rate:          {result.get('win_rate', 0):.1%}")
    logger.info(f"Avg Holding Days:  {result.get('avg_holding_days', 0):.1f}")
    logger.info(f"Wash Sale Blocks:  {result.get('wash_sale_stats', {}).get('total_blocked', 0)}")
    logger.info(f"Risk Period Days:  {result.get('market_regime_stats', {}).get('risk_period_days', 0)}")
    logger.info("=" * 80)
    
    # V46 审计结论
    total_trades = result.get('total_trades', 0)
    mdd = result.get('max_drawdown', 0)
    total_return = result.get('total_return', 0)
    
    audit_passed = True
    audit_messages = []
    
    # 检查交易次数
    if total_trades > TRADE_COUNT_FAIL_THRESHOLD:
        audit_messages.append(f"[FAILED] OVER-TRADING: {total_trades} > {TRADE_COUNT_FAIL_THRESHOLD}")
        audit_passed = False
    elif total_trades < 20:
        audit_messages.append(f"[WARNING] UNDER-TRADING: {total_trades} < 20")
    elif total_trades > 35:
        audit_messages.append(f"[WARNING] OVER-TRADING: {total_trades} > 35")
    else:
        audit_messages.append(f"[PASSED] Trade count: {total_trades} in [20, 35]")
    
    # 检查回撤
    if mdd > MAX_DRAWDOWN_TARGET:
        audit_messages.append(f"[FAILED] RISK TARGET: MDD {mdd:.2%} > {MAX_DRAWDOWN_TARGET:.0%}")
        audit_passed = False
    else:
        audit_messages.append(f"[PASSED] MDD: {mdd:.2%} <= {MAX_DRAWDOWN_TARGET:.0%}")
    
    # 检查收益
    if total_return < ANNUAL_RETURN_TARGET:
        audit_messages.append(f"[WARNING] RETURN TARGET: {total_return:.2%} < {ANNUAL_RETURN_TARGET:.0%}")
    else:
        audit_messages.append(f"[PASSED] Return: {total_return:.2%} >= {ANNUAL_RETURN_TARGET:.0%}")
    
    # 对比 V44（如果有数据）
    if v44_data:
        v44_return = v44_data.get('total_return', 0)
        v44_mdd = v44_data.get('max_drawdown', 0)
        
        if total_return < v44_return:
            audit_messages.append(f"[FAILED] NEGATIVE OPTIMIZATION: Return {total_return:.2%} < V44 {v44_return:.2%}")
            audit_passed = False
        else:
            audit_messages.append(f"[PASSED] Return vs V44: {total_return:.2%} >= {v44_return:.2%}")
        
        if mdd > v44_mdd:
            audit_messages.append(f"[FAILED] NEGATIVE OPTIMIZATION: MDD {mdd:.2%} > V44 {v44_mdd:.2%}")
            audit_passed = False
        else:
            audit_messages.append(f"[PASSED] MDD vs V44: {mdd:.2%} <= {v44_mdd:.2%}")
    
    # 打印审计结论
    logger.info("\nV46 AUDIT CONCLUSION:")
    for msg in audit_messages:
        logger.info(f"  {msg}")
    
    if audit_passed:
        logger.info("\n[V46 AUDIT PASSED] All constraints satisfied!")
    else:
        logger.error("\n[V46 AUDIT FAILED] Some constraints not satisfied!")
        logger.error("Please review the report and adjust parameters.")
    
    logger.info("=" * 80)
    
    return result


def main():
    """主入口"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行回测
    result = run_v46_backtest()
    
    return result


if __name__ == "__main__":
    main()