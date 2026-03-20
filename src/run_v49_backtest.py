"""
V49 回测运行脚本 - 持有为王与组合断路器

【使用说明】
1. 确保数据库已连接并包含必要的数据表
2. 运行：python src/run_v49_backtest.py
3. 报告将保存在 reports/V49_Backtest_Report_*.md

【V49 核心改进】
1. 强制时间锁：15 天持有期
2. 调仓过滤：30% 得分提升
3. 动态仓位：回撤>3% 时降至 10%
4. 组合断路器：-4% 总资金止损
5. ma5 Bug 修复

作者：量化系统
版本：V49.0
日期：2026-03-19
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

# 尝试导入历史数据
def load_historical_data():
    """加载 V40, V44, V47, V48 历史数据用于对比"""
    v40_data = None
    v44_data = None
    v47_data = None
    v48_data = None
    
    reports_dir = Path("reports")
    
    # 尝试加载 V44 数据（基准）
    v44_files = list(reports_dir.glob("*V44*Final*Summary*.md"))
    if not v44_files:
        v44_files = list(reports_dir.glob("*V44*.md"))
    
    # 尝试从 JSON 文件加载
    for json_file in reports_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'version' in data:
                    if data['version'] == 'V40' and v40_data is None:
                        v40_data = data
                        logger.info(f"Loaded V40 data from {json_file}")
                    elif data['version'] == 'V44' and v44_data is None:
                        v44_data = data
                        logger.info(f"Loaded V44 data from {json_file}")
                    elif data['version'] == 'V47' and v47_data is None:
                        v47_data = data
                        logger.info(f"Loaded V47 data from {json_file}")
                    elif data['version'] == 'V48' and v48_data is None:
                        v48_data = data
                        logger.info(f"Loaded V48 data from {json_file}")
        except Exception as e:
            pass
    
    return v40_data, v44_data, v47_data, v48_data


def main():
    """V49 回测主入口"""
    logger.info("=" * 70)
    logger.info("V49 BACKTEST - HOLDING LOCK & PORTFOLIO CIRCUIT BREAKER")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载历史数据
    v40_data, v44_data, v47_data, v48_data = load_historical_data()
    
    if v44_data:
        logger.info(f"V44 Baseline - Return: {v44_data.get('total_return', 0):.2%}, MDD: {v44_data.get('max_drawdown', 0):.2%}")
    if v48_data:
        logger.info(f"V48 Previous - Return: {v48_data.get('total_return', 0):.2%}, MDD: {v48_data.get('max_drawdown', 0):.2%}")
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V49 REQUIRES database connection - exiting")
        return None
    
    try:
        from v49_engine import V49Engine, FIXED_INITIAL_CAPITAL
        
        engine = V49Engine(config={
            'start_date': '2024-01-01',
            'end_date': '2024-12-31',
            'initial_capital': FIXED_INITIAL_CAPITAL,
            'v44_data': v44_data,
            'v40_data': v40_data,
            'v47_data': v47_data,
            'v48_data': v48_data,
            'auto_adjust_weights': True,
        }, db=db)
        
        result = engine.run_backtest(adjust_weights_if_needed=True)
        
        # 生成报告
        report = engine.generate_markdown_report(result)
        
        # 保存报告
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V49_Backtest_Report_{timestamp}.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to: {output_path}")
        
        # 保存 JSON 结果用于后续对比
        json_result = {
            'version': 'V49',
            'initial_capital': result.get('initial_capital', FIXED_INITIAL_CAPITAL),
            'final_value': result.get('final_value', FIXED_INITIAL_CAPITAL),
            'total_return': result.get('total_return', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'total_trades': result.get('total_trades', 0),
            'buy_trades': result.get('buy_trades', 0),
            'sell_trades': result.get('sell_trades', 0),
            'win_rate': result.get('win_rate', 0),
            'avg_holding_days': result.get('avg_holding_days', 0),
            'total_fees': result.get('total_fees', 0),
            'invalid_stop_count': result.get('invalid_stop_count', 0),
            'blacklist_blocks': result.get('blacklist_blocks', 0),
            'circuit_breaker_triggers': result.get('circuit_breaker_triggers', 0),
            'profit_lock_triggers': result.get('profit_lock_triggers', 0),
            'trailing_stop_triggers': result.get('trailing_stop_triggers', 0),
            'portfolio_stop_loss_triggers': result.get('portfolio_stop_loss_triggers', 0),
            'time_lock_triggers': result.get('time_lock_triggers', 0),
            'dynamic_position_triggers': result.get('dynamic_position_triggers', 0),
            'weights_adjusted': result.get('weights_adjusted', False),
        }
        
        json_path = output_dir / f"V49_backtest_result_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, default=str)
        logger.info(f"JSON result saved to: {json_path}")
        
        # 打印摘要
        logger.info("\n" + "=" * 70)
        logger.info("V49 BACKTEST SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Initial Capital:    {result.get('initial_capital', 0):,.2f} CNY")
        logger.info(f"Final Value:        {result.get('final_value', 0):,.2f} CNY")
        logger.info(f"Total Return:       {result.get('total_return', 0):.2%}")
        logger.info(f"Sharpe Ratio:       {result.get('sharpe_ratio', 0):.3f}")
        logger.info(f"Max Drawdown:       {result.get('max_drawdown', 0):.2%}")
        logger.info(f"Total Trades:       {result.get('total_trades', 0)}")
        logger.info(f"Win Rate:           {result.get('win_rate', 0):.1%}")
        logger.info(f"Invalid Stops:      {result.get('invalid_stop_count', 0)}")
        logger.info(f"Time Lock Triggers: {result.get('time_lock_triggers', 0)}")
        logger.info(f"Portfolio Stop Loss: {result.get('portfolio_stop_loss_triggers', 0)}")
        logger.info(f"Dynamic Position Triggers: {result.get('dynamic_position_triggers', 0)}")
        
        # 性能检查
        logger.info("\n" + "=" * 70)
        logger.info("PERFORMANCE AUDIT")
        logger.info("=" * 70)
        
        total_return = result.get('total_return', 0)
        max_drawdown = result.get('max_drawdown', 0)
        total_trades = result.get('total_trades', 0)
        
        # 收益率检查
        if total_return >= 0.15:
            logger.info(f"✅ RETURN TARGET MET: {total_return:.2%} >= 15%")
        else:
            logger.warning(f"❌ RETURN TARGET NOT MET: {total_return:.2%} < 15%")
        
        # 回撤检查
        if max_drawdown <= 0.04:
            logger.info(f"✅ DRAWDOWN TARGET MET: {max_drawdown:.2%} <= 4%")
        else:
            logger.warning(f"❌ DRAWDOWN TARGET NOT MET: {max_drawdown:.2%} > 4%")
        
        # 交易次数检查
        if total_trades > 35:
            logger.warning(f"⚠️ [V49 OVER-TRADING FAILURE] Trade count {total_trades} > 35")
        else:
            logger.info(f"✅ TRADE COUNT OK: {total_trades} <= 35")
        
        # vs V44 对比
        if v44_data:
            v44_return = v44_data.get('total_return', 0)
            v44_mdd = v44_data.get('max_drawdown', 0)
            
            if total_return >= v44_return:
                logger.info(f"✅ OUTPERFORMED V44: {total_return:.2%} >= {v44_return:.2%}")
            else:
                logger.warning(f"❌ UNDERPERFORMED V44: {total_return:.2%} < {v44_return:.2%}")
            
            if max_drawdown <= v44_mdd:
                logger.info(f"✅ BETTER DRAWDOWN THAN V44: {max_drawdown:.2%} <= {v44_mdd:.2%}")
            else:
                logger.warning(f"⚠️ WORSE DRAWDOWN THAN V44: {max_drawdown:.2%} > {v44_mdd:.2%}")
        
        logger.info("=" * 70)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return result
        
    except Exception as e:
        logger.error(f"V49 backtest FAILED: {e}")
        logger.error(f"Error details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    main()