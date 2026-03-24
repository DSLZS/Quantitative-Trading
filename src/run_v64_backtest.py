"""
V64 回测运行脚本

【使用说明】
1. 确保数据库中有 stock_daily 数据
2. 可选：确保数据库中有 stock_fund_flow 和 stock_industry 数据
3. 运行此脚本执行 V64 回测

【V64 交付物】
- data_fetcher_v64.py (数据脚本)
- v64_core.py (增强 Alpha 引擎)
- v64_engine.py (带资金流因子的回测器)
- 本脚本 (运行入口)

作者：量化系统
版本：V64.0
日期：2026-03-24
"""

import sys
import os
from datetime import datetime

# 添加 src 目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

# 配置日志
logger.remove()
logger.add(
    "logs/v64_backtest_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(
    sys.stdout,
    level="INFO",
    format="{time:HH:mm:ss} | {level} | {message}"
)

from db_manager import DatabaseManager
from v64_engine import V64BacktestEngine, V64BacktestMetrics


def run_v64_backtest(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    initial_capital: float = 100000.0,
    max_positions: int = 10
) -> V64BacktestMetrics:
    """
    运行 V64 回测
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    initial_capital : float
        初始资金
    max_positions : int
        最大持仓数量
    
    Returns
    -------
    V64BacktestMetrics
        回测业绩指标
    """
    logger.info("=" * 60)
    logger.info("V64 资金流增强型 RS-Pullback 回测系统")
    logger.info("=" * 60)
    
    # 初始化数据库管理器
    try:
        db = DatabaseManager()
        logger.info("数据库连接成功")
    except Exception as e:
        logger.error(f"数据库连接失败：{e}")
        raise
    
    # 配置回测参数
    config = {
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'max_positions': max_positions,
    }
    
    # 创建并运行回测引擎
    engine = V64BacktestEngine(db=db, config=config)
    metrics = engine.run_backtest()
    
    # 保存交易历史
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存交易记录
        trades = engine.get_trade_history()
        if trades:
            trade_file = f"reports/V64_Trades_{timestamp}.csv"
            with open(trade_file, 'w', encoding='utf-8') as f:
                f.write("trade_date,symbol,side,shares,price,amount,commission,slippage,reason\n")
                for t in trades:
                    f.write(f"{t.trade_date},{t.symbol},{t.side},{t.shares},{t.price:.4f},{t.amount:.2f},{t.commission:.2f},{t.slippage:.2f},{t.reason}\n")
            logger.info(f"交易记录已保存：{trade_file}")
        
        # 保存审计记录
        audits = engine.get_audit_history()
        if audits:
            audit_file = f"reports/V64_Audits_{timestamp}.csv"
            with open(audit_file, 'w', encoding='utf-8') as f:
                f.write("symbol,buy_date,sell_date,buy_price,sell_price,shares,gross_pnl,net_pnl,holding_days,is_profitable,sell_reason\n")
                for a in audits:
                    f.write(f"{a.symbol},{a.buy_date},{a.sell_date},{a.buy_price:.4f},{a.sell_price:.4f},{a.shares},{a.gross_pnl:.2f},{a.net_pnl:.2f},{a.holding_days},{a.is_profitable},{a.sell_reason}\n")
            logger.info(f"审计记录已保存：{audit_file}")
        
        # 保存每日记录
        daily_records = engine.get_daily_records()
        if daily_records:
            daily_file = f"reports/V64_Daily_{timestamp}.csv"
            with open(daily_file, 'w', encoding='utf-8') as f:
                f.write("trade_date,cash,position_value,total_value,daily_return,position_count,buy_count,sell_count,is_safe_period,forced_empty\n")
                for r in daily_records:
                    f.write(f"{r.trade_date},{r.cash:.2f},{r.position_value:.2f},{r.total_value:.2f},{r.daily_return:.6f},{r.position_count},{r.buy_count},{r.sell_count},{r.is_safe_period},{r.forced_empty}\n")
            logger.info(f"每日记录已保存：{daily_file}")
            
            # 保存资金曲线数据
            equity_file = f"reports/V64_Equity_{timestamp}.json"
            import json
            equity_data = {
                'dates': [r.trade_date for r in daily_records],
                'values': [r.total_value for r in daily_records],
                'returns': [r.daily_return for r in daily_records],
            }
            with open(equity_file, 'w', encoding='utf-8') as f:
                json.dump(equity_data, f, indent=2, ensure_ascii=False)
            logger.info(f"资金曲线数据已保存：{equity_file}")
        
    except Exception as e:
        logger.error(f"保存数据失败：{e}")
    
    return metrics


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V64 资金流增强型 RS-Pullback 回测系统')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='回测开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='回测结束日期')
    parser.add_argument('--initial-capital', type=float, default=100000.0, help='初始资金')
    parser.add_argument('--max-positions', type=int, default=10, help='最大持仓数量')
    
    args = parser.parse_args()
    
    metrics = run_v64_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        max_positions=args.max_positions
    )
    
    # 打印最终摘要
    logger.info("=" * 60)
    logger.info("V64 回测完成")
    logger.info("=" * 60)
    logger.info(f"考核指标 PF*ln(Trades): {metrics.pf_ln_trades:.2f}")
    logger.info(f"总收益率：{metrics.total_return*100:.2f}%")
    logger.info(f"最大回撤：{metrics.max_drawdown*100:.2f}%")
    logger.info(f"夏普比率：{metrics.sharpe_ratio:.2f}")
    logger.info(f"总交易数：{metrics.total_trades}")
    logger.info(f"胜率：{metrics.win_rate*100:.2f}%")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()