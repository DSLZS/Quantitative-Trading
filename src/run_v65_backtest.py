"""
V65 回测运行脚本 - 机构资金踪迹模型

【使用说明】
1. 首先运行数据加载脚本：python src/v65_data_loader.py
2. 然后运行回测脚本：python src/run_v65_backtest.py

【V65 核心特性】
1. 数据强制落库：拒绝自修复，数据为空直接 sys.exit(1)
2. 机构资金踪迹：行业护城河 + 资金共振 + VCP 动态阈值
3. 趋势破坏止损：跌破 MA10 且主力资金净流出时离场
4. AE 评价指标：(Win_Rate * P/L_Ratio) / Max_Drawdown
5. 数据可信度：标注每笔交易是基于"资金流 + 行业"双因子还是单因子

作者：量化系统
版本：V65.0
日期：2026-03-24
"""

import sys
from loguru import logger
from db_manager import get_db
from v65_engine import run_v65_backtest, V65_INITIAL_CAPITAL, V65_MAX_POSITIONS


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V65 机构资金踪迹模型回测")
    logger.info("=" * 60)
    
    # 回测配置
    config = {
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': V65_INITIAL_CAPITAL,
        'max_positions': V65_MAX_POSITIONS,
    }
    
    logger.info(f"回测区间：[{config['start_date']}, {config['end_date']}]")
    logger.info(f"初始资金：{config['initial_capital']:,.2f}")
    logger.info(f"最大持仓：{config['max_positions']}只")
    logger.info("=" * 60)
    
    # 初始化数据库
    db = get_db()
    
    # 运行回测
    try:
        metrics = run_v65_backtest(
            start_date=config['start_date'],
            end_date=config['end_date'],
            initial_capital=config['initial_capital'],
            max_positions=config['max_positions'],
            db=db
        )
        
        logger.info("=" * 60)
        logger.info("V65 回测完成")
        logger.info("=" * 60)
        
        # 输出关键指标
        logger.info(f"AE (Alpha-Efficiency): {metrics.alpha_efficiency:.4f}")
        logger.info(f"总收益率：{metrics.total_return*100:.2f}%")
        logger.info(f"最大回撤：{metrics.max_drawdown*100:.2f}%")
        logger.info(f"总交易次数：{metrics.total_trades}")
        logger.info(f"双因子交易占比：{metrics.data_credibility_ratio*100:.1f}%")
        
        # 检查交易次数约束
        if 20 <= metrics.total_trades <= 50:
            logger.info("✓ 交易次数符合约束 (20-50 次)")
        else:
            logger.warning(f"✗ 交易次数不符合约束 (20-50 次)，实际：{metrics.total_trades}")
        
        return metrics
        
    except SystemExit as e:
        logger.error(f"V65 回测被终止：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V65 回测失败：{e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()