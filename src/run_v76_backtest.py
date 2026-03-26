#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V76 回测运行脚本 - 行业拥挤度与波动率缩放

【使用说明】
1. 直接运行：python src/run_v76_backtest.py
2. 自定义回测区间：python src/run_v76_backtest.py --start 2024-01-01 --end 2024-12-31
3. 输出路径：python src/run_v76_backtest.py --output reports/v76_report.md

【V76 核心特性】
- 线性排名融合（回归 V74 基准）
- 行业拥挤度审计
- 波动率倒数加权
- 行业分散性控制

作者：量化系统
版本：V76.0
日期：2026-03-26
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.v76_engine import run_v76_backtest, V76BacktestEngine, V76BacktestResult
from src.db_manager import DatabaseManager
from loguru import logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='V76 回测运行脚本')
    
    parser.add_argument(
        '--start',
        type=str,
        default='2024-01-01',
        help='回测开始日期 (YYYY-MM-DD)，默认：2024-01-01'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default='2024-12-31',
        help='回测结束日期 (YYYY-MM-DD)，默认：2024-12-31'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='报告输出路径，默认：reports/v76_backtest_report_YYYYMMDD_HHMMSS.md'
    )
    
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='初始资金，默认：100000'
    )
    
    parser.add_argument(
        '--max-positions',
        type=int,
        default=10,
        help='最大持仓数，默认：10'
    )
    
    parser.add_argument(
        '--max-sector-weight',
        type=float,
        default=0.20,
        help='单行业最大权重，默认：0.20 (20%)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='启用详细日志输出'
    )
    
    return parser.parse_args()


def setup_logger(verbose: bool = False):
    """配置日志"""
    logger.remove()
    
    # 控制台输出
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    if verbose:
        logger.add(sys.stderr, format=log_format, level="DEBUG")
    else:
        logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
    
    # 文件输出
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.add(log_dir / f"v76_backtest_{timestamp}.log", level="DEBUG", rotation="100 MB")


def run_backtest(args):
    """运行回测"""
    logger.info("=" * 60)
    logger.info("V76 回测脚本启动")
    logger.info(f"回测区间：[{args.start}, {args.end}]")
    logger.info(f"初始资金：{args.initial_capital:,.0f}")
    logger.info(f"最大持仓数：{args.max_positions}")
    logger.info(f"单行业上限：{args.max_sector_weight*100:.1f}%")
    logger.info("=" * 60)
    
    # 配置
    config = {
        'initial_capital': args.initial_capital,
        'max_positions': args.max_positions,
        'max_sector_weight': args.max_sector_weight,
    }
    
    # 初始化数据库
    db = DatabaseManager()
    
    # 初始化引擎
    engine = V76BacktestEngine(db=db, config=config)
    
    # 运行回测
    result = engine.run_backtest(args.start, args.end)
    
    # 打印结果
    engine.print_backtest_result(result)
    
    # 生成报告
    output_path = args.output
    if output_path is None:
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(output_dir / f"V76_Backtest_Report_{timestamp}.md")
    
    report_content = engine.generate_report(result, output_path)
    logger.info(f"报告已保存至：{output_path}")
    
    # 打印 Rank IC 报告
    engine.rank_ic_calculator.print_rank_ic_report()
    
    # 验收指标检查
    logger.info("=" * 60)
    logger.info("V76 验收指标检查")
    logger.info("=" * 60)
    
    # 指标 A: Mean Rank IC >= 0.03
    rank_ic_pass = result.mean_rank_ic >= 0.03
    logger.info(f"指标 A: Mean Rank IC >= 0.03")
    logger.info(f"  实际值：{result.mean_rank_ic:.4f}")
    logger.info(f"  状态：{'✓ 通过' if rank_ic_pass else '✗ 未通过'}")
    
    # 指标 B: 最大回撤 < 12%
    drawdown_pass = result.max_drawdown < 0.12
    logger.info(f"指标 B: 最大回撤 < 12%")
    logger.info(f"  实际值：{result.max_drawdown*100:.2f}%")
    logger.info(f"  状态：{'✓ 通过' if drawdown_pass else '✗ 未通过'}")
    
    # 指标 C: 行业分散性控制
    max_sector = max(result.sector_allocation.values()) if result.sector_allocation else 0.0
    sector_pass = max_sector <= 0.20
    logger.info(f"指标 C: 单行业持仓 <= 20%")
    logger.info(f"  实际值：{max_sector*100:.2f}%")
    logger.info(f"  状态：{'✓ 通过' if sector_pass else '✗ 未通过'}")
    
    # 总体评估
    all_pass = rank_ic_pass and drawdown_pass and sector_pass
    logger.info("-" * 60)
    logger.info(f"总体评估：{'✓ 全部通过' if all_pass else '✗ 部分未通过'}")
    logger.info("=" * 60)
    
    return result


def main():
    """主函数"""
    args = parse_args()
    setup_logger(args.verbose)
    
    try:
        result = run_backtest(args)
        logger.info("V76 回测执行完成")
        return 0
    except Exception as e:
        logger.error(f"V76 回测失败：{e}")
        logger.exception("详细错误信息:")
        return 1


if __name__ == "__main__":
    sys.exit(main())