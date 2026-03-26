"""
V77 Backtest Runner - 量价交互特征挖掘与 Rank IC 爆发计划

【使用说明】
python src/run_v77_backtest.py

【验收指标】
- 指标 A：全年度 Mean Rank IC >= 0.035（这是本次唯一的及格线）
- 指标 B：Calmar Ratio > 2.0
- 指标 C：单月 Rank IC 出现负值的月份不得超过 2 个

作者：量化系统
版本：V77.0
日期：2026-03-26
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.v77_engine import run_v77_backtest, V77BacktestEngine, V77BacktestResult
from src.db_manager import DatabaseManager


def configure_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def main():
    """主函数"""
    configure_logger()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='V77 回测 - 量价交互特征挖掘与 Rank IC 爆发计划')
    parser.add_argument('--start-date', type=str, default='2024-01-01', help='开始日期')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='结束日期')
    parser.add_argument('--output', type=str, default=None, help='报告输出路径')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("V77 回测 - 量价交互特征挖掘与 Rank IC 爆发计划")
    logger.info("=" * 60)
    logger.info(f"回测区间：[{args.start_date}, {args.end_date}]")
    logger.info(f"报告输出：{args.output or '默认路径'}")
    logger.info("=" * 60)
    
    try:
        # 运行回测
        result = run_v77_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            output_path=args.output
        )
        
        # 验证验收指标
        logger.info("=" * 60)
        logger.info("V77 验收指标验证")
        logger.info("=" * 60)
        
        # 指标 A: Mean Rank IC >= 0.035
        rank_ic_pass = result.mean_rank_ic >= 0.035
        logger.info(f"指标 A (Mean Rank IC >= 0.035): {result.mean_rank_ic:.4f} - {'✓ 通过' if rank_ic_pass else '✗ 未通过'}")
        
        # 指标 B: Calmar Ratio > 2.0
        calmar_pass = result.calmar_ratio > 2.0
        logger.info(f"指标 B (Calmar Ratio > 2.0): {result.calmar_ratio:.3f} - {'✓ 通过' if calmar_pass else '✗ 未通过'}")
        
        # 指标 C: 负值月份 <= 2
        negative_months_pass = result.negative_months <= 2
        logger.info(f"指标 C (负值月份 <= 2): {result.negative_months} - {'✓ 通过' if negative_months_pass else '✗ 未通过'}")
        
        logger.info("=" * 60)
        
        # 综合结论
        all_pass = rank_ic_pass and calmar_pass and negative_months_pass
        if all_pass:
            logger.info("✓ V77 所有验收指标通过！")
        else:
            logger.warning("✗ V77 部分验收指标未通过，需要继续优化")
            if not rank_ic_pass:
                logger.warning(f"  - Rank IC {result.mean_rank_ic:.4f} < 0.035")
            if not calmar_pass:
                logger.warning(f"  - Calmar Ratio {result.calmar_ratio:.3f} <= 2.0")
            if not negative_months_pass:
                logger.warning(f"  - 负值月份 {result.negative_months} > 2")
        
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"V77 回测失败：{e}")
        logger.error(f"错误详情：{str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()