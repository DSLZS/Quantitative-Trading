"""
V67 回测运行脚本 - SPI 信号质量审计

【使用方法】
python src/run_v67_backtest.py

【功能】
1. 运行 V67 数据填充器（可选）
2. 验证数据充足性
3. 运行 V67 回测引擎
4. 生成回测报告

作者：量化系统
版本：V67.0
日期：2026-03-24
"""

import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

from loguru import logger

from v67_data_filler import V67DataFiller, fill_v67_data, verify_v67_data
from v67_engine import run_v67_backtest, V67BacktestResult
from v67_core import V67_SPI_TARGET, V67_SPI_MIN


def setup_logger():
    """配置日志输出"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 同时输出到文件
    log_file = f"logs/v67_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    Path("logs").mkdir(exist_ok=True)
    logger.add(log_file, level="DEBUG", rotation="10 MB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='V67 回测运行脚本')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                        help='开始日期 (默认：2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                        help='结束日期 (默认：2024-12-31)')
    parser.add_argument('--fill-data', action='store_true',
                        help='是否先运行数据填充器')
    parser.add_argument('--verify-only', action='store_true',
                        help='只验证数据充足性，不运行回测')
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logger()
    
    logger.info("=" * 80)
    logger.info("V67 回测系统 - SPI 信号质量审计")
    logger.info("=" * 80)
    logger.info(f"回测区间：[{args.start_date}, {args.end_date}]")
    logger.info(f"数据填充：{'是' if args.fill_data else '否'}")
    logger.info(f"只验证数据：{'是' if args.verify_only else '否'}")
    logger.info("=" * 80)
    
    # 1. 数据填充（可选）
    if args.fill_data:
        logger.info("=" * 60)
        logger.info("V67: 开始数据填充")
        logger.info("=" * 60)
        
        try:
            result = fill_v67_data(args.start_date, args.end_date)
            
            logger.info(f"V67: 数据填充完成")
            logger.info(f"  资金流成功：{result.get('success', 0)}只")
            logger.info(f"  资金流失败：{result.get('failure', 0)}只")
            logger.info(f"  总行数：{result.get('total_rows', 0):,}")
            logger.info(f"  数据充足：{result.get('is_sufficient', False)}")
            
        except SystemExit:
            logger.error("V67: 数据填充因连续失败已退出")
            sys.exit(1)
        except Exception as e:
            logger.error(f"V67: 数据填充失败：{e}")
            sys.exit(1)
    
    # 2. 验证数据充足性
    logger.info("=" * 60)
    logger.info("V67: 验证数据充足性")
    logger.info("=" * 60)
    
    is_sufficient, message = verify_v67_data()
    
    if not is_sufficient:
        logger.error(f"V67: {message}")
        logger.error("V67: 请先运行数据填充器：python src/run_v67_backtest.py --fill-data")
        sys.exit(1)
    
    logger.info(f"V67: 数据验证通过")
    
    # 3. 只验证数据模式
    if args.verify_only:
        logger.info("=" * 60)
        logger.info("V67: 数据验证完成")
        logger.info("=" * 60)
        sys.exit(0)
    
    # 4. 运行回测
    logger.info("=" * 60)
    logger.info("V67: 开始运行回测")
    logger.info("=" * 60)
    
    try:
        result = run_v67_backtest(args.start_date, args.end_date)
        
        # 保存结果
        if result.trade_count > 0:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = output_dir / f"V67_backtest_result_{timestamp}.json"
            
            # 转换为字典
            from dataclasses import asdict
            result_dict = asdict(result)
            
            # 保存 JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"V67: 回测结果已保存至 {output_file}")
            
            # 生成 Markdown 报告
            md_report = generate_markdown_report(result, args.start_date, args.end_date)
            md_file = output_dir / f"V67_backtest_report_{timestamp}.md"
            
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(md_report)
            
            logger.info(f"V67: Markdown 报告已保存至 {md_file}")
            
        else:
            logger.warning("V67: 回测结果为空，可能没有产生任何交易")
        
    except Exception as e:
        logger.error(f"V67: 回测运行失败：{e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
    
    logger.info("=" * 80)
    logger.info("V67: 回测运行完成")
    logger.info("=" * 80)


def generate_markdown_report(result: V67BacktestResult, start_date: str, end_date: str) -> str:
    """生成 Markdown 格式回测报告"""
    report = f"""# V67 回测报告

## 基本信息

- **回测区间**: {start_date} 至 {end_date}
- **报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **系统版本**: V67.0

## 核心指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {result.total_return*100:.2f}% |
| 年化收益率 | {result.annual_return*100:.2f}% |
| 最大回撤 | {result.max_drawdown*100:.2f}% |
| 夏普比率 | {result.sharpe_ratio:.2f} |
| AE 指标 | {result.ae_metric:.2f} |

## 交易统计

| 指标 | 数值 |
|------|------|
| 交易次数 | {result.trade_count} |
| 盈利次数 | {result.winning_trades} |
| 亏损次数 | {result.losing_trades} |
| 胜率 | {result.win_rate*100:.1f}% |
| 盈亏比 | {result.profit_loss_ratio:.2f} |

## SPI 信号质量审计

| 指标 | 数值 | 阈值 |
|------|------|------|
| Mean SPI | {result.mean_spi:.4f} | > {V67_SPI_TARGET} |
| Min SPI | {result.min_spi:.4f} | >= {V67_SPI_MIN} |
| Max SPI | {result.max_spi:.4f} | - |
| SPI Pass Ratio | {result.spi_pass_ratio*100:.1f}% | - |

## IC 审计

| 指标 | 数值 |
|------|------|
| Mean IC | {result.mean_ic:.4f} |
| IC Std | {result.ic_std:.4f} |
| IC IR | {result.ic_ir:.2f} |

## SPI 审计结论

"""
    
    # SPI 审计结论
    if result.mean_spi >= V67_SPI_TARGET:
        report += f"✅ **SPI 达标**: Mean SPI ({result.mean_spi:.4f}) >= 目标 ({V67_SPI_TARGET})\n\n"
    else:
        report += f"⚠️ **SPI 未达标**: Mean SPI ({result.mean_spi:.4f}) < 目标 ({V67_SPI_TARGET})\n\n"
    
    if result.min_spi < V67_SPI_MIN:
        report += f"⚠️ **SPI 低于最低容忍值**: Min SPI ({result.min_spi:.4f}) < {V67_SPI_MIN}\n"
        report += "建议检查策略逻辑或降低交易频率\n\n"
    
    if result.spi_pass_ratio >= 0.8:
        report += f"✅ **SPI 通过率高**: {result.spi_pass_ratio*100:.1f}% 的交易日 SPI 达标\n\n"
    else:
        report += f"⚠️ **SPI 通过率偏低**: 仅 {result.spi_pass_ratio*100:.1f}% 的交易日 SPI 达标\n"
        report += "建议调整策略参数或重新优化信号权重\n\n"
    
    # 添加交易明细（前 20 条）
    if result.trades:
        report += "## 交易明细（前 20 条）\n\n"
        report += "| 日期 | 股票代码 | 方向 | 价格 | 数量 | 原因 |\n"
        report += "|------|---------|------|------|------|------|\n"
        
        for trade in result.trades[:20]:
            report += f"| {trade['trade_date']} | {trade['symbol']} | {trade['side']} | {trade['price']:.2f} | {trade['shares']} | {trade['reason']} |\n"
        
        if len(result.trades) > 20:
            report += f"\n*共 {len(result.trades)} 条交易记录，此处仅显示前 20 条*\n"
    
    return report


if __name__ == "__main__":
    main()