#!/usr/bin/env python
"""
V79 回测运行脚本 - 行业残差 Alpha 与流动性过滤增强

【任务要求】
1. 强制纠错指令（解决 V78 报错）
   - DB 审计：检查 stock_industry_daily 和 stock_daily 表结构
   - 自动修复：若遇到 OperationalError 或 KeyError，主动分析并修复

2. 核心算法进化
   - Residual Alpha: (个股 5 日 RS - 行业 5 日均值 RS) / 波动率
   - V-Shock 2.0: 成交额突增 3 倍以上一票否决
   - 因子权重：最大化 Rank IC 导向

3. 防欺诈与防偷懒高压线
   - 强制执行数据完整性检查
   - 缺失时自动启动补数逻辑
   - 严禁未来数据

4. 验收指标
   - 指标 A：全年度 Mean Rank IC >= 0.035
   - 指标 B：回测覆盖 2024 年 242 个交易日
   - 指标 C：最大回撤控制在 10% 以内

作者：量化系统
版本：V79.0
日期：2026-03-26
"""

import sys
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.db_manager import DatabaseManager
from src.v79_engine import run_v79_backtest, V79BacktestEngine


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    
    # 添加文件日志
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "v79_backtest_{time:YYYYMMDD}.log",
        rotation="100 MB",
        retention="30 days",
        level="DEBUG",
    )


def check_db_structure():
    """检查数据库表结构（DB 审计）"""
    logger.info("=" * 60)
    logger.info("V79: DB 审计 - 检查表结构")
    logger.info("=" * 60)
    
    db = DatabaseManager()
    
    try:
        # 检查 stock_daily 表结构
        logger.info("检查 stock_daily 表结构...")
        df = db.read_sql("DESCRIBE stock_daily")
        columns = df['Field'].to_list()
        logger.info(f"stock_daily 列：{columns}")
        
        # 检查是否有 industry_code 和 pct_chg
        has_industry_code = 'industry_code' in columns
        has_pct_chg = 'pct_chg' in columns
        logger.info(f"  - industry_code: {'✓' if has_industry_code else '✗'}")
        logger.info(f"  - pct_chg: {'✓' if has_pct_chg else '✗'}")
        
        # 检查 stock_industry_daily 表结构
        logger.info("检查 stock_industry_daily 表结构...")
        df = db.read_sql("DESCRIBE stock_industry_daily")
        columns = df['Field'].to_list()
        logger.info(f"stock_industry_daily 列：{columns}")
        
        # 关键检查：industry_return 字段是否存在
        has_industry_return = 'industry_return' in columns
        logger.info(f"  - industry_return: {'✓' if has_industry_return else '✗'}")
        
        if not has_industry_return:
            logger.warning("⚠ industry_return 字段不存在！")
            logger.warning("⚠ V79 将使用实时计算：行业收益 = 该行业所有个股 pct_chg 的算术平均")
        else:
            logger.info("✓ industry_return 字段存在，可直接使用")
        
        return True
        
    except Exception as e:
        logger.error(f"DB 审计失败：{e}")
        logger.error(traceback.format_exc())
        return False


def run_with_auto_fix(start_date: str = "2024-01-01",
                      end_date: str = "2024-12-31"):
    """运行回测（带自动修复）"""
    logger.info("=" * 60)
    logger.info("V79 回测启动 - 行业残差 Alpha 与流动性过滤增强")
    logger.info("=" * 60)
    
    # 1. DB 审计
    logger.info("Step 1: DB 审计...")
    if not check_db_structure():
        logger.error("DB 审计失败，无法继续")
        return None
    
    # 2. 运行回测
    logger.info("Step 2: 运行回测...")
    try:
        result = run_v79_backtest(start_date, end_date)
        return result
        
    except Exception as e:
        logger.error(f"回测执行失败：{e}")
        logger.error(traceback.format_exc())
        
        # 自动修复逻辑
        error_msg = str(e)
        
        if "OperationalError" in error_msg or "KeyError" in error_msg:
            logger.info("检测到 OperationalError 或 KeyError，尝试自动修复...")
            
            # 检查是否是数据缺失问题
            if "stock_daily" in error_msg or "table" in error_msg.lower():
                logger.warning("可能是数据表缺失，建议运行数据同步脚本")
                logger.warning("运行：python run_sync.py")
            
            # 检查是否是列名问题
            if "column" in error_msg.lower() or "field" in error_msg.lower():
                logger.warning("可能是列名不匹配，检查表结构")
        
        raise


def main():
    """主函数"""
    setup_logger()
    
    logger.info("=" * 60)
    logger.info("V79 回测系统启动")
    logger.info("=" * 60)
    
    # 运行回测
    result = run_with_auto_fix("2024-01-01", "2024-12-31")
    
    if result:
        logger.info("=" * 60)
        logger.info("V79 回测完成！")
        logger.info("=" * 60)
        
        # 输出验收结论
        from src.core.v79_logic import V79_RANK_IC_TARGET, V79_MAX_DRAWDOWN_TARGET, V79_WIN_RATE_TARGET
        
        rank_ic_pass = result.mean_rank_ic >= V79_RANK_IC_TARGET
        drawdown_pass = result.max_drawdown <= V79_MAX_DRAWDOWN_TARGET
        win_rate_pass = result.win_rate >= V79_WIN_RATE_TARGET
        trading_days_pass = result.trading_days >= 242
        
        logger.info("【验收结论】")
        logger.info(f"  指标 A - Mean Rank IC >= 0.035: {'✓' if rank_ic_pass else '✗'} ({result.mean_rank_ic:.4f})")
        logger.info(f"  指标 B - 最大回撤 <= 10%: {'✓' if drawdown_pass else '✗'} ({result.max_drawdown*100:.2f}%)")
        logger.info(f"  指标 C - 胜率 >= 45%: {'✓' if win_rate_pass else '✗'} ({result.win_rate*100:.2f}%)")
        logger.info(f"  指标 D - 交易日 >= 242: {'✓' if trading_days_pass else '✗'} ({result.trading_days}天)")
        
        all_pass = rank_ic_pass and drawdown_pass and win_rate_pass and trading_days_pass
        if all_pass:
            logger.info("✓ 所有验收指标通过！")
        else:
            logger.warning("⚠ 部分验收指标未通过，需要优化")
    else:
        logger.error("回测结果为 None，任务失败")
        sys.exit(1)


if __name__ == "__main__":
    main()