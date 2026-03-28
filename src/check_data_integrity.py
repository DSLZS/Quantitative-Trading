"""
V83 数据完整性检查脚本

【检查标准】
- 2019 年：stock_daily 记录必须 > 500,000 条
- 2021 年：stock_daily 记录必须 > 500,000 条
- 2024 年：stock_daily 记录必须 > 500,000 条

【执行逻辑】
1. 若数量不达标，提示运行 v83_data_repairer.py
2. 输出每个年份的有效交易天数

作者：量化系统
版本：V83.0
日期：2026-03-28
"""

import sys
import os
from typing import Dict, Tuple, Any
from datetime import datetime
from loguru import logger

# 尝试导入数据库管理器
try:
    from src.db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    try:
        from db_manager import DatabaseManager, get_db
        DB_AVAILABLE = True
    except ImportError:
        DB_AVAILABLE = False
        logger.error("V83: db_manager 模块未找到")

# 导入 V83 配置
try:
    from src.core.v83_logic import V83_MIN_STOCK_DAILY_ROWS, V83_RANK_IC_OOS_YEARS
except ImportError:
    V83_MIN_STOCK_DAILY_ROWS = 500000
    V83_RANK_IC_OOS_YEARS = ["2019", "2021", "2024"]


# ===========================================
# V83 数据完整性检查器
# ===========================================

class V83DataIntegrityChecker:
    """V83 数据完整性检查器"""
    
    def __init__(self, db=None):
        if db is None and DB_AVAILABLE:
            try:
                self.db = get_db()
            except Exception as e:
                logger.error(f"V83: 数据库连接失败 - {e}")
                self.db = None
        else:
            self.db = db
        
        self.required_years = V83_RANK_IC_OOS_YEARS
        self.min_rows = V83_MIN_STOCK_DAILY_ROWS
    
    def check_year_integrity(self, year: str) -> Tuple[bool, Dict[str, Any]]:
        """
        检查指定年份的数据完整性
        
        Parameters
        ----------
        year : str
            年份
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            (是否通过，详细信息)
        """
        if self.db is None:
            return False, {
                'error': '数据库连接未初始化',
                'stock_count': 0,
                'total_rows': 0,
                'trading_days': 0,
                'date_range': (None, None),
            }
        
        try:
            # 查询股票数量、总行数、交易天数、日期范围
            query = f"""
                SELECT 
                    COUNT(DISTINCT symbol) as stock_count,
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT trade_date) as trading_days,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date
                FROM stock_daily
                WHERE trade_date >= '{year}-01-01' 
                  AND trade_date <= '{year}-12-31'
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return False, {
                    'error': '无法查询 stock_daily 表',
                    'stock_count': 0,
                    'total_rows': 0,
                    'trading_days': 0,
                    'date_range': (None, None),
                }
            
            stock_count = int(df['stock_count'][0]) if 'stock_count' in df.columns else 0
            total_rows = int(df['total_rows'][0]) if 'total_rows' in df.columns else 0
            trading_days = int(df['trading_days'][0]) if 'trading_days' in df.columns else 0
            min_date = str(df['min_date'][0]) if 'min_date' in df.columns else None
            max_date = str(df['max_date'][0]) if 'max_date' in df.columns else None
            
            # 判断是否通过
            passed = total_rows >= self.min_rows
            
            return passed, {
                'stock_count': stock_count,
                'total_rows': total_rows,
                'trading_days': trading_days,
                'date_range': (min_date, max_date),
                'min_required': self.min_rows,
                'deficit': max(0, self.min_rows - total_rows),
            }
            
        except Exception as e:
            return False, {
                'error': f'检查失败：{e}',
                'stock_count': 0,
                'total_rows': 0,
                'trading_days': 0,
                'date_range': (None, None),
            }
    
    def check_all_years(self) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """检查所有必需年份的数据完整性"""
        results = {}
        
        for year in self.required_years:
            passed, details = self.check_year_integrity(year)
            results[year] = (passed, details)
        
        return results
    
    def print_report(self, results: Dict[str, Tuple[bool, Dict[str, Any]]]) -> bool:
        """
        打印数据完整性报告
        
        Parameters
        ----------
        results : Dict[str, Tuple[bool, Dict[str, Any]]]
            检查结果
            
        Returns
        -------
        bool
            是否所有年份都通过
        """
        logger.info("=" * 60)
        logger.info("V83 数据完整性检查报告")
        logger.info("=" * 60)
        logger.info(f"检查标准：每年 stock_daily 记录 > {self.min_rows:,} 条")
        logger.info(f"检查年份：{', '.join(self.required_years)}")
        logger.info("")
        
        all_passed = True
        total_rows_sum = 0
        total_trading_days = 0
        
        for year, (passed, details) in results.items():
            error = details.get('error')
            if error:
                logger.error(f"{year}年：✗ {error}")
                all_passed = False
                continue
            
            stock_count = details['stock_count']
            total_rows = details['total_rows']
            trading_days = details['trading_days']
            min_date, max_date = details['date_range']
            deficit = details['deficit']
            
            total_rows_sum += total_rows
            total_trading_days += trading_days
            
            status = "✓" if passed else "✗"
            logger.info(f"{year}年：{status} 股票数={stock_count}, 总行数={total_rows:,}, "
                       f"交易天数={trading_days}, 日期范围={min_date} 至 {max_date}")
            
            if not passed:
                logger.warning(f"    └─ 数据缺口：{deficit:,} 条记录")
                all_passed = False
        
        logger.info("")
        logger.info("-" * 60)
        logger.info(f"总计：{len(self.required_years)}年，总行数={total_rows_sum:,}, "
                   f"总交易天数={total_trading_days}")
        
        required_total = self.min_rows * len(self.required_years)
        completion_rate = total_rows_sum / required_total if required_total > 0 else 0.0
        logger.info(f"数据完整性：{completion_rate:.2%} (要求：≥99%)")
        
        logger.info("")
        if all_passed and completion_rate >= 0.99:
            logger.info("【结果】✓ 数据完整性检查通过")
        else:
            logger.warning("【结果】✗ 数据完整性检查未通过")
            logger.warning("")
            logger.warning("【建议】请运行以下命令修复数据：")
            logger.warning("  python src/v83_data_repairer.py")
            logger.warning("")
            logger.warning("v83_data_repairer.py 具备以下特性：")
            logger.warning("  - Exponential Backoff（指数退避重试机制）")
            logger.warning("  - Connection Aborted 自动 sleep(60) 并断点续传")
            logger.warning("  - 跳过失败股票并记录到 failed_symbols.txt")
            logger.warning("  - 全部运行完后自动重试失败列表")
        
        logger.info("=" * 60)
        
        return all_passed and completion_rate >= 0.99


# ===========================================
# 主程序
# ===========================================

def setup_logging() -> None:
    """配置日志"""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/v83_data_integrity_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )


def main() -> int:
    """主函数"""
    setup_logging()
    
    logger.info("=" * 60)
    logger.info("V83 数据完整性检查器 - 启动")
    logger.info("=" * 60)
    
    # 检查数据库是否可用
    if not DB_AVAILABLE:
        logger.error("V83: db_manager 模块未找到")
        return 1
    
    # 初始化数据库
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V83: 数据库连接失败 - {e}")
        return 1
    
    # 初始化检查器
    checker = V83DataIntegrityChecker(db=db)
    
    # 执行检查
    results = checker.check_all_years()
    
    # 打印报告
    all_passed = checker.print_report(results)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())