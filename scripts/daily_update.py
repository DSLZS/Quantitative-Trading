#!/usr/bin/env python3
# scripts/daily_update.py
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_fetcher import DataFetcher
from src.data.database_manager import DatabaseManager
from config.settings import MYSQL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/daily_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def update_single_stock(symbol: str, days: int = 30):
    """更新单只股票数据"""
    logger.info(f"开始更新股票: {symbol}")
    
    # 获取最近日期
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 获取数据
    fetcher = DataFetcher()
    df = fetcher.fetch_stock_daily(symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"未获取到 {symbol} 的数据")
        return False
    
    # 保存到数据库
    with DatabaseManager(MYSQL_CONFIG) as db:
        affected_rows = db.insert_stock_daily(df, replace=True)
        logger.info(f"{symbol} 更新完成，影响行数: {affected_rows}")
    
    return True

def update_stock_list(symbols: List[str], days: int = 30):
    """批量更新股票列表"""
    logger.info(f"开始批量更新 {len(symbols)} 只股票")
    
    # 获取数据
    fetcher = DataFetcher()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    df = fetcher.fetch_multiple_stocks(
        symbols, 
        start_date=start_date, 
        end_date=end_date,
        batch_size=20,
        delay=0.3
    )
    
    if df.empty:
        logger.error("未获取到任何数据")
        return False
    
    # 保存到数据库
    with DatabaseManager(MYSQL_CONFIG) as db:
        affected_rows = db.insert_stock_daily(df, replace=True)
        logger.info(f"批量更新完成，总记录数: {len(df)}，影响行数: {affected_rows}")
    
    return True

def update_all_a_shares(days: int = 30):
    """更新全部A股"""
    logger.info("开始更新全部A股数据")
    
    # 获取股票列表
    fetcher = DataFetcher()
    all_symbols = fetcher.fetch_stock_list()
    
    # 只取前100只作为示例（实际使用时可以全部更新）
    sample_symbols = all_symbols[:100]
    
    return update_stock_list(sample_symbols, days)

def main():
    parser = argparse.ArgumentParser(description='股票日数据更新脚本')
    parser.add_argument('--symbol', type=str, help='单只股票代码，如 sh000001')
    parser.add_argument('--file', type=str, help='股票代码列表文件路径')
    parser.add_argument('--market', type=str, choices=['sh', 'sz', 'bj'], help='更新指定市场')
    parser.add_argument('--days', type=int, default=30, help='更新最近多少天的数据')
    parser.add_argument('--all', action='store_true', help='更新全部A股')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    try:
        if args.symbol:
            # 更新单只股票
            update_single_stock(args.symbol, args.days)
            
        elif args.file:
            # 从文件读取股票列表
            with open(args.file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            update_stock_list(symbols, args.days)
            
        elif args.market:
            # 更新指定市场
            fetcher = DataFetcher()
            symbols = fetcher.fetch_stock_list(args.market)
            update_stock_list(symbols[:50], args.days)  # 只取前50只作为示例
            
        elif args.all:
            # 更新全部A股
            update_all_a_shares(args.days)
            
        else:
            # 默认更新主要指数
            default_symbols = ['sh000001', 'sh000300', 'sz399001', 'sz399006']
            update_stock_list(default_symbols, args.days)
            
        logger.info("数据更新任务完成")
        
    except Exception as e:
        logger.error(f"更新失败: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())