#!/usr/bin/env python3
# scripts/daily_update.py
'''
daily_update.py代码用于拉取股票数据
'''
import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from typing import List
import pandas as pd

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_fetcher import DataFetcher
from src.data.database_manager import DatabaseManager
from config.settings import MYSQL_CONFIG

'''
运行示例
# 支持多种输入格式
python scripts/daily_update.py --symbol 510300.SH
python scripts/daily_update.py --symbol sh510300
python scripts/daily_update.py --symbol 510300
更新上证指数最近30天数据
python scripts/daily_update.py --symbol 000001.SH --days 30
更新较长时间 更新贵州茅台1年数据
python scripts/daily_update.py --symbol 600519.SH --days 365
'''
'''
从文件更新
股票列表文件stock_list.txt，每行一个股票代码（标准格式）
000001.SH  # 上证指数
399001.SZ  # 深证成指
600519.SH  # 贵州茅台
000858.SZ  # 五粮液
601318.SH  # 中国平安
000002.SZ  # 万科A
600036.SH  # 招商银行
300750.SZ  # 宁德时代
python scripts/daily_update.py --file stock_list.txt --days 60
'''

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
    # 标准化输入符号
    fetcher = DataFetcher()
    normalized_symbol = fetcher.normalize_symbol(symbol)
    logger.info(f"开始更新股票: {normalized_symbol} (原始输入: {symbol})")
    
    # 获取最近日期
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 获取数据（使用标准化后的符号）
    df = fetcher.fetch_stock_daily(normalized_symbol, start_date, end_date)
    
    if df.empty:
        logger.error(f"未获取到 {normalized_symbol} 的数据")
        return False
    
    # 保存到数据库（使用连接池）
    db = DatabaseManager()
    affected_rows = db.insert_stock_daily(df, replace=True)
    logger.info(f"{normalized_symbol} 更新完成，影响行数: {affected_rows}")
    
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
    
    # 保存到数据库（使用连接池）
    db = DatabaseManager()
    affected_rows = db.insert_stock_daily(df, replace=True)
    logger.info(f"批量更新完成，总记录数: {len(df)}，影响行数: {affected_rows}")
    
    return True

def update_all_a_shares(days: int = 30):
    """更新全部A股"""
    logger.info("开始更新全部A股数据")
    
    # 获取股票列表（返回标准格式）
    fetcher = DataFetcher()
    all_symbols = fetcher.fetch_stock_list()
    
    # 只取前100只作为示例
    sample_symbols = all_symbols[:100]
    
    return update_stock_list(sample_symbols, days)

def main():
    parser = argparse.ArgumentParser(description='股票日数据更新脚本')
    parser.add_argument('--symbol', type=str, help='单只股票代码，如 000001.SH 或 sh000001')
    parser.add_argument('--file', type=str, help='股票代码列表文件路径')
    parser.add_argument('--market', type=str, choices=['SH', 'SZ', 'BJ'], help='更新指定市场')
    parser.add_argument('--days', type=int, default=30, help='更新最近多少天的数据')
    parser.add_argument('--all', action='store_true', help='更新全部A股')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    try:
        if args.symbol:
            # 直接使用标准化函数
            fetcher = DataFetcher()
            normalized_symbol = fetcher.normalize_symbol(args.symbol)
            update_single_stock(normalized_symbol, args.days)
            
        elif args.file:
            # 从文件读取股票列表
            with open(args.file, 'r') as f:
                symbols = [line.strip() for line in f if line.strip()]
            
            # 批量标准化
            fetcher = DataFetcher()
            normalized_symbols = []
            for s in symbols:
                # 跳过注释行
                if s.startswith('#'):
                    continue
                # 提取代码（可能后面有注释）
                code_part = s.split('#')[0].strip()
                if code_part:
                    normalized_symbols.append(fetcher.normalize_symbol(code_part))
            
            # 显示转换结果
            if symbols != normalized_symbols:
                logger.info("符号转换结果:")
                for orig, new in zip(symbols, normalized_symbols):
                    if orig != new:
                        logger.info(f"  {orig} -> {new}")
            
            update_stock_list(normalized_symbols, args.days)
            
        elif args.market:
            # 更新指定市场（这里market应该是标准格式：SH/SZ/BJ）
            fetcher = DataFetcher()
            symbols = fetcher.fetch_stock_list(args.market)
            update_stock_list(symbols[:50], args.days)
            
        elif args.all:
            # 更新全部A股
            update_all_a_shares(args.days)
            
        else:
            # 默认更新主要指数（已经使用标准格式）
            default_symbols = ['000001.SH', '000300.SH', '399001.SZ', '399006.SZ']
            update_stock_list(default_symbols, args.days)
            
        logger.info("数据更新任务完成")
        
    except Exception as e:
        logger.error(f"更新失败: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())