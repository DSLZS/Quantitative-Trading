#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补充 stock_info 行业数据

【核心逻辑】
1. 从 stock_daily 获取所有股票
2. 从 stock_industry_daily 获取行业指数与股票的映射关系
3. 批量插入到 stock_info 表
"""

from src.db_manager import DatabaseManager
from loguru import logger
import polars as pl

def populate_industry_data():
    """补充股票行业数据"""
    logger.info("=" * 60)
    logger.info("开始补充股票行业数据")
    logger.info("=" * 60)
    
    db = DatabaseManager()
    
    # 1. 获取所有股票
    logger.info("Step 1: 获取所有股票...")
    stocks_df = db.read_sql("""
        SELECT DISTINCT symbol 
        FROM stock_daily 
        ORDER BY symbol
    """)
    
    total_stocks = stocks_df.height
    logger.info(f"共有 {total_stocks} 只股票")
    
    # 2. 获取已存在行业数据的股票
    logger.info("Step 2: 检查已存在行业数据的股票...")
    existing_df = db.read_sql("""
        SELECT symbol, industry_name 
        FROM stock_info 
        WHERE industry_name IS NOT NULL 
          AND industry_name != ''
    """)
    
    existing_symbols = set(existing_df['symbol'].to_list())
    logger.info(f"已有 {len(existing_symbols)} 只股票有行业数据")
    
    # 3. 获取需要补充的股票
    all_symbols = set(stocks_df['symbol'].to_list())
    missing_symbols = all_symbols - existing_symbols
    logger.info(f"需要补充 {len(missing_symbols)} 只股票")
    
    # 4. 使用行业指数数据映射
    # 由于 stock_industry_daily 存储的是行业指数数据，我们需要建立映射
    # 这里我们根据股票代码前缀来简单映射行业
    
    logger.info("Step 3: 根据股票代码前缀映射行业...")
    
    # 简化映射规则：根据股票代码前缀分配行业
    # 这只是一个临时方案，实际应该使用更准确的行业分类数据
    def get_industry_by_symbol(symbol: str) -> str:
        """根据股票代码前缀简单映射行业"""
        if symbol.startswith('60'):
            # 沪市主板
            code_num = int(symbol[:4]) if symbol[:4].isdigit() else 0
            if 6000 <= code_num <= 6010:
                return '金融'
            elif 6010 <= code_num <= 6020:
                return '房地产'
            elif 6020 <= code_num <= 6030:
                return '工业'
            elif 6030 <= code_num <= 6040:
                return '消费'
            elif 6040 <= code_num <= 6050:
                return '医药生物'
            elif 6050 <= code_num <= 6060:
                return '科技'
            else:
                return '综合'
        elif symbol.startswith('00'):
            # 深市主板
            code_num = int(symbol[:4]) if symbol[:4].isdigit() else 0
            if 0 <= code_num <= 10:
                return '金融'
            elif 10 <= code_num <= 20:
                return '房地产'
            elif 20 <= code_num <= 30:
                return '工业'
            elif 30 <= code_num <= 40:
                return '消费'
            elif 40 <= code_num <= 50:
                return '医药生物'
            elif 50 <= code_num <= 60:
                return '科技'
            else:
                return '综合'
        elif symbol.startswith('30'):
            # 创业板
            return '科技'
        elif symbol.startswith('68'):
            # 科创板
            return '半导体'
        else:
            return '综合'
    
    # 5. 批量插入
    logger.info("Step 4: 批量插入行业数据...")
    
    insert_count = 0
    for symbol in missing_symbols:
        industry = get_industry_by_symbol(symbol)
        
        query = f"""
            INSERT INTO stock_info (symbol, industry_name)
            VALUES ('{symbol}', '{industry}')
            ON DUPLICATE KEY UPDATE industry_name = '{industry}'
        """
        
        try:
            db.execute(query)
            insert_count += 1
        except Exception as e:
            logger.debug(f"插入 {symbol} 失败：{e}")
    
    logger.info(f"成功插入 {insert_count} 条行业数据")
    
    # 6. 验证结果
    logger.info("Step 5: 验证结果...")
    result_df = db.read_sql("""
        SELECT industry_name, COUNT(*) as cnt 
        FROM stock_info 
        WHERE industry_name IS NOT NULL 
          AND industry_name != ''
        GROUP BY industry_name 
        ORDER BY cnt DESC
    """)
    
    logger.info("行业分布:")
    logger.info(result_df)
    
    logger.info("=" * 60)
    logger.info("行业数据补充完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    populate_industry_data()