#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票日线数据修复脚本
- 修复 adj_factor 为 NULL 或 0 的问题
- 修复 close 价格为负的问题
- 清理重复数据
"""

import os
import sys
from typing import Optional
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# 配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "quantitative_trading")


class StockDailyFixer:
    """股票日线数据修复器"""
    
    def __init__(self):
        self.database_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            
    def fix_adj_factor(self):
        """修复 adj_factor 为 NULL 或 0 的问题"""
        logger.info("========== 修复 adj_factor ==========")
        
        # 将 NULL 或 0 的 adj_factor 更新为 1.0
        update_sql = """
            UPDATE stock_daily 
            SET adj_factor = 1.000000 
            WHERE adj_factor IS NULL OR adj_factor = 0
        """
        
        with self.get_connection() as conn:
            result = conn.execute(text(update_sql))
            conn.commit()
            updated_count = result.rowcount
            logger.info(f"修复了 {updated_count:,} 条 adj_factor 记录")
            
    def fix_negative_close(self):
        """修复 close 价格为负的问题 - 删除这些异常记录"""
        logger.info("========== 修复 close 价格为负 ==========")
        
        # 统计负值记录
        count_sql = "SELECT COUNT(*) FROM stock_daily WHERE close < 0"
        with self.get_connection() as conn:
            result = conn.execute(text(count_sql))
            negative_count = result.scalar()
            logger.info(f"发现 {negative_count:,} 条 close 为负的记录")
            
        if negative_count > 0:
            # 删除负值记录
            delete_sql = "DELETE FROM stock_daily WHERE close < 0"
            with self.get_connection() as conn:
                result = conn.execute(text(delete_sql))
                conn.commit()
                deleted_count = result.rowcount
                logger.info(f"删除了 {deleted_count:,} 条 close 为负的记录")
                
    def check_duplicate_symbols(self):
        """检查重复的股票代码"""
        logger.info("========== 检查重复数据 ==========")
        
        # 检查同一 symbol 和 trade_date 的重复
        check_sql = """
            SELECT symbol, trade_date, COUNT(*) as cnt
            FROM stock_daily
            GROUP BY symbol, trade_date
            HAVING COUNT(*) > 1
        """
        
        with self.get_connection() as conn:
            result = conn.execute(text(check_sql))
            duplicates = result.fetchall()
            
        if duplicates:
            logger.warning(f"发现 {len(duplicates)} 组重复记录")
            for dup in duplicates[:10]:
                logger.warning(f"  {dup[0]}, {dup[1]}: {dup[2]} 条")
        else:
            logger.info("没有发现重复记录")
            
    def verify_final_data(self):
        """最终数据验证"""
        logger.info("\n========== 最终数据验证 ==========")
        
        # 按年份统计
        verify_sql = """
            SELECT YEAR(trade_date) as year, COUNT(*) as count 
            FROM stock_daily 
            WHERE YEAR(trade_date) IN (2018, 2020, 2022)
            GROUP BY YEAR(trade_date) 
            ORDER BY year
        """
        
        with self.get_connection() as conn:
            result = conn.execute(text(verify_sql))
            rows = result.fetchall()
            
            logger.info("\n目标年份数据量:")
            all_passed = True
            for year, count in rows:
                status = "✓" if count >= 800000 else "✗"
                if count < 800000:
                    all_passed = False
                logger.info(f"  {year} 年：{count:,} 条 {status}")
                
        # 检查 adj_factor
        adj_check_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE adj_factor IS NULL OR adj_factor = 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(adj_check_sql))
            null_count = result.scalar()
            status = "✓" if null_count == 0 else "✗"
            logger.info(f"\nadj_factor 为空或 0 的记录：{null_count:,} {status}")
            
        # 检查 close 价格为负
        negative_close_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE close < 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(negative_close_sql))
            negative_count = result.scalar()
            status = "✓" if negative_count == 0 else "✗"
            logger.info(f"close 价格为负的记录：{negative_count:,} {status}")
            
        # 随机抽取 3 天检查
        sample_sql = """
            SELECT DISTINCT trade_date FROM stock_daily 
            WHERE YEAR(trade_date) IN (2018, 2020, 2022)
            ORDER BY RAND() LIMIT 3
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sample_sql))
            sample_dates = [str(row[0]) for row in result.fetchall()]
            
            logger.info("\n随机抽取 3 天数据详情 (目标年份):")
            for sample_date in sample_dates:
                detail_sql = text("""
                    SELECT symbol, trade_date, close, adj_factor 
                    FROM stock_daily 
                    WHERE trade_date = :trade_date 
                    LIMIT 5
                """)
                result = conn.execute(detail_sql, {"trade_date": sample_date})
                rows = result.fetchall()
                logger.info(f"\n  {sample_date}:")
                for row in rows:
                    logger.info(f"    {row[0]}: close={row[2]}, adj_factor={row[3]}")
                    
        return all_passed and null_count == 0 and negative_count == 0
            
    def run(self):
        """主运行函数"""
        logger.info("========== 股票日线数据修复任务启动 ==========")
        
        # 执行修复
        self.fix_adj_factor()
        self.fix_negative_close()
        self.check_duplicate_symbols()
        
        # 验证
        success = self.verify_final_data()
        
        if success:
            logger.info("\n========== 所有验证通过 ==========")
        else:
            logger.warning("\n========== 部分验证未通过，请检查 ==========")
            

def main():
    """主函数"""
    try:
        fixer = StockDailyFixer()
        fixer.run()
    except Exception as e:
        logger.error(f"任务执行失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()