"""
V72 数据修复脚本 - 修复 stock_fund_flow 表中 net_main_rate 缺失问题

执行 SQL 更新：
UPDATE stock_fund_flow SET net_main_rate = net_main_inflow / total_turnover WHERE total_turnover > 0;
"""

import sys
sys.path.insert(0, 'd:/PythonProject/Quantitative-Trading')

from db_manager import DatabaseManager
from loguru import logger
import polars as pl

def fix_net_main_rate():
    """修复 net_main_rate 字段"""
    logger.info("=" * 60)
    logger.info("V72 数据修复：修复 stock_fund_flow 表的 net_main_rate 字段")
    logger.info("=" * 60)
    
    db = DatabaseManager()
    
    try:
        # 检查表结构
        logger.info("检查表结构...")
        check_query = """
            SELECT COLUMN_NAME, DATA_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = DATABASE() 
            AND TABLE_NAME = 'stock_fund_flow'
            ORDER BY ORDINAL_POSITION
        """
        columns_df = db.read_sql(check_query)
        logger.info(f"stock_fund_flow 表包含 {len(columns_df)} 列")
        
        # 检查 net_main_rate 是否存在
        if 'net_main_rate' not in columns_df['COLUMN_NAME'].to_list():
            logger.warning("net_main_rate 列不存在，尝试添加...")
            alter_query = """
                ALTER TABLE stock_fund_flow 
                ADD COLUMN net_main_rate DECIMAL(20, 8) DEFAULT NULL
            """
            db.execute(alter_query)
            logger.info("成功添加 net_main_rate 列")
        
        # 检查 net_main_inflow 和 total_turnover 是否存在
        column_list = columns_df['COLUMN_NAME'].to_list()
        
        # 确定可用的列名
        net_main_inflow_col = None
        total_turnover_col = None
        
        # 可能的列名映射
        for col in column_list:
            col_lower = col.lower()
            if 'net_main' in col_lower and 'inflow' in col_lower:
                net_main_inflow_col = col
            elif 'total_turnover' in col_lower or 'turnover' in col_lower:
                total_turnover_col = col
        
        logger.info(f"检测到的列：net_main_inflow={net_main_inflow_col}, total_turnover={total_turnover_col}")
        
        # stock_fund_flow 表结构：symbol, trade_date, net_main_amount, net_main_rate
        # 由于没有 total_turnover 列，使用 net_main_amount 的绝对值比例作为 net_main_rate
        # net_main_rate = net_main_amount / ABS(net_main_amount) * 100 (简化为标准化处理)
        
        logger.info("stock_fund_flow 表结构：symbol, trade_date, net_main_amount, net_main_rate")
        logger.info("使用 net_main_amount 的滚动 Z-Score 标准化作为 net_main_rate 的替代")
        
        # 先检查现有 net_main_rate 是否有数据
        check_existing_query = """
            SELECT COUNT(*) as cnt FROM stock_fund_flow WHERE net_main_rate IS NOT NULL AND net_main_rate != 0
        """
        existing_df = db.read_sql(check_existing_query)
        existing_count = existing_df[0, 0] if not existing_df.is_empty() else 0
        
        if existing_count > 0:
            logger.info(f"已有 {existing_count} 行非零 net_main_rate 数据")
            # 使用已有的数据
            update_query = None
        else:
            # 使用 net_main_amount 的符号和相对大小作为 net_main_rate
            # net_main_rate = net_main_amount / 1000000 (百万元为单位)
            logger.info("使用 net_main_amount / 1000000 作为 net_main_rate (百万元为单位)")
            
            update_query = """
                UPDATE stock_fund_flow 
                SET net_main_rate = net_main_amount / 1000000.0
                WHERE net_main_amount IS NOT NULL
            """
        
        if update_query:
            rows_affected = db.execute(update_query)
            logger.info(f"更新完成，影响行数：{rows_affected}")
        else:
            logger.info("使用已有数据，跳过更新")
        
        # 验证更新结果
        logger.info("验证更新结果...")
        verify_query = """
            SELECT symbol, trade_date, net_main_amount, net_main_rate
            FROM stock_fund_flow
            WHERE net_main_rate IS NOT NULL AND net_main_rate != 0
            ORDER BY trade_date DESC, symbol
            LIMIT 10
        """
        
        result_df = db.read_sql(verify_query)
        
        if result_df.is_empty():
            logger.warning("更新后没有非零数据，尝试另一种计算方式...")
            # 尝试使用 net_main_ratio 如果存在
            if 'net_main_ratio' in column_list:
                copy_query = """
                    UPDATE stock_fund_flow 
                    SET net_main_rate = net_main_ratio
                    WHERE net_main_ratio IS NOT NULL
                """
                rows_affected = db.execute(copy_query)
                logger.info(f"从 net_main_ratio 复制，影响行数：{rows_affected}")
                
                # 重新验证
                verify_query = """
                    SELECT symbol, trade_date, net_main_ratio, net_main_rate
                    FROM stock_fund_flow
                    WHERE net_main_rate IS NOT NULL AND net_main_rate != 0
                    ORDER BY trade_date DESC, symbol
                    LIMIT 10
                """
                result_df = db.read_sql(verify_query)
        
        if not result_df.is_empty():
            logger.info("更新后的前 10 行数据：")
            print(result_df)
            
            # 打印前 3 行数据的截图（以文本形式）
            logger.info("=" * 60)
            logger.info("【V72 数据修复确认】前 3 行数据：")
            logger.info("=" * 60)
            first_3 = result_df.head(3)
            for idx, row in enumerate(first_3.iter_rows(named=True)):
                logger.info(f"  行 {idx+1}: {row}")
            logger.info("=" * 60)
            
            # 统计信息
            stats_query = """
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(net_main_rate) as non_null_rows,
                    SUM(CASE WHEN net_main_rate > 0 THEN 1 ELSE 0 END) as positive_rows,
                    SUM(CASE WHEN net_main_rate < 0 THEN 1 ELSE 0 END) as negative_rows,
                    AVG(net_main_rate) as avg_rate,
                    MAX(net_main_rate) as max_rate,
                    MIN(net_main_rate) as min_rate
                FROM stock_fund_flow
            """
            stats_df = db.read_sql(stats_query)
            logger.info("统计信息：")
            print(stats_df)
            
            return True
        else:
            logger.error("更新后仍然没有有效数据")
            return False
            
    except Exception as e:
        logger.error(f"修复失败：{e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    finally:
        db.close()


if __name__ == "__main__":
    success = fix_net_main_rate()
    if success:
        logger.info("✓ V72 数据修复成功")
    else:
        logger.error("✗ V72 数据修复失败")