#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V93 审计版 - 数据库字段检查脚本
检查 stock_daily 和 stock_industry_daily 的表结构
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db_manager import DatabaseManager
from sqlalchemy import inspect
import polars as pl

def check_table_structure(db: DatabaseManager, table_name: str) -> dict:
    """检查表结构"""
    inspector = inspect(db.engine)
    
    if not inspector.has_table(table_name):
        return {"exists": False, "columns": [], "error": f"表 {table_name} 不存在"}
    
    columns = inspector.get_columns(table_name)
    column_names = [col['name'] for col in columns]
    
    return {
        "exists": True,
        "columns": column_names,
        "column_count": len(column_names),
        "column_details": columns
    }

def check_industry_mapping(db: DatabaseManager) -> pl.DataFrame:
    """检查行业映射数据"""
    query = """
    SELECT DISTINCT industry_name, COUNT(*) as stock_count
    FROM stock_industry_daily
    GROUP BY industry_name
    ORDER BY stock_count DESC
    """
    try:
        df = db.read_sql(query)
        return df
    except Exception as e:
        print(f"查询行业映射失败：{e}")
        return pl.DataFrame()

def check_banking_stocks(db: DatabaseManager) -> pl.DataFrame:
    """检查银行股数据"""
    # 尝试多种方式查找银行相关行业
    queries = [
        "SELECT DISTINCT industry_name FROM stock_industry_daily WHERE industry_name LIKE '%银行%'",
        "SELECT DISTINCT industry_name FROM stock_industry_daily WHERE industry_name LIKE '%金融%'",
        "SELECT DISTINCT industry_name FROM stock_industry_daily",
    ]
    
    for i, query in enumerate(queries):
        try:
            df = db.read_sql(query)
            print(f"\n查询 {i+1} 结果:")
            print(df)
            if not df.is_empty():
                return df
        except Exception as e:
            print(f"查询 {i+1} 失败：{e}")
    
    return pl.DataFrame()

def sample_data_check(db: DatabaseManager, table_name: str, limit: int = 5) -> pl.DataFrame:
    """采样检查数据"""
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    try:
        df = db.read_sql(query)
        return df
    except Exception as e:
        print(f"采样查询失败：{e}")
        return pl.DataFrame()

def check_2024_data(db: DatabaseManager, table_name: str) -> pl.DataFrame:
    """检查 2024 年数据"""
    query = f"""
    SELECT 
        MIN(trade_date) as min_date, 
        MAX(trade_date) as max_date,
        COUNT(*) as row_count
    FROM {table_name}
    WHERE trade_date >= '20240101'
    """
    try:
        df = db.read_sql(query)
        return df
    except Exception as e:
        print(f"检查失败：{e}")
        return pl.DataFrame()

def main():
    print("=" * 80)
    print("V93 审计版 - 数据库字段检查")
    print("=" * 80)
    
    db = DatabaseManager()
    
    # 1. 检查 stock_daily 表结构
    print("\n[1] 检查 stock_daily 表结构...")
    stock_daily_info = check_table_structure(db, "stock_daily")
    if stock_daily_info["exists"]:
        print(f"  表存在：是")
        print(f"  列数：{stock_daily_info['column_count']}")
        print(f"  列名列表：{stock_daily_info['columns']}")
        
        # 检查关键字段 - 使用实际字段名
        required_columns = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing = [col for col in required_columns if col not in stock_daily_info['columns']]
        if missing:
            print(f"  ⚠️ 缺少关键字段：{missing}")
        else:
            print(f"  ✓ 关键字段完整")
        
        # 检查是否有 ts_code 字段（需要映射）
        if 'ts_code' not in stock_daily_info['columns']:
            print(f"  ⚠️ 无 ts_code 字段，使用 symbol 字段代替")
        if 'vol' not in stock_daily_info['columns']:
            print(f"  ⚠️ 无 vol 字段，使用 volume 字段代替")
        
        # 采样数据
        print("\n  采样数据 (前 5 行):")
        sample = sample_data_check(db, "stock_daily", 5)
        if not sample.is_empty():
            print(sample)
    else:
        print(f"  ❌ {stock_daily_info.get('error', '表不存在')}")
    
    # 2. 检查 stock_industry_daily 表结构
    print("\n[2] 检查 stock_industry_daily 表结构...")
    industry_info = check_table_structure(db, "stock_industry_daily")
    if industry_info["exists"]:
        print(f"  表存在：是")
        print(f"  列数：{industry_info['column_count']}")
        print(f"  列名列表：{industry_info['columns']}")
        
        # 采样数据
        print("\n  采样数据 (前 5 行):")
        sample = sample_data_check(db, "stock_industry_daily", 5)
        if not sample.is_empty():
            print(sample)
    else:
        print(f"  ❌ {industry_info.get('error', '表不存在')}")
    
    # 3. 检查行业映射
    print("\n[3] 检查行业映射分布...")
    industry_dist = check_industry_mapping(db)
    if not industry_dist.is_empty():
        print(industry_dist)
        
        # 检查是否包含"银行"行业
        if 'industry_name' in industry_dist.columns:
            banking_industries = industry_dist.filter(
                pl.col("industry_name").str.contains("银行", literal=True)
            )
            if not banking_industries.is_empty():
                print(f"\n  ✓ 发现银行相关行业：{banking_industries['industry_name'].to_list()}")
            else:
                print(f"\n  ⚠️ 未发现包含'银行'的行业名称")
                # 显示所有行业名称供检查
                all_industries = industry_dist['industry_name'].to_list()
                print(f"  所有行业名称：{all_industries}")
    else:
        print("  无法获取行业映射数据")
    
    # 4. 专门检查银行股
    print("\n[4] 专门检查银行相关行业...")
    check_banking_stocks(db)
    
    # 5. 检查 2024 年数据是否存在
    print("\n[5] 检查 2024 年数据覆盖情况...")
    for table in ["stock_daily", "stock_industry_daily"]:
        print(f"  {table} (2024 年数据):")
        df = check_2024_data(db, table)
        if not df.is_empty() and df.height > 0:
            print(f"    日期范围：{df['min_date'][0]} - {df['max_date'][0]}")
            print(f"    行数：{df['row_count'][0]}")
        else:
            print(f"    ❌ 无 2024 年数据")
    
    print("\n" + "=" * 80)
    print("检查完成")
    print("=" * 80)

if __name__ == "__main__":
    main()