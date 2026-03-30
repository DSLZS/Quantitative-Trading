"""测试 V89 数据加载"""

from src.db_manager import get_db
import polars as pl

db = get_db()

# 测试简单查询
query = """
    SELECT symbol, trade_date, open, high, low, close, volume, amount, 
           pct_chg, industry_code, total_mv, is_st
    FROM stock_daily
    WHERE trade_date >= '2019-01-01' 
      AND trade_date <= '2019-01-31'
    ORDER BY symbol, trade_date
    LIMIT 100
"""

print("执行查询...")
try:
    df = db.read_sql(query)
    print(f"查询成功！行数：{len(df)}")
    print(f"列名：{df.columns}")
    print(df.head())
except Exception as e:
    print(f"查询失败：{e}")