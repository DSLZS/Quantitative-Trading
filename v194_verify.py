from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    # 总记录数
    r = conn.execute(text("SELECT COUNT(*) FROM stock_daily WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")).fetchone()
    print(f"2025 年总记录数：{r[0]}")
    
    # 交易日数
    r = conn.execute(text("SELECT COUNT(DISTINCT trade_date) FROM stock_daily WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'")).fetchone()
    print(f"交易日数：{r[0]}")
    
    # 平均每日数据量
    r = conn.execute(text("""
        SELECT AVG(daily_cnt) FROM (
            SELECT trade_date, COUNT(*) as daily_cnt 
            FROM stock_daily 
            WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'
            GROUP BY trade_date
        ) t
    """)).fetchone()
    print(f"平均每日股票数：{r[0]:.0f}")
    
    # 示例日期
    print("\n示例日期数据量:")
    r = conn.execute(text("""
        SELECT trade_date, COUNT(*) as cnt 
        FROM stock_daily 
        WHERE trade_date >= '2025-01-01' AND trade_date <= '2025-12-31'
        GROUP BY trade_date 
        ORDER BY trade_date 
        LIMIT 5
    """)).fetchall()
    for row in r:
        print(f"  {row[0]}: {row[1]}条")