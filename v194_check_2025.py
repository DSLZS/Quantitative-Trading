"""检查 2025 年数据"""
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(os.getenv('DATABASE_URL'))

with engine.connect() as conn:
    # 检查 2025 年数据
    result = conn.execute(text("""
        SELECT trade_date, COUNT(*) as cnt 
        FROM stock_daily 
        WHERE trade_date >= '2025-01-01' AND trade_date < '2026-01-01'
        GROUP BY trade_date 
        ORDER BY trade_date 
        LIMIT 10
    """))
    print('2025 年数据示例:')
    for row in result.fetchall():
        print(f'  {row[0]}: {row[1]} rows')
    
    # 统计总数
    result = conn.execute(text("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date)
        FROM stock_daily 
        WHERE trade_date >= '2025-01-01' AND trade_date < '2026-01-01'
    """))
    row = result.fetchone()
    print(f'\n2025 年总计：{row[0]} 条记录，{row[1]} 个交易日')
    
    # 检查平均每日股票数
    result = conn.execute(text("""
        SELECT AVG(daily_cnt) as avg_daily
        FROM (
            SELECT trade_date, COUNT(symbol) as daily_cnt
            FROM stock_daily
            WHERE trade_date >= '2025-01-01' AND trade_date < '2026-01-01'
            GROUP BY trade_date
        ) t
    """))
    row = result.fetchone()
    print(f'平均每日股票数：{row[0]:.0f}')