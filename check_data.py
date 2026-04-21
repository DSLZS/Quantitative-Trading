"""检查现有数据"""
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import os

load_dotenv()
db_url = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
engine = create_engine(db_url, poolclass=QueuePool, pool_pre_ping=True)

print('='*60)
print('现有数据检查')
print('='*60)

for table in ['stock_daily', 'index_daily', 'stock_industry_daily', 'stock_fund_flow']:
    print(f'\n{table}:')
    try:
        with engine.connect() as conn:
            # 检查表是否存在
            result = conn.execute(text(f"SHOW TABLES LIKE '{table}'"))
            if not result.fetchone():
                print('  表不存在')
                continue
            
            # 获取年份统计
            query = text(f"SELECT YEAR(trade_date) as year, COUNT(*) as cnt FROM {table} GROUP BY year ORDER BY year")
            result = conn.execute(query)
            for row in result.fetchall():
                print(f'  {row[0]}: {row[1]:,} rows')
    except Exception as e:
        print(f'  错误：{e}')

print('\n' + '='*60)