"""检查 V82 所需数据完整性"""
from src.db_manager import DatabaseManager

db = DatabaseManager()

print("=" * 60)
print("V82 数据完整性检查")
print("=" * 60)

for year in ['2019', '2021', '2024']:
    query = f"""
    SELECT COUNT(DISTINCT symbol) as stock_count, COUNT(*) as total_rows, 
           MIN(trade_date) as min_date, MAX(trade_date) as max_date
    FROM stock_daily
    WHERE trade_date >= '{year}-01-01' AND trade_date <= '{year}-12-31'
    """
    try:
        df = db.read_sql(query)
        if not df.is_empty():
            print(f'{year}年：股票数={int(df["stock_count"][0])}, 总行数={int(df["total_rows"][0])}, 日期范围={df["min_date"][0]} 至 {df["max_date"][0]}')
        else:
            print(f'{year}年：无数据')
    except Exception as e:
        print(f'{year}年：查询失败 - {e}')

print("=" * 60)