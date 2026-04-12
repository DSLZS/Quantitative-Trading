"""V194 数据恢复 - 缺口扫描"""
import os
import time
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv('DATABASE_URL')
engine = create_engine(DB_URL, pool_pre_ping=True)

print("=" * 60)
print("2025 年数据缺口扫描")
print("=" * 60)

with engine.connect() as conn:
    # 检查 2025 年每日数据量
    query = text("""
        SELECT trade_date, COUNT(symbol) as cnt
        FROM stock_daily
        WHERE trade_date >= '20250101' AND trade_date < '20260101'
        GROUP BY trade_date
        ORDER BY trade_date
    """)
    result = conn.execute(query)
    rows = [(row[0], row[1]) for row in result.fetchall()]

total_days = len(rows)
total_rows = sum(r[1] for r in rows)
avg_daily = total_rows / total_days if total_days > 0 else 0
missing_dates = [(d, c) for d, c in rows if c < 5000]

print(f"2025 年已有数据:")
print(f"  交易日期数：{total_days}")
print(f"  总记录数：{total_rows}")
print(f"  平均每日股票数：{avg_daily:.0f}")
print(f"  不达标日期数 (<5000 条): {len(missing_dates)}")

if missing_dates:
    print(f"\n不达标日期 (前 20 个):")
    for d, c in missing_dates[:20]:
        print(f"  {d}: {c}条")
    if len(missing_dates) > 20:
        print(f"  ... 还有{len(missing_dates) - 20}个")

# 获取 2025 年所有交易日历
print("\n" + "=" * 60)
print("获取交易日历...")
try:
    import tushare as ts
    TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    cal_df = pro.trade_cal(exchange='SSE', start_date='20250101', end_date='20251231')
    trade_dates = cal_df[cal_df['is_open'] == '1']['cal_date'].tolist()
    print(f"2025 年交易日总数：{len(trade_dates)}")
    
    db_dates = set(r[0] for r in rows)
    cal_dates = set(trade_dates)
    
    missing_cal = cal_dates - db_dates
    print(f"完全缺失的交易日：{len(missing_cal)}个")
    if missing_cal:
        sorted_missing = sorted(list(missing_cal))[:20]
        print(f"  缺失日期：{', '.join(sorted_missing)}")
        if len(missing_cal) > 20:
            print(f"  ... 共{len(missing_cal)}个")
except Exception as e:
    print(f"获取交易日历失败：{type(e).__name__}")

print("=" * 60)