"""V194 数据修复 - 使用原生 SQL 执行 REPLACE INTO"""
import os
import time
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import tushare as ts

load_dotenv()

DB_URL = os.getenv('DATABASE_URL')
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()
engine = create_engine(DB_URL, pool_pre_ping=True)

SLEEP_BETWEEN_REQUESTS = 2

def get_trade_dates(year: int) -> list:
    df = pro.trade_cal(exchange='SZSE', start_date=f"{year}0101", end_date=f"{year}1231")
    if df is not None and not df.empty:
        return df[df['is_open'] == 1]['cal_date'].tolist()
    return []

def fetch_data(trade_date: str) -> pd.DataFrame:
    try:
        df = pro.daily(trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        if '429' in str(e) or '频率' in str(e) or '积分' in str(e):
            print("[LIMIT] Waiting 60s...")
            time.sleep(60)
            return fetch_data(trade_date)
    return pd.DataFrame()

def escape_value(val):
    """转义 SQL 值"""
    if val is None or pd.isna(val):
        return 'NULL'
    if isinstance(val, str):
        return f"'{val.replace("'", "''")}'"
    if isinstance(val, (int, float)):
        return str(val)
    return f"'{val}'"

def upsert_batch(df: pd.DataFrame):
    """使用 REPLACE INTO 批量插入"""
    if df is None or df.empty:
        return 0
    
    df = df.copy()
    df = df.rename(columns={'ts_code': 'symbol', 'vol': 'volume'})
    df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
    
    for col in ['adj_factor', 'turnover_rate', 'industry_code', 'total_mv', 'is_st']:
        if col not in df.columns:
            df[col] = '' if col == 'industry_code' else (0 if col in ['total_mv', 'is_st'] else None)
    
    cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 
            'change', 'pct_chg', 'volume', 'amount', 'adj_factor', 'turnover_rate',
            'industry_code', 'total_mv', 'is_st']
    df = df[cols]
    df = df.drop_duplicates(subset=['symbol', 'trade_date'])
    df = df.dropna(subset=['symbol', 'trade_date'])
    
    total = 0
    # 从 SQLAlchemy engine 获取连接参数
    conn_info = engine.url
    
    with pymysql.connect(
        host=conn_info.host,
        user=conn_info.username,
        password=conn_info.password,
        database=conn_info.database,
        charset='utf8mb4'
    ) as conn:
        with conn.cursor() as cursor:
            for i in range(0, len(df), 50):
                chunk = df.iloc[i:i+50]
                values = []
                for _, row in chunk.iterrows():
                    v = (
                        escape_value(row['symbol']),
                        escape_value(row['trade_date']),
                        escape_value(row['open']),
                        escape_value(row['high']),
                        escape_value(row['low']),
                        escape_value(row['close']),
                        escape_value(row['pre_close']),
                        escape_value(row['change']),
                        escape_value(row['pct_chg']),
                        escape_value(row['volume']),
                        escape_value(row['amount']),
                        escape_value(row['adj_factor']),
                        escape_value(row['turnover_rate']),
                        escape_value(row['industry_code']),
                        escape_value(row['total_mv']),
                        escape_value(row['is_st'])
                    )
                    values.append(f"({','.join(v)})")
                
                sql = f"REPLACE INTO stock_daily (symbol, trade_date, `open`, `high`, `low`, `close`, `pre_close`, `change`, `pct_chg`, `volume`, `amount`, `adj_factor`, `turnover_rate`, `industry_code`, `total_mv`, `is_st`) VALUES {','.join(values)}"
                cursor.execute(sql)
                total += len(chunk)
            conn.commit()
    return total

def main():
    print("=" * 60)
    print("V194 数据修复 - 2025 年全市场数据补齐")
    print("=" * 60)
    
    trade_dates = get_trade_dates(2025)
    if not trade_dates:
        print("无法获取交易日历")
        return
    
    print(f"2025 年交易日：{len(trade_dates)}天")
    
    success, skip, fail = 0, 0, 0
    
    for i, td in enumerate(trade_dates):
        with engine.connect() as conn:
            cnt = conn.execute(text("SELECT COUNT(*) FROM stock_daily WHERE trade_date = :td"), {"td": td}).fetchone()[0]
        
        if cnt >= 5000:
            skip += 1
            continue
        
        df = fetch_data(td)
        if df.empty:
            print(f"[{i+1}/{len(trade_dates)}] {td}: Fetch failed")
            fail += 1
            continue
        
        inserted = upsert_batch(df)
        if inserted > 0:
            print(f"[{i+1}/{len(trade_dates)}] {td}: Success {inserted} rows")
            success += 1
        else:
            print(f"[{i+1}/{len(trade_dates)}] {td}: Write failed")
            fail += 1
        
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    print("=" * 60)
    print(f"完成：成功{success}, 跳过{skip}, 失败{fail}")
    print("=" * 60)
    
    print("\n验收结果:")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT COUNT(*) as total, COUNT(DISTINCT trade_date) as days, AVG(daily_cnt) as avg_daily
            FROM (SELECT trade_date, COUNT(symbol) as daily_cnt FROM stock_daily
            WHERE trade_date >= '2025-01-01' AND trade_date < '2026-01-01' GROUP BY trade_date) t
        """)).fetchone()
        print(f"  总记录数：{result[0]}")
        print(f"  交易日数：{result[1]}")
        print(f"  平均每日股票数：{result[2]:.0f}")

if __name__ == "__main__":
    main()