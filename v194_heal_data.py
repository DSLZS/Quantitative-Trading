"""V194 数据恢复 - 补齐 2025 年全市场数据"""
import os
import time
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv
import tushare as ts

load_dotenv()

DB_URL = os.getenv('DATABASE_URL')
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

engine = create_engine(DB_URL, poolclass=QueuePool, pool_pre_ping=True)

TARGET_MIN_SYMBOLS = 5000
CHUNK_SIZE = 500
SLEEP_BETWEEN_REQUESTS = 2

def get_trade_dates_from_db(year: int) -> list:
    """从数据库获取已有交易日期"""
    with engine.connect() as conn:
        query = text("""
            SELECT DISTINCT trade_date FROM stock_daily
            WHERE trade_date >= :start AND trade_date < :end
            ORDER BY trade_date
        """)
        result = conn.execute(query, {"start": f"{year}0101", "end": f"{year+1}0101"})
        return [row[0] for row in result.fetchall()]

def get_trade_dates_from_tushare(year: int) -> list:
    """从 Tushare 获取交易日历"""
    trade_dates = []
    
    # 使用 SZSE 获取
    df = pro.trade_cal(exchange='SZSE', start_date=f"{year}0101", end_date=f"{year}1231")
    if df is not None and not df.empty:
        # 注意：is_open 是整数 1，不是字符串 '1'
        trade_dates = df[df['is_open'] == 1]['cal_date'].tolist()
        print(f"从 SZSE 获取 {len(trade_dates)} 个交易日期")
    
    return trade_dates

def fetch_daily_data(trade_date: str) -> pd.DataFrame:
    """获取单日全市场数据"""
    try:
        df = pro.daily(trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception as e:
        if '429' in str(e) or '频率' in str(e) or '积分' in str(e):
            print(f"[LIMIT] Waiting 60s for API...")
            time.sleep(60)
            return fetch_daily_data(trade_date)
    return pd.DataFrame()

def fetch_daily_basic(trade_date: str) -> pd.DataFrame:
    """获取单日基本面数据"""
    try:
        df = pro.daily_basic(trade_date=trade_date)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return pd.DataFrame()

def upsert_to_database(df: pd.DataFrame, trade_date: str):
    """使用 UPSERT 逻辑写入数据库"""
    if df is None or df.empty:
        return 0
    
    try:
        df = df.copy()
        
        # 列映射
        column_mapping = {
            'ts_code': 'symbol',
            'vol': 'volume',
        }
        df = df.rename(columns=column_mapping)
        
        # 转换 trade_date 为 date 类型 (20251231 -> 2025-12-31)
        df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        
        # 目标列 (包含表的所有列)
        target_columns = [
            'symbol', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'volume', 'amount',
            'adj_factor', 'turnover_rate', 'industry_code', 'total_mv', 'is_st'
        ]
        
        # 确保列存在
        for col in target_columns:
            if col not in df.columns:
                if col in ['industry_code']:
                    df[col] = ''
                elif col in ['total_mv', 'is_st']:
                    df[col] = 0
                else:
                    df[col] = None
        
        df = df[target_columns]
        df = df.drop_duplicates(subset=['symbol', 'trade_date'])
        
        # 分块写入
        total_inserted = 0
        with engine.connect() as conn:
            for i in range(0, len(df), CHUNK_SIZE):
                chunk = df.iloc[i:i+CHUNK_SIZE]
                chunk.to_sql('stock_daily', conn, if_exists='append', index=False, method='multi')
                total_inserted += len(chunk)
        
        return total_inserted
        
    except Exception as e:
        print(f"Write error: {type(e).__name__}: {str(e)[:100]}")
        return 0

def merge_data(market_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """合并市场数据和基本面数据"""
    if market_df.empty and basic_df.empty:
        return pd.DataFrame()
    
    if market_df.empty:
        return basic_df
    
    if basic_df.empty:
        return market_df
    
    # 合并
    merged = pd.merge(market_df, basic_df, on=['ts_code', 'trade_date'], how='left')
    return merged

def main():
    """主函数"""
    print("=" * 60)
    print("V194 数据恢复 - 2025 年全市场数据补齐")
    print("=" * 60)
    
    year = 2025
    
    # 获取交易日历
    trade_dates = get_trade_dates_from_tushare(year)
    if not trade_dates:
        print("无法获取交易日历，退出")
        return
    
    print(f"2025 年交易日：{len(trade_dates)}天")
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    for i, trade_date in enumerate(trade_dates):
        # 检查是否需要补齐
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) FROM stock_daily WHERE trade_date = :td")
            result = conn.execute(query, {"td": trade_date})
            count = result.fetchone()[0]
        
        if count >= TARGET_MIN_SYMBOLS:
            skip_count += 1
            continue
        
        # 获取数据
        market_df = fetch_daily_data(trade_date)
        basic_df = fetch_daily_basic(trade_date)
        
        if market_df.empty:
            print(f"[{i+1}/{len(trade_dates)}] {trade_date}: Fetch failed")
            fail_count += 1
            continue
        
        # 合并数据
        merged = merge_data(market_df, basic_df)
        
        # 写入数据库
        inserted = upsert_to_database(merged, trade_date)
        
        if inserted > 0:
            print(f"[{i+1}/{len(trade_dates)}] {trade_date}: Success {inserted} rows")
            success_count += 1
        else:
            print(f"[{i+1}/{len(trade_dates)}] {trade_date}: Write failed")
            fail_count += 1
        
        # 限流
        time.sleep(SLEEP_BETWEEN_REQUESTS)
    
    print("=" * 60)
    print(f"完成：成功{success_count}, 跳过{skip_count}, 失败{fail_count}")
    print("=" * 60)
    
    # 验收
    print("\n验收结果:")
    with engine.connect() as conn:
        query = text("""
            SELECT 
                COUNT(*) as total,
                COUNT(DISTINCT trade_date) as days,
                AVG(daily_cnt) as avg_daily
            FROM (
                SELECT trade_date, COUNT(symbol) as daily_cnt
                FROM stock_daily
                WHERE trade_date >= '20250101' AND trade_date < '20260101'
                GROUP BY trade_date
            ) t
        """)
        result = conn.execute(query)
        row = result.fetchone()
        print(f"  总记录数：{row[0]}")
        print(f"  交易日数：{row[1]}")
        print(f"  平均每日股票数：{row[2]:.0f}")

if __name__ == "__main__":
    main()