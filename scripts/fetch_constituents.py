"""
Fetch CSI 300 index constituents using tushare and store in database
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
import pandas as pd
import tushare as ts
from dotenv import load_dotenv

load_dotenv()

TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
DB_URL = os.getenv('DATABASE_URL')

def main():
    print("[Step 1] Initializing tushare...")
    sys.stdout.flush()
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    print("[Step 2] Creating index_constituents table...")
    sys.stdout.flush()
    engine = create_engine(DB_URL)
    
    with engine.connect() as conn:
        # First clear existing data
        conn.execute(text("DELETE FROM index_constituents WHERE index_code = '000300.SH'"))
        conn.commit()
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS index_constituents (
                index_code VARCHAR(20) NOT NULL,
                index_name VARCHAR(50) DEFAULT NULL,
                con_code VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                weight DECIMAL(10,4) DEFAULT NULL,
                PRIMARY KEY (index_code, con_code, trade_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """))
        conn.commit()
    print("  Table created successfully")
    sys.stdout.flush()
    
    # Define index code for CSI 300
    INDEX_CODE = '000300.SH'
    INDEX_NAME = '沪深300'
    
    # Fetch constituents for each year-end and mid-year
    dates = []
    for year in range(2019, 2027):
        dates.append(f"{year}0101")
        dates.append(f"{year}0630")
        dates.append(f"{year}1231")
    
    print(f"[Step 3] Fetching constituents for dates: {dates}")
    sys.stdout.flush()
    
    all_records = []
    
    for date in dates:
        try:
            print(f"  Fetching for {date}...")
            sys.stdout.flush()
            df = pro.index_weight(index_code=INDEX_CODE, start_date=date, end_date=date)
            
            if df is not None and not df.empty:
                df['index_code'] = INDEX_CODE
                df['index_name'] = INDEX_NAME
                
                for _, row in df.iterrows():
                    record = (
                        INDEX_CODE,
                        INDEX_NAME,
                        row.get('con_code', ''),
                        row.get('trade_date', ''),
                        float(row.get('weight', 0)) if pd.notna(row.get('weight')) else 0.0
                    )
                    all_records.append(record)
                print(f"    Got {len(df)} constituents")
            else:
                print(f"    No data for {date}")
            
            sys.stdout.flush()
            time.sleep(0.5)  # rate limit
            
        except Exception as e:
            print(f"    Error fetching {date}: {e}")
            sys.stdout.flush()
    
    if all_records:
        print(f"\n[Step 4] Inserting {len(all_records)} records into database...")
        sys.stdout.flush()
        
        # Convert tuples to dict for SQLAlchemy 2.0
        record_dicts = [
            {
                'index_code': r[0],
                'index_name': r[1],
                'con_code': r[2],
                'trade_date': r[3],
                'weight': r[4]
            }
            for r in all_records
        ]
        
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT IGNORE INTO index_constituents 
                (index_code, index_name, con_code, trade_date, weight)
                VALUES (:index_code, :index_name, :con_code, :trade_date, :weight)
            """), record_dicts)
            conn.commit()
        
        # Verify
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM index_constituents"))
            count = result.scalar()
            print(f"  Total records in table: {count}")
            
            result = conn.execute(text("SELECT DISTINCT trade_date FROM index_constituents ORDER BY trade_date"))
            dates_in_db = [str(r[0]) for r in result]
            print(f"  Dates in DB: {dates_in_db}")
    else:
        print("\n  No records to insert")
        print("  Trying index constituents without weight...")
        sys.stdout.flush()
        
        # Fallback: use index constituents list (without weights)
        try:
            df = pro.index_member(index_code=INDEX_CODE)
            if df is not None and not df.empty:
                print(f"  Got {len(df)} constituents from index_member")
                fallback_records = []
                for _, row in df.iterrows():
                    record = {
                        'index_code': INDEX_CODE,
                        'index_name': INDEX_NAME,
                        'con_code': row.get('con_code', ''),
                        'trade_date': pd.Timestamp.today().date(),
                        'weight': 0.0
                    }
                    fallback_records.append(record)
                
                with engine.connect() as conn:
                    conn.execute(text("""
                        INSERT IGNORE INTO index_constituents 
                        (index_code, index_name, con_code, trade_date, weight)
                        VALUES (:index_code, :index_name, :con_code, :trade_date, :weight)
                    """), fallback_records)
                    conn.commit()
        except Exception as e:
            print(f"  Error with fallback: {e}")
    
    print("\nDone.")
    sys.stdout.flush()

if __name__ == '__main__':
    main()