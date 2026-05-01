"""Quick database schema check"""
import sys
from sqlalchemy import create_engine, text

engine = create_engine('mysql+pymysql://root:123456@localhost:3306/quantitative_trading')
with engine.connect() as conn:
    print("=== TABLES ===")
    result = conn.execute(text('SHOW TABLES'))
    for row in result:
        print(row[0])
    
    print("\n=== stock_daily columns ===")
    result = conn.execute(text('DESCRIBE stock_daily'))
    for row in result:
        print(f"  {row[0]}: {row[1]}")
    
    print("\n=== index_daily columns ===")
    result = conn.execute(text('DESCRIBE index_daily'))
    for row in result:
        print(f"  {row[0]}: {row[1]}")
    
    print("\n=== index_daily sample ===")
    result = conn.execute(text("SELECT * FROM index_daily WHERE symbol='000300.SH' ORDER BY trade_date DESC LIMIT 5"))
    for row in result:
        print(row)
    
    print("\n=== index_constituents exists? ===")
    result = conn.execute(text("SHOW TABLES LIKE 'index_constituents'"))
    rows = list(result)
    if rows:
        print("  YES - table exists")
        result = conn.execute(text('DESCRIBE index_constituents'))
        for row in result:
            print(f"  {row[0]}: {row[1]}")
    else:
        print("  NO - table does not exist")

print("\nDone.")
sys.stdout.flush()