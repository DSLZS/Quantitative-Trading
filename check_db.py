from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")
if not db_url:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_engine(db_url)

with engine.connect() as conn:
    result = conn.execute(text('DESCRIBE stock_daily'))
    columns = [row[0] for row in result]
    print("Columns in stock_daily table:")
    for col in columns:
        print(f"  - {col}")