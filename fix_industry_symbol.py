"""
修复 stock_industry_daily 表中的 symbol 格式问题

问题：表中混用了 6 位代码 (000001) 和 9 位代码 (000001.SZ)
解决：
1. 先删除 6 位代码的重复记录（因为已有 9 位代码记录）
2. 将剩余的 6 位代码转换为 9 位代码格式
"""

from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from loguru import logger

load_dotenv()

db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

logger.info("开始修复 stock_industry_daily 表 symbol 格式...")

# 步骤 1: 检查当前数据状态
query = text("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN LENGTH(symbol) = 6 THEN 1 END) as len6,
        COUNT(CASE WHEN LENGTH(symbol) = 9 THEN 1 END) as len9
    FROM stock_industry_daily
""")
with engine.connect() as conn:
    result = conn.execute(query).fetchone()
    logger.info(f"修复前：total={result[0]}, 6 位={result[1]}, 9 位={result[2]}")

# 步骤 2: 删除 6 位代码记录（因为对应的 9 位代码记录已存在）
# 原理：对于同一 trade_date，如果既有 000001 又有 000001.SZ，删除 000001
logger.info("步骤 1: 删除与 9 位代码重复的 6 位代码记录...")

delete_query = text("""
    DELETE sid6 FROM stock_industry_daily sid6
    INNER JOIN stock_industry_daily sid9
    ON LEFT(sid9.symbol, 6) = sid6.symbol
    AND sid9.trade_date = sid6.trade_date
    WHERE LENGTH(sid6.symbol) = 6 AND LENGTH(sid9.symbol) = 9
""")

with engine.connect() as conn:
    result = conn.execute(delete_query)
    conn.commit()
    logger.info(f"已删除 {result.rowcount} 条重复的 6 位代码记录")

# 步骤 3: 验证删除结果
query = text("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN LENGTH(symbol) = 6 THEN 1 END) as len6,
        COUNT(CASE WHEN LENGTH(symbol) = 9 THEN 1 END) as len9
    FROM stock_industry_daily
""")
with engine.connect() as conn:
    result = conn.execute(query).fetchone()
    logger.info(f"删除后：total={result[0]}, 6 位={result[1]}, 9 位={result[2]}")

# 步骤 4: 将剩余的 6 位代码转换为 9 位代码
logger.info("步骤 2: 将剩余 6 位代码转换为 9 位代码...")

simple_update = text("""
    UPDATE stock_industry_daily
    SET symbol = CONCAT(
        LEFT(symbol, 6),
        '.',
        CASE
            WHEN LEFT(symbol, 1) IN ('0', '3') THEN 'SZ'
            WHEN LEFT(symbol, 1) IN ('6', '8') THEN 'SH'
            WHEN LEFT(symbol, 3) = '688' THEN 'SH'
            WHEN LEFT(symbol, 1) IN ('4', '5', '7', '9') THEN 'SH'
            ELSE 'SH'
        END
    )
    WHERE LENGTH(symbol) = 6
""")

with engine.connect() as conn:
    result = conn.execute(simple_update)
    conn.commit()
    logger.info(f"已更新 {result.rowcount} 条记录为 9 位代码")

# 步骤 5: 验证修复结果
query = text("""
    SELECT 
        COUNT(*) as total,
        COUNT(CASE WHEN LENGTH(symbol) = 6 THEN 1 END) as len6,
        COUNT(CASE WHEN LENGTH(symbol) = 9 THEN 1 END) as len9,
        COUNT(DISTINCT symbol) as unique_symbols
    FROM stock_industry_daily
""")
with engine.connect() as conn:
    result = conn.execute(query).fetchone()
    logger.info(f"修复后：total={result[0]}, 6 位={result[1]}, 9 位={result[2]}, 唯一 symbol={result[3]}")

# 检查各年份数据
for year in [2020, 2022, 2024]:
    query = text(f"""
        SELECT COUNT(*) as cnt, COUNT(DISTINCT symbol) as symbols
        FROM stock_industry_daily 
        WHERE YEAR(trade_date) = {year}
    """)
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        logger.info(f"{year}年：records={result[0]}, symbols={result[1]}")

if result[1] == 0:
    logger.info("✓ 所有 6 位代码已转换为 9 位代码格式")
else:
    logger.warning(f"仍有 {result[1]} 条 6 位代码记录")

logger.info("修复完成！")