#!/usr/bin/env python
"""清理 2023 年的旧数据（volume 字段为 NULL 的数据）"""

from dotenv import load_dotenv
load_dotenv()

import os
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv("DATABASE_URL"))

print("正在清理 2023 年的旧数据...")

with engine.connect() as conn:
    trans = conn.begin()
    try:
        result = conn.execute(text("""
            DELETE FROM stock_daily 
            WHERE trade_date BETWEEN '2023-01-01' AND '2023-12-31'
        """))
        print(f"已删除 {result.rowcount} 行 2023 年数据")
        trans.commit()
        print("清理完成！")
    except Exception as e:
        print(f"清理失败：{e}")
        trans.rollback()