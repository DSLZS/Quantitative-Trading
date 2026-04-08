# V178: DATA FIX MODE - STRICT EXECUTION
"""
V178 数据补全模块 - 2023 年数据硬通关
=====================================

【Mission】
停止所有算法逻辑！只修复 2023 年数据！

【强制执行逻辑】
1. 按交易日循环 - 严禁按股票循环！严禁按月循环！
2. 获取 2023-01-01 到 2023-12-31 的所有交易日列表
3. 遍历每一个 trade_date，调用 pro.daily(trade_date=date)
4. 每天只需 1 次 API 调用，全年 242 次，严禁触发 Tushare 频控！

【实时反馈】
- 每一天数据写入 MySQL 后，必须立即打印进度
- 严禁连续 10 秒钟不打印任何日志！

【数据库事务强制提交】
- 使用 engine.connect() 开启事务
- 每写入一天数据必须执行一次 connection.commit()
- 严禁所有数据塞进内存最后一次性写入！

【反欺诈条款】
- 禁止修改初始资金：始终锁定 100,000
- 禁止美化结果：如果数据拉不下来，严禁使用 2024 年的平均值填充 2023 年！
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# 配置常量
YEAR_TARGET = 2023
MIN_ROWS_2023 = 500000
TUSHARE_TIMEOUT = 30
TUSHARE_MAX_RETRIES = 5
RETRY_SLEEP = 5
PROGRESS_INTERVAL = 1  # 每天都打印进度
CHECKPOINT_FILE = "data/sync_status/v178_data_fix_checkpoint.json"


class V178DataFixer:
    """
    V178 数据修复器 - 按交易日暴力拉取 2023 年数据
    """
    
    def __init__(self, db_url: Optional[str] = None, tushare_token: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.engine = None
        self.ts_pro = None
        self.checkpoint = {"completed_dates": [], "failed_dates": [], "total_inserted": 0}
        
        self._init_database()
        self._init_tushare()
        self._load_checkpoint()
    
    def _init_database(self):
        """初始化数据库连接"""
        if not self.db_url:
            logger.error("[V178] DATABASE_URL not configured!")
            return
        
        try:
            from sqlalchemy import create_engine
            self.engine = create_engine(
                self.db_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            logger.info("[V178] Database connection initialized")
        except Exception as e:
            logger.error(f"[V178] Failed to init database: {e}")
    
    def _init_tushare(self):
        """初始化 Tushare API"""
        if not self.tushare_token:
            logger.error("[V178] TUSHARE_TOKEN not configured!")
            return
        
        try:
            import tushare as ts
            ts.set_token(self.tushare_token)
            self.ts_pro = ts.pro_api()
            logger.info("[V178] Tushare API initialized")
        except Exception as e:
            logger.error(f"[V178] Failed to init Tushare: {e}")
    
    def _load_checkpoint(self):
        """加载断点文件"""
        try:
            checkpoint_path = Path(CHECKPOINT_FILE)
            if checkpoint_path.exists():
                with open(checkpoint_path, 'r') as f:
                    self.checkpoint = json.load(f)
                logger.info(f"[V178] Loaded checkpoint: {len(self.checkpoint.get('completed_dates', []))} dates completed")
        except Exception as e:
            logger.warning(f"[V178] Failed to load checkpoint: {e}")
            self.checkpoint = {"completed_dates": [], "failed_dates": [], "total_inserted": 0}
    
    def _save_checkpoint(self):
        """保存断点文件"""
        try:
            checkpoint_path = Path(CHECKPOINT_FILE)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                json.dump(self.checkpoint, f, indent=2)
            logger.info(f"[V178] Checkpoint saved")
        except Exception as e:
            logger.error(f"[V178] Failed to save checkpoint: {e}")
    
    def _get_trading_dates(self, year: int) -> List[str]:
        """获取指定年份的所有交易日"""
        if not self.ts_pro:
            logger.error("[V178] Tushare API not available!")
            return []
        
        try:
            start_date = f"{year}0101"
            end_date = f"{year}1231"
            cal_df = self.ts_pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
            
            if cal_df is not None and not cal_df.empty:
                dates = sorted(cal_df['cal_date'].tolist())
                logger.info(f"[V178] Found {len(dates)} trading days in {year}")
                return dates
            return []
        except Exception as e:
            logger.error(f"[V178] Failed to fetch trading dates: {e}")
            return []
    
    def _get_existing_dates(self, year: int) -> set:
        """获取数据库中已存在的日期"""
        if not self.engine:
            return set()
        
        try:
            from sqlalchemy import text
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            query = text("""
                SELECT DISTINCT trade_date FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
                existing = {row[0] for row in result.fetchall()}
            
            logger.info(f"[V178] Found {len(existing)} existing dates in database for {year}")
            return existing
        except Exception as e:
            logger.error(f"[V178] Failed to fetch existing dates: {e}")
            return set()
    
    def _fetch_daily_data(self, trade_date: str) -> Optional[Dict]:
        """
        获取某一天的全部股票数据
        使用 pro.daily(trade_date=date) 方式，每天只需 1 次 API 调用
        """
        if not self.ts_pro:
            return None
        
        for attempt in range(TUSHARE_MAX_RETRIES):
            try:
                # 关键：使用 trade_date 参数一次性获取全市场数据
                df = self.ts_pro.daily(trade_date=trade_date)
                
                if df is None or df.empty:
                    logger.warning(f"[V178] No data returned for {trade_date}")
                    return None
                
                # 转换为字典格式便于处理
                data = {
                    'trade_date': trade_date,
                    'records': []
                }
                
                for _, row in df.iterrows():
                    record = {
                        'symbol': row.get('ts_code', ''),
                        'trade_date': trade_date,
                        'open': float(row['open']) if 'open' in row and pd.notna(row['open']) else None,
                        'high': float(row['high']) if 'high' in row and pd.notna(row['high']) else None,
                        'low': float(row['low']) if 'low' in row and pd.notna(row['low']) else None,
                        'close': float(row['close']) if 'close' in row and pd.notna(row['close']) else None,
                        'vol': float(row['vol']) if 'vol' in row and pd.notna(row['vol']) else None,
                        'amount': float(row['amount']) if 'amount' in row and pd.notna(row['amount']) else None,
                        'adj_factor': float(row['adj_factor']) if 'adj_factor' in row and pd.notna(row['adj_factor']) else None,
                    }
                    data['records'].append(record)
                
                logger.info(f"[V178] Fetched {len(data['records'])} records for {trade_date}")
                return data
                
            except Exception as e:
                error_str = str(e).lower()
                if '403' in error_str or '抱歉' in str(e) or 'limit' in error_str or 'flow' in error_str:
                    logger.warning(f"[V178] Rate limit hit for {trade_date}, sleeping {RETRY_SLEEP}s... (attempt {attempt+1}/{TUSHARE_MAX_RETRIES})")
                    time.sleep(RETRY_SLEEP)
                else:
                    logger.error(f"[V178] Failed to fetch {trade_date}: {e}")
                    if attempt < TUSHARE_MAX_RETRIES - 1:
                        time.sleep(RETRY_SLEEP)
        
        return None
    
    def _insert_day_data(self, trade_date: str, records: List[Dict]) -> int:
        """
        插入某一天的数据到数据库
        使用事务强制提交
        """
        if not self.engine or not records:
            return 0
        
        try:
            from sqlalchemy import text
            
            # 使用事务
            with self.engine.connect() as conn:
                # 开始事务
                trans = conn.begin()
                
                try:
                    # 先检查是否已存在
                    check_query = text("""
                        SELECT COUNT(*) FROM stock_daily
                        WHERE trade_date = :trade_date
                    """)
                    result = conn.execute(check_query, {"trade_date": trade_date})
                    existing_count = result.scalar()
                    
                    if existing_count > 0:
                        logger.info(f"[V178] {trade_date}: Already has {existing_count} records, skipping")
                        trans.rollback()
                        return 0
                    
                    # 插入数据
                    insert_query = text("""
                        INSERT INTO stock_daily 
                        (symbol, trade_date, open, high, low, close, vol, amount, adj_factor)
                        VALUES 
                        (:symbol, :trade_date, :open, :high, :low, :close, :vol, :amount, :adj_factor)
                    """)
                    
                    inserted = 0
                    for record in records:
                        conn.execute(insert_query, record)
                        inserted += 1
                    
                    # 强制提交事务
                    trans.commit()
                    
                    logger.info(f"[V178] {trade_date}: Inserted {inserted} records, COMMITTED")
                    return inserted
                    
                except Exception as e:
                    logger.error(f"[V178] {trade_date}: Transaction failed: {e}")
                    trans.rollback()
                    return 0
                    
        except Exception as e:
            logger.error(f"[V178] {trade_date}: Failed to insert: {e}")
            return 0
    
    def _verify_data_count(self, year: int) -> int:
        """验证数据库中指定年份的数据行数"""
        if not self.engine:
            return 0
        
        try:
            from sqlalchemy import text
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"
            
            query = text("""
                SELECT COUNT(*) FROM stock_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"start_date": start_date, "end_date": end_date})
                count = result.scalar()
            
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"[V178] Failed to verify data count: {e}")
            return 0
    
    def fix_year(self, year: int) -> bool:
        """
        修复指定年份的数据
        按交易日循环，每天 1 次 API 调用
        """
        logger.info("=" * 80)
        logger.info(f"[V178] Starting DATA FIX for year {year}")
        logger.info(f"[V178] Strategy: By TRADING DAY (1 API call per day)")
        logger.info(f"[V178] Target: > {MIN_ROWS_2023} rows for 2023")
        logger.info("=" * 80)
        
        if not self.ts_pro:
            logger.error("[V178] Cannot fix without Tushare API!")
            return False
        
        if not self.engine:
            logger.error("[V178] Cannot fix without database connection!")
            return False
        
        # 获取交易日列表
        trading_dates = self._get_trading_dates(year)
        if not trading_dates:
            logger.error(f"[V178] No trading dates found for {year}")
            return False
        
        # 获取已存在的日期
        existing_dates = self._get_existing_dates(year)
        
        # 过滤需要处理的日期
        dates_to_process = [d for d in trading_dates if d not in existing_dates]
        
        logger.info(f"[V178] Total trading days: {len(trading_dates)}")
        logger.info(f"[V178] Existing dates: {len(existing_dates)}")
        logger.info(f"[V178] Dates to process: {len(dates_to_process)}")
        
        if not dates_to_process:
            logger.info(f"[V178] All dates already processed!")
            final_count = self._verify_data_count(year)
            logger.info(f"[V178] Final count: {final_count}")
            return final_count >= MIN_ROWS_2023
        
        # 按交易日循环处理
        total_inserted = 0
        processed_count = 0
        
        for i, trade_date in enumerate(dates_to_process):
            processed_count += 1
            
            # 获取当天数据
            logger.info(f"[V178] Fetching {trade_date} ({processed_count}/{len(dates_to_process)})...")
            data = self._fetch_daily_data(trade_date)
            
            if data is None or not data.get('records'):
                logger.warning(f"[V178] {trade_date}: No data to insert")
                self.checkpoint['failed_dates'].append(trade_date)
                self._save_checkpoint()
                continue
            
            # 插入数据
            inserted = self._insert_day_data(trade_date, data['records'])
            total_inserted += inserted
            
            # 更新断点
            self.checkpoint['completed_dates'].append(trade_date)
            self.checkpoint['total_inserted'] = total_inserted
            self._save_checkpoint()
            
            # 实时反馈 - 每天都打印
            current_total = self._verify_data_count(year)
            logger.info(
                f"[PROGRESS] {trade_date} | "
                f"New: {inserted} rows | "
                f"DB Total: {current_total} | "
                f"Memory: N/A"
            )
            
            # 小睡避免频控
            if processed_count % 10 == 0:
                time.sleep(1)
        
        # 最终验证
        logger.info("=" * 80)
        final_count = self._verify_data_count(year)
        logger.info(f"[V178] DATA FIX COMPLETE")
        logger.info(f"[V178] Total inserted: {total_inserted}")
        logger.info(f"[V178] Final DB count: {final_count}")
        logger.info(f"[V178] Target: {MIN_ROWS_2023}")
        
        if final_count < MIN_ROWS_2023:
            logger.error(f"[V178] FAILED: Count {final_count} < {MIN_ROWS_2023}")
            raise RuntimeError(f"Data count verification failed: {final_count} < {MIN_ROWS_2023}")
        else:
            logger.info(f"[V178] SUCCESS: Count {final_count} >= {MIN_ROWS_2023}")
        
        logger.info("=" * 80)
        return True


# 需要导入 pandas
import pandas as pd


if __name__ == "__main__":
    logger.info("[V178] Testing DataFixer...")
    fixer = V178DataFixer()
    if fixer.ts_pro and fixer.engine:
        fixer.fix_year(2023)
    else:
        logger.error("[V178] Cannot test - missing configuration")