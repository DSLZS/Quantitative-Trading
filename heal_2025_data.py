"""
V194 数据补全脚本 - 暴力补齐 2025 年全市场数据

【核心职责】
1. 从 Tushare API 拉取全市场股票每日数据（不限制 symbol）
2. 使用 wait_and_retry 机制处理频率限制
3. 直接写入 stock_daily 表，使用 chunksize=500 防止内存溢出
4. 打印验收矩阵

【验收标准】
- 2025 年 1 月至今，每日平均 symbols 数量必须 > 5000
- 缺失日期必须全部补齐
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from loguru import logger
from dotenv import load_dotenv
import tushare as ts

load_dotenv()

# 配置
DB_URL = os.getenv('DATABASE_URL')
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
TARGET_MIN_SYMBOLS = 5000  # 每日最少股票数
CHUNK_SIZE = 500  # 数据库写入块大小，防止内存溢出

# Tushare 配置
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 数据库引擎
engine = create_engine(
    DB_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
)

# 频率限制控制
REQUEST_COUNT = 0
LAST_REQUEST_TIME = time.time()
MAX_REQUESTS_PER_MINUTE = 60


def wait_and_retry(max_retries: int = 3, base_sleep: float = 1.0):
    """等待并重试装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global REQUEST_COUNT, LAST_REQUEST_TIME
            
            for attempt in range(max_retries):
                try:
                    # 频率控制
                    current_time = time.time()
                    time_since_last = current_time - LAST_REQUEST_TIME
                    if time_since_last < (60.0 / MAX_REQUESTS_PER_MINUTE):
                        sleep_time = (60.0 / MAX_REQUESTS_PER_MINUTE) - time_since_last
                        time.sleep(sleep_time)
                    
                    result = func(*args, **kwargs)
                    REQUEST_COUNT += 1
                    LAST_REQUEST_TIME = time.time()
                    return result
                    
                except Exception as e:
                    error_msg = str(e)
                    if '积分不足' in error_msg or '频率限制' in error_msg:
                        sleep_time = base_sleep * (2 ** attempt)
                        logger.warning(f"Tushare 频率限制，等待 {sleep_time:.1f}秒后重试...")
                        time.sleep(sleep_time)
                    elif '400' in error_msg or '500' in error_msg:
                        logger.error(f"API 错误：{error_msg}")
                        return None
                    else:
                        logger.error(f"未知错误：{error_msg}")
                        return None
            
            logger.error(f"达到最大重试次数 {max_retries}")
            return None
        return wrapper
    return decorator


def get_all_trade_dates(start_date: str, end_date: str) -> list:
    """
    获取所有交易日期
    
    优先从数据库已有数据中提取交易日期，避免依赖 Tushare API
    """
    try:
        # 从数据库获取已有交易日期
        with engine.connect() as conn:
            query = text("""
                SELECT DISTINCT trade_date 
                FROM stock_daily 
                WHERE trade_date >= :start_date AND trade_date <= :end_date
                ORDER BY trade_date
            """)
            result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date
            })
            trade_dates = [row[0] for row in result.fetchall()]
            
        if trade_dates:
            logger.info(f"从数据库获取 {len(trade_dates)} 个交易日期")
            return trade_dates
        else:
            logger.warning("数据库中没有交易日期，尝试从 Tushare 获取...")
            
    except Exception as e:
        logger.warning(f"从数据库获取交易日期失败：{e}，尝试从 Tushare 获取...")
    
    # 备用方案：从 Tushare 获取
    try:
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            trade_dates = df[df['is_open'] == '1']['cal_date'].tolist()
            return trade_dates
    except Exception as e:
        logger.error(f"从 Tushare 获取交易日历失败：{e}")
    
    return []


def get_stock_list(date: str) -> list:
    """获取指定日期的股票列表"""
    try:
        # 获取股票列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
        if df is not None and not df.empty:
            return df['ts_code'].tolist()
        return []
    except Exception as e:
        logger.error(f"获取股票列表失败：{e}")
        return []


@wait_and_retry(max_retries=3, base_sleep=2.0)
def fetch_daily_data(trade_date: str) -> pd.DataFrame:
    """
    拉取指定日期的全市场数据
    
    关键：不传 ts_code 参数，直接按日期全量拉取
    """
    try:
        # 使用 daily 接口，不传 ts_code，按日期全量拉取
        df = pro.daily(trade_date=trade_date)
        if df is not None and not df.empty:
            logger.info(f"拉取 {trade_date} 数据：{len(df)}条")
            return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"拉取 {trade_date} 数据失败：{e}")
        return pd.DataFrame()


def check_existing_data(trade_date: str) -> int:
    """检查数据库中已存在的该日期数据量"""
    with engine.connect() as conn:
        query = text("""
            SELECT COUNT(*) as cnt FROM stock_daily 
            WHERE trade_date = :trade_date
        """)
        result = conn.execute(query, {"trade_date": trade_date})
        row = result.fetchone()
        return row[0] if row else 0


def save_to_database(df: pd.DataFrame, chunksize: int = CHUNK_SIZE):
    """
    保存数据到数据库
    
    关键修复：使用 chunksize 参数分批写入，防止内存溢出
    """
    if df is None or df.empty:
        return
    
    try:
        # 数据预处理
        df = df.copy()
        
        # 重命名列
        column_mapping = {
            'ts_code': 'symbol',
            'cal_date': 'trade_date',
        }
        df = df.rename(columns=column_mapping)
        
        # 确保 trade_date 格式正确
        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)
        
        # 选择需要的列
        target_columns = [
            'symbol', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'vol', 'amount'
        ]
        
        # 确保所有列存在
        for col in target_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[target_columns]
        
        # 删除重复
        df = df.drop_duplicates(subset=['symbol', 'trade_date'])
        
        # 保存到数据库 - 关键修复：使用 chunksize 分批写入
        with engine.connect() as conn:
            df.to_sql('stock_daily', conn, if_exists='append', index=False, 
                     method='multi', chunksize=chunksize)
        
        logger.info(f"保存 {len(df)} 条记录到数据库 (chunksize={chunksize})")
        
    except Exception as e:
        logger.error(f"保存数据失败：{e}")


def heal_missing_dates(trade_dates: list, target_year: int = 2025):
    """补齐缺失日期的数据"""
    logger.info("=" * 70)
    logger.info(f"开始补齐 {target_year} 年数据...")
    logger.info("=" * 70)
    
    healed_dates = []
    failed_dates = []
    
    for i, trade_date in enumerate(trade_dates):
        # 将日期转换为字符串（处理 datetime.date 对象）
        if hasattr(trade_date, 'strftime'):
            trade_date_str = trade_date.strftime('%Y%m%d')
        else:
            trade_date_str = str(trade_date)
        
        # 只处理目标年份的数据
        if int(trade_date_str[:4]) != target_year and target_year == 2025:
            # 但也要处理 2026 年部分数据（如果有）
            if int(trade_date_str[:4]) > 2026:
                continue
        
        # 更新 trade_date 为字符串格式
        trade_date = trade_date_str
        
        # 检查已存在的数据量
        existing_count = check_existing_data(trade_date)
        
        if existing_count >= TARGET_MIN_SYMBOLS:
            logger.info(f"[{i+1}/{len(trade_dates)}] {trade_date}: 已存在 {existing_count}条，跳过")
            continue
        
        logger.info(f"[{i+1}/{len(trade_dates)}] {trade_date}: 已存在 {existing_count}条，需要补齐")
        
        # 拉取数据
        df = fetch_daily_data(trade_date)
        
        if df is not None and not df.empty:
            # 保存数据 - 使用 chunksize 防止内存溢出
            save_to_database(df, chunksize=CHUNK_SIZE)
            healed_dates.append(trade_date)
            
            # 每 10 次请求休息一下
            if (i + 1) % 10 == 0:
                logger.info("已处理 10 个日期，休息 5 秒...")
                time.sleep(5)
        else:
            logger.warning(f"{trade_date}: 拉取失败")
            failed_dates.append(trade_date)
    
    logger.info("=" * 70)
    logger.info(f"补齐完成！成功：{len(healed_dates)}, 失败：{len(failed_dates)}")
    if failed_dates:
        logger.warning(f"失败日期：{failed_dates}")
    logger.info("=" * 70)
    
    return healed_dates, failed_dates


def print_acceptance_matrix(target_year: int = 2025):
    """打印验收矩阵"""
    logger.info("=" * 70)
    logger.info("验收矩阵")
    logger.info("=" * 70)
    
    with engine.connect() as conn:
        # 查询 2025 年数据 - 使用 symbol 列名
        query = text("""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT trade_date) as trade_days,
                AVG(daily_count) as avg_daily_symbols,
                MIN(daily_count) as min_daily_symbols,
                MAX(daily_count) as max_daily_symbols
            FROM (
                SELECT trade_date, COUNT(symbol) as daily_count
                FROM stock_daily
                WHERE trade_date >= :start_date AND trade_date < :end_date
                GROUP BY trade_date
            ) t
        """)
        
        # 2025 年
        result = conn.execute(query, {
            "start_date": f"{target_year}0101",
            "end_date": f"{target_year + 1}0101"
        })
        row = result.fetchone()
        
        if row:
            total_rows = row[0] or 0
            trade_days = row[1] or 0
            avg_daily = row[2] or 0
            min_daily = row[3] or 0
            max_daily = row[4] or 0
            
            logger.info(f"{target_year}年已补齐天数：{trade_days}天")
            logger.info(f"{target_year}年平均每日股票数：{avg_daily:.0f}只 (必须 > {TARGET_MIN_SYMBOLS})")
            
            # 检查缺失日期
            query_missing = text("""
                SELECT trade_date, COUNT(symbol) as cnt
                FROM stock_daily
                WHERE trade_date >= :start_date AND trade_date < :end_date
                GROUP BY trade_date
                HAVING COUNT(symbol) < :threshold
                ORDER BY trade_date
            """)
            missing_result = conn.execute(query_missing, {
                "start_date": f"{target_year}0101",
                "end_date": f"{target_year + 1}0101",
                "threshold": TARGET_MIN_SYMBOLS
            })
            missing_dates = [(row[0], row[1]) for row in missing_result]
            
            if missing_dates:
                logger.warning(f"缺失日期自查结果：{len(missing_dates)}个日期不达标")
                for date, count in missing_dates[:10]:  # 只显示前 10 个
                    logger.warning(f"  {date}: {count}只")
                if len(missing_dates) > 10:
                    logger.warning(f"  ... 还有{len(missing_dates) - 10}个")
            else:
                logger.info("缺失日期自查结果：无")
            
            # 状态判定
            if avg_daily >= TARGET_MIN_SYMBOLS:
                logger.info("✅ 验收通过！")
            else:
                logger.error("❌ 验收失败！平均每日股票数不足")
        
        logger.info("=" * 70)


def main():
    """主函数"""
    logger.info("=" * 70)
    logger.info("V194 数据补全脚本 - 2025 年全市场数据暴力补齐")
    logger.info("=" * 70)
    
    # 获取交易日期
    logger.info("获取交易日期...")
    trade_dates = get_all_trade_dates('20250101', '20251231')
    logger.info(f"共 {len(trade_dates)} 个交易日")
    
    if not trade_dates:
        logger.error("无法获取交易日期，退出")
        return
    
    # 补齐数据
    healed_dates, failed_dates = heal_missing_dates(trade_dates, target_year=2025)
    
    # 打印验收矩阵
    print_acceptance_matrix(target_year=2025)
    
    # 最终统计
    logger.info("=" * 70)
    logger.info("最终统计")
    logger.info("=" * 70)
    logger.info(f"成功补齐：{len(healed_dates)}天")
    logger.info(f"失败：{len(failed_dates)}天")


if __name__ == "__main__":
    main()