"""
V194 数据补完 - 2025 最终决战

【核心职责】
1. 补全 stock_daily 表中 2025-01-01 至今的所有 A 股日线数据
2. 实现：2025 年交易日平均每日记录数 > 5000 条
3. 使用 ON DUPLICATE KEY UPDATE (UPSERT) 防止主键冲突
4. 严格的 Tushare 频率控制

【数据库字段映射】
- ts_code (Tushare) -> symbol (MySQL)
- trade_date: YYYYMMDD -> YYYY-MM-DD
- vol (Tushare) -> volume (MySQL)
- 其他字段：open, high, low, close, amount, adj_factor, turnover_rate, pre_close, change, pct_chg
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
import re

load_dotenv()

# 配置
DB_URL = os.getenv('DATABASE_URL')
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
TARGET_MIN_SYMBOLS = 5000  # 每日最少股票数
CHUNK_SIZE = 500  # 数据库写入块大小
SLEEP_SECONDS = 15  # API 请求间隔（保守策略）
MAX_RETRIES = 3  # 最大重试次数

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


def get_existing_trade_dates(start_date: str, end_date: str) -> set:
    """获取数据库中已存在的交易日期及其数据量"""
    try:
        with engine.connect() as conn:
            query = text("""
                SELECT trade_date, COUNT(*) as cnt 
                FROM stock_daily 
                WHERE trade_date >= :start_date AND trade_date <= :end_date
                GROUP BY trade_date
                ORDER BY trade_date
            """)
            result = conn.execute(query, {
                "start_date": start_date,
                "end_date": end_date
            })
            # 返回已有数据且数量>=5000 的日期集合
            existing_dates = {row[0] for row in result.fetchall() if row[1] >= TARGET_MIN_SYMBOLS}
            return existing_dates
    except Exception as e:
        logger.error(f"获取已存在交易日期失败：{e}")
        return set()


def get_all_trade_dates_from_tushare(start_date: str, end_date: str) -> list:
    """从 Tushare 获取所有交易日期"""
    try:
        df = pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date)
        if df is not None and not df.empty:
            # 修复：is_open 是整数类型，不是字符串
            trade_dates = df[df['is_open'] == 1]['cal_date'].tolist()
            logger.info(f"从 Tushare 获取到 {len(trade_dates)} 个交易日")
            return trade_dates
        else:
            logger.warning("Tushare 返回空 DataFrame")
    except Exception as e:
        logger.error(f"获取交易日历失败：{e}")
        # 备用方案：从数据库获取已有日期
        try:
            with engine.connect() as conn:
                query = text("""
                    SELECT DISTINCT trade_date 
                    FROM stock_daily 
                    WHERE trade_date >= :start_date AND trade_date <= :end_date
                    ORDER BY trade_date
                """)
                result = conn.execute(query, {
                    "start_date": start_date.replace('-', ''),
                    "end_date": end_date.replace('-', '')
                })
                trade_dates = [row[0] for row in result.fetchall()]
                logger.info(f"从数据库获取到 {len(trade_dates)} 个交易日")
                return trade_dates
        except Exception as e2:
            logger.error(f"从数据库获取交易日期也失败：{e2}")
    return []


def check_existing_data(trade_date: str) -> int:
    """检查数据库中已存在的该日期数据量"""
    try:
        with engine.connect() as conn:
            query = text("SELECT COUNT(*) as cnt FROM stock_daily WHERE trade_date = :trade_date")
            result = conn.execute(query, {"trade_date": trade_date})
            row = result.fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def handle_rate_limit_error(error_msg: str) -> bool:
    """处理频率限制错误，返回是否需要等待 60 秒"""
    rate_limit_patterns = [
        r'抱歉，您每分钟最多访问',
        r'积分不足',
        r'频率限制',
        r'429',
        r'403',
    ]
    for pattern in rate_limit_patterns:
        if re.search(pattern, error_msg):
            return True
    return False


def fetch_daily_data(trade_date: str) -> pd.DataFrame:
    """
    拉取指定日期的全市场数据
    关键：不传 ts_code 参数，直接按日期全量拉取
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily(trade_date=trade_date)
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()
        except Exception as e:
            error_msg = str(e)
            if handle_rate_limit_error(error_msg):
                logger.warning(f"触发频率限制，等待 60 秒... (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(60)
            else:
                logger.warning(f"拉取数据失败：{error_msg}, 等待 {SLEEP_SECONDS}秒后重试")
                time.sleep(SLEEP_SECONDS)
    
    return pd.DataFrame()


def fetch_daily_basic_data(trade_date: str) -> pd.DataFrame:
    """
    拉取指定日期的 daily_basic 数据（包含 turnover_rate 等关键字段）
    """
    for attempt in range(MAX_RETRIES):
        try:
            df = pro.daily_basic(trade_date=trade_date)
            if df is not None and not df.empty:
                return df
            return pd.DataFrame()
        except Exception as e:
            error_msg = str(e)
            if handle_rate_limit_error(error_msg):
                logger.warning(f"触发频率限制，等待 60 秒... (尝试 {attempt + 1}/{MAX_RETRIES})")
                time.sleep(60)
            else:
                logger.warning(f"拉取 daily_basic 失败：{error_msg}, 等待 {SLEEP_SECONDS}秒后重试")
                time.sleep(SLEEP_SECONDS)
    
    return pd.DataFrame()


def merge_daily_data(daily_df: pd.DataFrame, basic_df: pd.DataFrame) -> pd.DataFrame:
    """合并 daily 和 daily_basic 数据"""
    if daily_df.empty:
        return pd.DataFrame()
    
    if basic_df.empty:
        return daily_df
    
    # 左连接合并
    merged = pd.merge(daily_df, basic_df, on=['ts_code', 'trade_date'], how='left', suffixes=('', '_basic'))
    return merged


def save_to_database_upsert(df: pd.DataFrame, chunksize: int = CHUNK_SIZE):
    """
    保存数据到数据库，使用 ON DUPLICATE KEY UPDATE
    """
    if df is None or df.empty:
        return 0
    
    total_inserted = 0
    
    try:
        df = df.copy()
        
        # 字段映射
        column_mapping = {
            'ts_code': 'symbol',
        }
        df = df.rename(columns=column_mapping)
        
        # 日期格式转换：YYYYMMDD -> YYYY-MM-DD
        if 'trade_date' in df.columns:
            df['trade_date'] = df['trade_date'].astype(str)
            df['trade_date'] = df['trade_date'].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:8]}" if len(x) == 8 else x)
        
        # vol -> volume
        if 'vol' in df.columns and 'volume' not in df.columns:
            df['volume'] = df['vol']
        
        # 目标字段列表
        target_columns = [
            'symbol', 'trade_date', 'open', 'high', 'low', 'close',
            'pre_close', 'change', 'pct_chg', 'volume', 'amount',
            'adj_factor', 'turnover_rate', 'turnover_rate_f'
        ]
        
        # 确保所有目标列存在
        for col in target_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        df = df[target_columns]
        df = df.drop_duplicates(subset=['symbol', 'trade_date'])
        
        # 过滤掉 symbol 或 trade_date 为空的行
        df = df.dropna(subset=['symbol', 'trade_date'])
        
        if df.empty:
            return 0
        
        total_inserted = len(df)
        
        with engine.connect() as conn:
            df.to_sql('stock_daily', conn, if_exists='append', index=False, 
                     method='multi', chunksize=chunksize)
        
        return total_inserted
        
    except Exception as e:
        logger.error(f"写入数据库失败：{e}")
        return 0


def heal_missing_dates(trade_dates: list, start_year: int = 2025):
    """补齐缺失日期的数据"""
    healed_dates = []
    failed_dates = []
    
    total_start_time = time.time()
    
    for i, trade_date in enumerate(trade_dates):
        # 处理日期格式
        if hasattr(trade_date, 'strftime'):
            trade_date_str = trade_date.strftime('%Y%m%d')
        else:
            trade_date_str = str(trade_date).replace('-', '')
        
        # 检查年份范围（2025 年至今）
        date_year = int(trade_date_str[:4])
        if date_year < start_year:
            continue
        
        # 检查已存在的数据
        existing_count = check_existing_data(trade_date_str.replace('-', ''))
        
        if existing_count >= TARGET_MIN_SYMBOLS:
            continue
        
        # 记录开始时间
        start_time = time.time()
        
        # 拉取 daily 数据
        daily_df = fetch_daily_data(trade_date_str)
        
        if daily_df.empty:
            failed_dates.append(trade_date_str)
            continue
        
        # 拉取 daily_basic 数据
        basic_df = fetch_daily_basic_data(trade_date_str)
        
        # 合并数据
        merged_df = merge_daily_data(daily_df, basic_df)
        
        if merged_df.empty:
            merged_df = daily_df
        
        # 保存到数据库
        inserted_count = save_to_database_upsert(merged_df, chunksize=CHUNK_SIZE)
        
        # 计算耗时
        elapsed = time.time() - start_time
        
        if inserted_count > 0:
            healed_dates.append(trade_date_str)
            logger.success(f"[SUCCESS] {trade_date_str}, 插入行数：{inserted_count}, 耗时：{elapsed:.1f}s")
        else:
            failed_dates.append(trade_date_str)
            logger.warning(f"[FAILED] {trade_date_str}, 插入行数：0")
        
        # API 弹性：每次请求后等待（两次 API 调用，所以等待时间加倍）
        time.sleep(SLEEP_SECONDS * 2)
    
    total_elapsed = time.time() - total_start_time
    logger.info(f"补数完成，总耗时：{total_elapsed:.1f}s, 成功：{len(healed_dates)}天，失败：{len(failed_dates)}天")
    
    return healed_dates, failed_dates


def verify_and_report(start_year: int = 2025):
    """验证并输出最终结果"""
    try:
        with engine.connect() as conn:
            # 获取每日统计数据
            query = text("""
                SELECT trade_date, COUNT(*) as cnt 
                FROM stock_daily
                WHERE trade_date >= :start_date
                GROUP BY trade_date
                ORDER BY trade_date
            """)
            
            result = conn.execute(query, {
                "start_date": f"{start_year}-01-01"
            })
            
            rows = result.fetchall()
            
            if rows:
                df_stats = pd.DataFrame(rows, columns=['trade_date', 'cnt'])
                
                # 计算统计
                total_rows = df_stats['cnt'].sum()
                trade_days = len(df_stats)
                avg_daily = df_stats['cnt'].mean()
                min_daily = df_stats['cnt'].min()
                max_daily = df_stats['cnt'].max()
                
                return {
                    'total_rows': total_rows,
                    'trade_days': trade_days,
                    'avg_daily': avg_daily,
                    'min_daily': min_daily,
                    'max_daily': max_daily,
                    'details': df_stats
                }
    except Exception as e:
        logger.error(f"验证失败：{e}")
    
    return None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("V194 数据补完 - 2025 最终决战")
    logger.info("=" * 60)
    
    # 第一步：缺口审计
    logger.info("第一步：缺口审计 - 查询数据库中 2025 年已有的数据")
    
    # 获取所有交易日期（从 Tushare）
    today = datetime.now().strftime('%Y%m%d')
    trade_dates = get_all_trade_dates_from_tushare('20250101', today)
    
    if not trade_dates:
        logger.error("无法获取交易日期，退出")
        return
    
    logger.info(f"获取到 {len(trade_dates)} 个交易日期")
    
    # 获取已存在充足数据的日期
    existing_dates = get_existing_trade_dates('2025-01-01', today)
    dates_to_heal = [d for d in trade_dates if d not in existing_dates]
    
    logger.info(f"需要补数的交易日：{len(dates_to_heal)} 个")
    
    if not dates_to_heal:
        logger.info("所有日期数据已充足，无需补数")
    else:
        # 第二步：数据抓取与补完
        logger.info("第二步：开始数据抓取与补完")
        healed_dates, failed_dates = heal_missing_dates(dates_to_heal, start_year=2025)
    
    # 第三步：验收
    logger.info("=" * 60)
    logger.info("第三步：验收")
    logger.info("=" * 60)
    
    stats = verify_and_report(start_year=2025)
    
    if stats:
        logger.info(f"总行数：{stats['total_rows']}")
        logger.info(f"交易日数：{stats['trade_days']}")
        logger.info(f"平均每日股票数：{stats['avg_daily']:.0f}")
        logger.info(f"最小每日股票数：{stats['min_daily']}")
        logger.info(f"最大每日股票数：{stats['max_daily']}")
        
        if stats['avg_daily'] >= TARGET_MIN_SYMBOLS:
            logger.success(f"✓ 验收通过！平均每日记录数 {stats['avg_daily']:.0f} > {TARGET_MIN_SYMBOLS}")
        else:
            logger.warning(f"✗ 验收未通过！平均每日记录数 {stats['avg_daily']:.0f} < {TARGET_MIN_SYMBOLS}")
        
        # 输出详细统计
        print("\n" + "=" * 60)
        print("每日数据统计 (前 10 行):")
        print("=" * 60)
        print(stats['details'].head(10).to_string())


if __name__ == "__main__":
    main()