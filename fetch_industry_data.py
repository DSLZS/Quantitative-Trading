"""
使用 Tushare API 拉取股票行业数据并补全到 stock_industry_daily 表
"""

import pandas as pd
import tushare as ts
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from datetime import datetime, timedelta
from loguru import logger
import sys

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# Tushare Token
TUSHARE_TOKEN = "319a931c8fd896d1c42edfd29f7787ab26b06fae5915c5db57557a21"

# 数据库配置
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/quantitative_trading"

# 初始化 Tushare
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 数据库引擎
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_pre_ping=True)


def get_trading_dates(start_date: str, end_date: str) -> list:
    """获取交易日历 - 从数据库中获取已有的交易日期"""
    logger.info(f"Getting trading dates from {start_date} to {end_date}...")
    
    # 从数据库中获取 stock_daily 表中的交易日期
    try:
        query = text("""
            SELECT DISTINCT trade_date 
            FROM stock_daily 
            WHERE YEAR(trade_date) = :year 
            ORDER BY trade_date
        """)
        year = int(start_date[:4])
        
        with engine.connect() as conn:
            result = conn.execute(query, {'year': year})
            dates = [row[0] for row in result.fetchall()]
        
        # 转换为字符串格式
        trading_dates = [pd.to_datetime(d).strftime('%Y%m%d') for d in dates]
        logger.info(f"Found {len(trading_dates)} trading days from database")
        return trading_dates
    except Exception as e:
        logger.warning(f"Failed to get trading dates from database: {e}")
        # 返回一个简化的日期列表
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq='B')  # 工作日
        return [d.strftime('%Y%m%d') for d in dates]


def get_stock_industry_info() -> pd.DataFrame:
    """获取股票行业分类信息（使用最新数据）"""
    logger.info("Fetching stock industry classification from Tushare...")
    
    try:
        # 获取申万行业分类
        df = pro.index_classify(level='L1')
        logger.info(f"Found {len(df)} industry entries")
        return df
    except Exception as e:
        logger.warning(f"Failed to get industry classification: {e}")
        return pd.DataFrame()


def get_stock_basic_info() -> pd.DataFrame:
    """获取股票基本信息"""
    logger.info("Fetching stock basic info from Tushare...")
    
    try:
        # 获取股票列表
        df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
        logger.info(f"Found {len(df)} stocks")
        return df
    except Exception as e:
        logger.warning(f"Failed to get stock basic info: {e}")
        return pd.DataFrame()


def fetch_industry_data_for_date(trade_date: str) -> pd.DataFrame:
    """
    获取指定日期的股票行业数据
    
    由于行业分类相对稳定，我们使用股票基本信息中的行业数据
    """
    try:
        # 获取股票基本信息
        stock_info = get_stock_basic_info()
        
        if stock_info.empty:
            return pd.DataFrame()
        
        # 格式化日期
        trade_date_dt = pd.to_datetime(trade_date)
        
        # 准备数据
        result_data = []
        for _, row in stock_info.iterrows():
            ts_code = row['ts_code']
            symbol = row['symbol']
            
            # 确保 symbol 格式正确（6 位数字）
            if symbol and len(symbol) <= 6:
                symbol = symbol.zfill(6)
            else:
                continue
            
            # 行业信息
            industry_name = row.get('industry', '')
            if pd.isna(industry_name) or industry_name == '':
                industry_name = 'Unknown'
            
            # 行业代码（使用行业名称作为代码）
            industry_code = f"IND_{industry_name}" if industry_name != 'Unknown' else 'UNKNOWN'
            
            result_data.append({
                'symbol': symbol,
                'trade_date': trade_date_dt,
                'industry_name': industry_name,
                'industry_code': industry_code,
            })
        
        return pd.DataFrame(result_data)
    
    except Exception as e:
        logger.error(f"Error fetching industry data for {trade_date}: {e}")
        return pd.DataFrame()


def save_industry_data_to_db(df: pd.DataFrame):
    """保存行业数据到数据库"""
    if df.empty:
        logger.warning("Empty dataframe, skipping save")
        return 0
    
    saved_count = 0
    
    try:
        # 批量插入/更新
        batch_size = 5000
        
        with engine.connect() as conn:
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    try:
                        # 先检查是否存在
                        check_query = text("""
                            SELECT COUNT(*) FROM stock_industry_daily 
                            WHERE symbol = :symbol AND trade_date = :trade_date
                        """)
                        count = conn.execute(check_query, {
                            'symbol': row['symbol'],
                            'trade_date': row['trade_date']
                        }).scalar()
                        
                        if count > 0:
                            # 更新
                            update_query = text("""
                                UPDATE stock_industry_daily 
                                SET industry_name = :industry_name, industry_code = :industry_code
                                WHERE symbol = :symbol AND trade_date = :trade_date
                            """)
                            conn.execute(update_query, {
                                'industry_name': row['industry_name'],
                                'industry_code': row['industry_code'],
                                'symbol': row['symbol'],
                                'trade_date': row['trade_date']
                            })
                        else:
                            # 插入
                            insert_query = text("""
                                INSERT INTO stock_industry_daily (symbol, trade_date, industry_name, industry_code)
                                VALUES (:symbol, :trade_date, :industry_name, :industry_code)
                            """)
                            conn.execute(insert_query, {
                                'symbol': row['symbol'],
                                'trade_date': row['trade_date'],
                                'industry_name': row['industry_name'],
                                'industry_code': row['industry_code']
                            })
                        
                        saved_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to save {row['symbol']}/{row['trade_date']}: {e}")
                
                conn.commit()
        
        logger.info(f"Saved {saved_count} rows to database")
        return saved_count
    
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        return 0


def fill_industry_data_for_year(year: int):
    """补全指定年份的行业数据"""
    logger.info(f"\n{'='*80}")
    logger.info(f"Filling industry data for year {year}")
    logger.info(f"{'='*80}")
    
    # 获取该年份的交易日期
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    trading_dates = get_trading_dates(start_date, end_date)
    
    if not trading_dates:
        logger.warning(f"No trading dates found for {year}")
        return 0
    
    total_saved = 0
    
    # 先获取股票基本信息（一次获取，重复使用）
    logger.info("Fetching stock basic info once for efficiency...")
    stock_info = get_stock_basic_info()
    
    if stock_info.empty:
        logger.warning("No stock info found, skipping")
        return 0
    
    # 准备股票映射
    stock_industry_map = {}
    for _, row in stock_info.iterrows():
        ts_code = row['ts_code']
        symbol = row['symbol']
        
        if symbol and len(symbol) <= 6:
            symbol = symbol.zfill(6)
        
        industry_name = row.get('industry', '')
        if pd.isna(industry_name) or industry_name == '':
            industry_name = 'Unknown'
        
        industry_code = f"IND_{industry_name}" if industry_name != 'Unknown' else 'UNKNOWN'
        
        stock_industry_map[symbol] = {
            'industry_name': industry_name,
            'industry_code': industry_code,
        }
    
    logger.info(f"Prepared industry mapping for {len(stock_industry_map)} stocks")
    
    # 按日期处理
    with engine.connect() as conn:
        for i, trade_date in enumerate(trading_dates):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing date {i+1}/{len(trading_dates)}: {trade_date}")
            
            trade_date_dt = pd.to_datetime(trade_date)
            
            # 准备该日期的数据
            batch_data = []
            for symbol, industry_info in stock_industry_map.items():
                batch_data.append({
                    'symbol': symbol,
                    'trade_date': trade_date_dt,
                    'industry_name': industry_info['industry_name'],
                    'industry_code': industry_info['industry_code'],
                })
            
            df = pd.DataFrame(batch_data)
            
            # 批量保存
            batch_size = 5000
            for j in range(0, len(df), batch_size):
                batch = df.iloc[j:j+batch_size]
                
                for _, row in batch.iterrows():
                    try:
                        # 检查是否存在
                        check_query = text("""
                            SELECT COUNT(*) FROM stock_industry_daily 
                            WHERE symbol = :symbol AND trade_date = :trade_date
                        """)
                        count = conn.execute(check_query, {
                            'symbol': row['symbol'],
                            'trade_date': row['trade_date']
                        }).scalar()
                        
                        if count > 0:
                            # 更新
                            update_query = text("""
                                UPDATE stock_industry_daily 
                                SET industry_name = :industry_name, industry_code = :industry_code
                                WHERE symbol = :symbol AND trade_date = :trade_date
                            """)
                            conn.execute(update_query, {
                                'industry_name': row['industry_name'],
                                'industry_code': row['industry_code'],
                                'symbol': row['symbol'],
                                'trade_date': row['trade_date']
                            })
                        else:
                            # 插入
                            insert_query = text("""
                                INSERT INTO stock_industry_daily (symbol, trade_date, industry_name, industry_code)
                                VALUES (:symbol, :trade_date, :industry_name, :industry_code)
                            """)
                            conn.execute(insert_query, {
                                'symbol': row['symbol'],
                                'trade_date': row['trade_date'],
                                'industry_name': row['industry_name'],
                                'industry_code': row['industry_code']
                            })
                        
                        total_saved += 1
                    except Exception as e:
                        pass
                
                conn.commit()
    
    logger.info(f"Saved/Updated {total_saved} rows for year {year}")
    return total_saved


def verify_industry_data(years: list):
    """验证行业数据"""
    logger.info(f"\n{'='*80}")
    logger.info("Verifying industry data...")
    logger.info(f"{'='*80}")
    
    with engine.connect() as conn:
        for year in years:
            query = text("""
                SELECT 
                    COUNT(*) as total_rows,
                    COUNT(DISTINCT symbol) as unique_stocks,
                    COUNT(DISTINCT industry_code) as unique_industries,
                    COUNT(DISTINCT industry_name) as unique_industry_names
                FROM stock_industry_daily
                WHERE YEAR(trade_date) = :year
            """)
            result = conn.execute(query, {'year': year}).fetchone()
            
            logger.info(f"Year {year}: {result.total_rows:,} rows, "
                       f"{result.unique_stocks:,} stocks, "
                       f"{result.unique_industries} industry codes, "
                       f"{result.unique_industry_names} industry names")


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("Tushare Industry Data Fetcher")
    logger.info("="*80)
    
    # 需要补全年份
    years_to_fill = [2020, 2022, 2024]
    
    for year in years_to_fill:
        fill_industry_data_for_year(year)
    
    # 验证结果
    verify_industry_data(years_to_fill)
    
    logger.info("\n" + "="*80)
    logger.info("Industry data fetch complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()