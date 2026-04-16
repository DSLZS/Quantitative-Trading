"""
V197 数据完整性检查脚本
检查 2023-2025 每日股票数量是否 > 5000
"""
import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine
from loguru import logger
import sys

logger.remove()
logger.add(sys.stdout, format='{message}', level='INFO')

load_dotenv()
db_url = os.getenv('DATABASE_URL')
engine = create_engine(db_url)

def check_data_integrity():
    """检查数据完整性"""
    logger.info('=' * 70)
    logger.info('V197 数据完整性检查 (2023-2025)')
    logger.info('=' * 70)
    
    # 检查 2023-2025 年每日股票数量
    query = '''
    SELECT trade_date, COUNT(*) as stock_count 
    FROM stock_daily 
    WHERE trade_date >= '2023-01-01' AND trade_date <= '2025-12-31'
    GROUP BY trade_date 
    ORDER BY trade_date
    '''
    
    logger.info('\n[Check] Querying daily stock counts...')
    df = pd.read_sql(query, engine)
    
    # 统计
    total_days = len(df)
    days_below_5000 = len(df[df['stock_count'] < 5000])
    avg_count = df['stock_count'].mean()
    min_count = df['stock_count'].min()
    
    logger.info(f'\n[Summary]')
    logger.info(f'  Total trading days: {total_days}')
    logger.info(f'  Days with stock_count < 5000: {days_below_5000}')
    logger.info(f'  Average stock count: {avg_count:.0f}')
    logger.info(f'  Min stock count: {min_count}')
    
    # 按年份统计
    df['year'] = pd.to_datetime(df['trade_date']).dt.year
    for year in [2023, 2024, 2025]:
        year_data = df[df['year'] == year]
        logger.info(f'  Year {year}: {len(year_data)} days, avg={year_data["stock_count"].mean():.0f}, min={year_data["stock_count"].min()}')
    
    # 显示 stock_count < 5000 的日期
    if days_below_5000 > 0:
        logger.warning('\n[Warning] Dates with stock_count < 5000:')
        for _, row in df[df['stock_count'] < 5000].head(50).iterrows():
            logger.warning(f'    {row["trade_date"]}: {row["stock_count"]}')
        return False, df
    else:
        logger.info('\n[PASS] All dates have stock_count >= 5000')
        return True, df

def check_factor_data():
    """检查因子数据"""
    logger.info('\n[Check] Checking factor data availability...')
    
    # 检查 stock_daily 表中的因子列
    query = '''
    SELECT COLUMN_NAME, DATA_TYPE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'stock_daily'
    ORDER BY ORDINAL_POSITION
    '''
    
    columns_df = pd.read_sql(query, engine)
    factor_cols = [
        'momentum_5', 'momentum_10', 'reversion_5',
        'volatility_5', 'volatility_20',
        'volume_rank', 'volume_price_contradiction', 'liquidity_alpha'
    ]
    
    available_cols = []
    for col in factor_cols:
        if col in columns_df['COLUMN_NAME'].values:
            available_cols.append(col)
            logger.info(f'  [OK] {col}')
        else:
            logger.warning(f'  [MISSING] {col}')
    
    return available_cols

def analyze_market_regime():
    """分析市场状态（基于价格和成交量）"""
    logger.info('\n' + '=' * 70)
    logger.info('2023-2025 市场状态分析')
    logger.info('=' * 70)
    
    # 获取 OHLCV 数据
    query = '''
    SELECT trade_date, symbol, open, high, low, close, volume, pct_chg
    FROM stock_daily 
    WHERE trade_date >= '2023-01-01' AND trade_date <= '2025-12-31'
    ORDER BY trade_date, symbol
    '''
    
    logger.info('\n[Analysis] Loading market data...')
    df = pd.read_sql(query, engine)
    
    # 计算市场状态指标
    # 1. 计算 ATR (Average True Range)
    df['prev_close'] = df.groupby('symbol')['close'].transform(lambda x: x.shift(1))
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df.groupby('symbol')['true_range'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['atr_ma20'] = df.groupby('symbol')['atr'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    
    # 2. 计算波动率状态
    df['volatility_regime'] = df['atr'] / (df['atr_ma20'] + 1e-10)
    
    # 3. 计算成交量异动
    df['volume_ma20'] = df.groupby('symbol')['volume'].transform(lambda x: x.rolling(20, min_periods=5).mean())
    df['volume_anomaly'] = df['volume'] / (df['volume_ma20'] + 1e-10)
    
    # 4. 计算市场偏度（用收益率分布）
    df['returns'] = df.groupby('symbol')['close'].transform(lambda x: x.pct_change())
    
    # 按日期聚合市场状态
    daily_regime = df.groupby('trade_date').agg({
        'volatility_regime': 'mean',
        'volume_anomaly': 'mean',
        'returns': ['mean', 'std', 'skew', 'kurt']
    }).reset_index()
    
    daily_regime.columns = ['trade_date', 'vol_regime', 'vol_anomaly', 'ret_mean', 'ret_std', 'ret_skew', 'ret_kurt']
    
    # 分类市场状态
    # 趋势：波动率低 + 收益率偏度高
    # 震荡：波动率低 + 收益率偏度低
    # 极端：波动率高
    
    vol_threshold_high = daily_regime['vol_regime'].quantile(0.8)
    vol_threshold_low = daily_regime['vol_regime'].quantile(0.2)
    skew_threshold = daily_regime['ret_skew'].quantile(0.5)
    
    def classify_regime(row):
        if row['vol_regime'] > vol_threshold_high:
            return 'EXTREME'  # 极端
        elif row['ret_skew'] > skew_threshold:
            return 'TREND'  # 趋势
        else:
            return 'RANGE'  # 震荡
    
    daily_regime['regime'] = daily_regime.apply(classify_regime, axis=1)
    
    # 按年份统计
    daily_regime['year'] = pd.to_datetime(daily_regime['trade_date']).dt.year
    
    logger.info('\n[Market Regime Summary]')
    for year in [2023, 2024, 2025]:
        year_data = daily_regime[daily_regime['year'] == year]
        regime_counts = year_data['regime'].value_counts()
        logger.info(f'  Year {year}:')
        for regime, count in regime_counts.items():
            pct = count / len(year_data) * 100
            logger.info(f'    {regime}: {count} days ({pct:.1f}%)')
    
    # 2024 年详细分析
    logger.info('\n[2024 Deep Analysis]')
    regime_2024 = daily_regime[daily_regime['year'] == 2024]
    
    for regime in ['TREND', 'RANGE', 'EXTREME']:
        sub = regime_2024[regime_2024['regime'] == regime]
        if len(sub) > 0:
            logger.info(f'  {regime} regime:')
            logger.info(f'    Days: {len(sub)}')
            logger.info(f'    Avg Vol Regime: {sub["vol_regime"].mean():.3f}')
            logger.info(f'    Avg Return Skew: {sub["ret_skew"].mean():.3f}')
            logger.info(f'    Avg Return Kurt: {sub["ret_kurt"].mean():.3f}')
    
    return daily_regime

if __name__ == '__main__':
    passed, df = check_data_integrity()
    check_factor_data()
    analyze_market_regime()
    
    logger.info('\n' + '=' * 70)
    if passed:
        logger.info('[RESULT] Data integrity check PASSED')
    else:
        logger.warning('[RESULT] Data integrity check FAILED - Need to run DataHealer')
    logger.info('=' * 70)
