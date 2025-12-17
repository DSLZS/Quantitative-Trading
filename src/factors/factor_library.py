import pandas as pd
import numpy as np
from typing import Union, Optional

def momentum(close: pd.Series, window: int = 20) -> pd.Series:
    """动量因子：过去N日的收益率。"""
    return close.pct_change(window)

def volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """波动率因子：过去N日收益率的标准差。"""
    returns = close.pct_change()
    return returns.rolling(window).std()

def ma_ratio(close: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """均线比率因子：短期均线 / 长期均线。"""
    ma_short = close.rolling(short_window).mean()
    ma_long = close.rolling(long_window).mean()
    return (ma_short / ma_long) - 1  # 中心化，使得0表示均线交叉点

def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI相对强弱指标。"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD指标：DIF线。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    # dea = dif.ewm(span=signal, adjust=False).mean()
    # histogram = dif - dea
    return dif

def turnover_ratio(volume: pd.Series, float_mv: Optional[pd.Series] = None) -> pd.Series:
    """换手率因子：成交量 / 流通市值（如有），或使用成交量本身。"""
    if float_mv is not None and not float_mv.isnull().all():
        return volume / float_mv
    return volume  # 若无流通市值，则直接使用成交量作为代理

def price_to_earnings(pe_ratio: pd.Series) -> pd.Series:
    """市盈率因子（通常取其倒数或分位数）。"""
    # 为避免无穷大，处理PE为负或零的情况
    pe_pos = pe_ratio.where(pe_ratio > 0, np.nan)
    return 1 / pe_pos  # EP比率，越高代表估值越低

def book_to_price(pb_ratio: pd.Series) -> pd.Series:
    """市净率因子（取其倒数，即BP比率）。"""
    pb_pos = pb_ratio.where(pb_ratio > 0, np.nan)
    return 1 / pb_pos

def liquidity(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """流动性因子：Amihud非流动性指标，衡量单位交易量引起的价格冲击。"""
    returns = close.pct_change().abs()
    dollar_volume = close * volume
    illiq = returns / dollar_volume
    return illiq.rolling(window).mean()