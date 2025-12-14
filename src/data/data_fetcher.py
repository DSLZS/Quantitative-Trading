# src/data/data_fetcher.py
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)

class Market(Enum):
    """市场枚举"""
    SSE = "SSE"      # 上海证券交易所
    SZSE = "SZSE"    # 深圳证券交易所
    BSE = "BSE"      # 北京证券交易所

class DataFetcher:
    """数据获取类"""
    
    def __init__(self, source='akshare'):
        self.source = source
        
        # 市场代码映射
        self.market_mapping = {
            'sh': Market.SSE.value,    # 上海
            'sz': Market.SZSE.value,   # 深圳
            'bj': Market.BSE.value,    # 北京
        }
        
        # 定义数据库表所需的完整列结构
        self.expected_columns = [
            'symbol', 'market', 'trade_date', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'adjust_factor_back', 'adjust_factor_forward',
            'change', 'pct_change', 'turnover_rate'
        ]
    
    def fetch_stock_daily(self, symbol: str, start_date: str = None, 
                          end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取单只股票的日线数据
        
        Args:
            symbol: 股票代码（带市场前缀，如'sh000001'或'sz000001'）
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            adjust: 复权类型 'qfq': 前复权, 'hfq': 后复权, None: 不复权
            
        Returns:
            DataFrame 包含日线数据，包含所有数据库表需要的列
        """
        try:
            # 解析市场和代码
            market_code = symbol[:2].lower()
            stock_code = symbol[2:]
            
            if market_code not in self.market_mapping:
                logger.error(f"不支持的市场代码: {market_code}")
                return pd.DataFrame()
            
            market = self.market_mapping[market_code]
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"正在获取 {symbol} 从 {start_date} 到 {end_date} 的数据")
            
            # 使用akshare获取数据
            df = pd.DataFrame()
            if market_code == 'sh':
                # 上海交易所
                if stock_code.startswith('000') or stock_code.startswith('001'): # 上证主板
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code, period="daily", 
                        start_date=start_date.replace('-', ''), 
                        end_date=end_date.replace('-', ''),
                        adjust=adjust
                    )
                elif stock_code.startswith('688'):  
                    # 科创板 不支持日期传参
                    df = ak.stock_zh_kcb_daily(symbol=stock_code)
                elif stock_code.startswith('50') or stock_code.startswith('51'):
                    # ETF/LOF 不支持日期传参
                    df = ak.fund_etf_hist_sina(symbol=f"sh{stock_code}")
                else:
                    logger.error(f"不支持的上海股票代码: {stock_code}")
                    return pd.DataFrame()
                    
            elif market_code == 'sz':
                # 深圳交易所
                if stock_code.startswith('000') or stock_code.startswith('001') or stock_code.startswith('002'):
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code, period="daily", 
                        start_date=start_date.replace('-', ''), 
                        end_date=end_date.replace('-', ''),
                        adjust=adjust
                    )
                elif stock_code.startswith('300'):
                    # 创业板
                    df = ak.stock_zh_a_hist(
                        symbol=stock_code, period="daily", 
                        start_date=start_date.replace('-', ''), 
                        end_date=end_date.replace('-', ''),
                        adjust=adjust
                    )
                elif stock_code.startswith('15') or stock_code.startswith('16'):
                    # ETF/LOF 不支持日期传参
                    df = ak.fund_etf_hist_sina(symbol=f"sz{stock_code}")
                else:
                    logger.error(f"不支持的深圳股票代码: {stock_code}")
                    return pd.DataFrame()
                    
            elif market_code == 'bj':
                # 北京交易所
                df = ak.stock_bj_a_hist(symbol=stock_code)
            
            else:
                logger.error(f"不支持的市场: {market_code}")
                return pd.DataFrame()
            
            if df.empty:
                logger.warning(f"未获取到 {symbol} 的数据")
                return pd.DataFrame()
            
            # 重命名列
            df = self._standardize_columns(df, market_code)
            
            # 添加股票基本信息
            df['symbol'] = stock_code
            df['market'] = market
            
            # 确保所有需要的列都存在
            df = self._ensure_all_columns(df, adjust)
            
            # 格式化数据
            df = self._format_data(df)
            
            logger.info(f"成功获取 {symbol} 共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame, market_code: str) -> pd.DataFrame:
        """标准化列名和格式"""
        column_mapping = {
            '日期': 'trade_date',
            'date': 'trade_date',
            '开盘': 'open',
            '开盘价': 'open',
            '收盘': 'close',
            '收盘价': 'close',
            '最高': 'high',
            '最高价': 'high',
            '最低': 'low',
            '最低价': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
            '涨跌幅': 'pct_change',
            '涨跌额': 'change',
            '换手率': 'turnover_rate',
            '振幅': 'amplitude',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df
    
    def _ensure_all_columns(self, df: pd.DataFrame, adjust: str) -> pd.DataFrame:
        """确保DataFrame包含所有需要的列"""
        
        # 检查并添加缺失的列
        for col in self.expected_columns:
            if col not in df.columns:
                # 根据列名设置默认值
                if col == 'adjust_factor_back':
                    # 后复权因子：如果是后复权，可能是1.0；其他情况可能需要计算
                    df[col] = 1.0 if adjust != 'hfq' else 1.0
                elif col == 'adjust_factor_forward':
                    # 前复权因子：如果是前复权，可能是1.0；其他情况可能需要计算
                    df[col] = 1.0 if adjust != 'qfq' else 1.0
                elif col == 'change':
                    # 涨跌额：如果close存在，计算差值
                    if 'close' in df.columns:
                        df[col] = df['close'].diff()
                    else:
                        df[col] = None
                elif col == 'pct_change':
                    # 涨跌幅：如果change和close存在，计算百分比
                    if 'change' in df.columns and 'close' in df.columns:
                        df[col] = df['change'] / df['close'].shift(1) * 100
                    else:
                        df[col] = None
                elif col == 'turnover_rate':
                    # 换手率：如果不存在，设为None
                    df[col] = None
                else:
                    # 其他缺失的数值列设为None
                    df[col] = None
        
        # 确保列的顺序一致
        df = df[self.expected_columns]
        return df
    
    def _format_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """格式化数据"""
        # 格式化日期
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.strftime('%Y-%m-%d')
        
        # 确保数值类型
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount',
                       'adjust_factor_back', 'adjust_factor_forward',
                       'change', 'pct_change', 'turnover_rate']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 确保volume是整数类型
        if 'volume' in df.columns:
            df['volume'] = df['volume'].astype('Int64')
        
        return df
    
    def fetch_stock_list(self, market: str = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            market: 市场代码 'sh', 'sz', 'bj' 或 None（全部）
            
        Returns:
            股票代码列表
        """
        try:
            if market is None:
                # 获取全部A股
                df = ak.stock_zh_a_spot()
            elif market == 'sh':
                df = ak.stock_sh_a_spot()
            elif market == 'sz':
                df = ak.stock_sz_a_spot()
            elif market == 'bj':
                df = ak.stock_bj_a_spot()
            else:
                logger.error(f"不支持的市场: {market}")
                return []
            
            # 提取代码列表
            codes = df['代码'].tolist()
            # 添加市场前缀
            market_prefix = market if market else ''
            stock_list = [f"{market_prefix}{code}" for code in codes]
            
            logger.info(f"成功获取 {market or '全部'} 共 {len(stock_list)} 只股票")
            return stock_list
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []
    
    def fetch_multiple_stocks(self, symbols: List[str], start_date: str = None, 
                             end_date: str = None, batch_size: int = 10, 
                             delay: float = 0.5) -> pd.DataFrame:
        """
        批量获取多只股票的日线数据
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            batch_size: 每批处理的数量
            delay: 请求间隔（秒），避免被封
            
        Returns:
            合并后的DataFrame
        """
        all_data = []
        
        for i, symbol in enumerate(symbols):
            try:
                logger.info(f"正在获取第 {i+1}/{len(symbols)} 只股票: {symbol}")
                df = self.fetch_stock_daily(symbol, start_date, end_date)
                
                if not df.empty:
                    all_data.append(df)
                else:
                    logger.warning(f"股票 {symbol} 数据为空")
                
                # 控制请求频率
                if i < len(symbols) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"获取 {symbol} 失败: {e}")
                continue
        
        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            logger.info(f"批量获取完成，共获取 {len(result)} 条数据")
            return result
        else:
            logger.warning("未获取到任何数据")
            return pd.DataFrame()