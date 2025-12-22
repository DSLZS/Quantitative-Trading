# src/data/data_fetcher.py
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import logging
import time
from enum import Enum
import re

'''
# 输入: 任意格式 -> 标准格式 -> 分离存储
用户输入: "sh000001" 
↓ DataFetcher.normalize_symbol()
标准格式: "000001.SH"
↓ DataFetcher.parse_symbol() 
解析结果: (code="000001", exchange="SH", akshare_symbol="sh000001")
↓ 数据库存储
symbol列: "000001" (纯代码)
market列: "SSE" (市场代码)
'''

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
        
        # 标准代码到akshare代码的映射
        self.code_to_akshare = {
            'SH': 'sh',
            'SZ': 'sz',
            'BJ': 'bj'
        }
        
        # 反向映射
        self.akshare_to_code = {v: k for k, v in self.code_to_akshare.items()}
        
        # 定义数据库表所需的完整列结构
        self.expected_columns = [
            'symbol', 'market', 'trade_date', 'open', 'high', 'low', 'close',
            'volume', 'amount', 'adjust_factor_back', 'adjust_factor_forward',
            'change', 'pct_change', 'turnover_rate'
        ]
    
    def parse_symbol(self, symbol: str) -> Tuple[str, str, str]:
        """
        解析标准格式股票代码
        
        Args:
            symbol: 标准格式股票代码，如 '510300.SH'
            
        Returns:
            (纯代码, 市场后缀, akshare格式)
        """
        # 支持多种格式输入，但统一转为标准格式处理
        symbol = str(symbol).strip()
        
        if '.' in symbol:
            # 标准格式: 510300.SH
            code, exchange = symbol.split('.')
            exchange = exchange.upper()
        elif re.match(r'^[a-z]{2}\d{6}$', symbol.lower()):
            # akshare格式: sh000001
            exchange = symbol[:2].upper()
            code = symbol[2:]
        elif re.match(r'^\d{6}$', symbol):
            # 纯数字格式，需要根据代码规则判断市场
            code = symbol
            if code.startswith(('60', '68')):
                exchange = 'SH'
            elif code.startswith(('00', '30')):
                exchange = 'SZ'
            elif code.startswith(('43', '83', '87')):
                exchange = 'BJ'
            else:
                exchange = 'SZ'  # 默认深市
        else:
            raise ValueError(f"不支持的股票代码格式: {symbol}")
        
        # 转换为akshare需要的格式
        akshare_symbol = f"{self.code_to_akshare.get(exchange, exchange.lower())}{code}"
        
        return code, exchange, akshare_symbol
    
    def normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码格式
        
        Args:
            symbol: 任意格式的股票代码
            
        Returns:
            标准格式: 代码.市场 (如 510300.SH)
        """
        try:
            code, exchange, _ = self.parse_symbol(symbol)
            return f"{code}.{exchange}"
        except:
            # 如果解析失败，返回原样
            return symbol
    
    def get_market_code(self, exchange: str) -> str:
        """将交易所代码转换为市场代码"""
        market_mapping = {
            'SH': Market.SSE.value,
            'SZ': Market.SZSE.value,
            'BJ': Market.BSE.value,
        }
        return market_mapping.get(exchange.upper(), exchange.upper())
    
    def fetch_stock_daily(self, symbol: str, start_date: str = None, 
                          end_date: str = None, adjust: str = "qfq") -> pd.DataFrame:
        """
        获取单只股票的日线数据
        
        Args:
            symbol: 股票代码（标准格式'510300.SH'或akshare格式'sh000001'）
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期，格式'YYYY-MM-DD'
            adjust: 复权类型 'qfq': 前复权, 'hfq': 后复权, None: 不复权
            
        Returns:
            DataFrame 包含日线数据，包含所有数据库表需要的列
            返回标准格式: symbol='510300', market='SSE'
        """
        try:
            # 解析股票代码
            code, exchange, akshare_symbol = self.parse_symbol(symbol)
            
            # 设置默认日期
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            logger.info(f"正在获取 {code}.{exchange} 从 {start_date} 到 {end_date} 的数据")
            
            # 根据市场类型选择不同的akshare函数
            market_code = akshare_symbol[:2]
            stock_code = akshare_symbol[2:]
            
            df = pd.DataFrame()
            
            # 对于ETF基金，使用专门的函数
            if stock_code.startswith(('50', '51', '15', '16')):
                logger.info(f"检测到ETF/LOF基金: {stock_code}")
                if market_code == 'sh':
                    df = ak.fund_etf_hist_sina(symbol=f"sh{stock_code}")
                elif market_code == 'sz':
                    df = ak.fund_etf_hist_sina(symbol=f"sz{stock_code}")
                
                if not df.empty:
                    # ETF数据列名特殊处理
                    if 'date' in df.columns:
                        df = df.rename(columns={'date': 'trade_date'})
                    if '开盘' in df.columns:
                        df = df.rename(columns={'开盘': 'open'})
                    if '收盘' in df.columns:
                        df = df.rename(columns={'收盘': 'close'})
                    if '最高' in df.columns:
                        df = df.rename(columns={'最高': 'high'})
                    if '最低' in df.columns:
                        df = df.rename(columns={'最低': 'low'})
                    if '成交量' in df.columns:
                        df = df.rename(columns={'成交量': 'volume'})
                    if '成交额' in df.columns:
                        df = df.rename(columns={'成交额': 'amount'})
                    
                    # 添加基本信息
                    df['symbol'] = stock_code
                    df['market'] = self.get_market_code(exchange)
            
            elif market_code == 'sh':
                # 上海交易所股票
                df = ak.stock_zh_a_hist(
                    symbol=stock_code, period="daily", 
                    start_date=start_date.replace('-', ''), 
                    end_date=end_date.replace('-', ''),
                    adjust=adjust
                )
                
            elif market_code == 'sz':
                # 深圳交易所股票
                df = ak.stock_zh_a_hist(
                    symbol=stock_code, period="daily", 
                    start_date=start_date.replace('-', ''), 
                    end_date=end_date.replace('-', ''),
                    adjust=adjust
                )
                
            elif market_code == 'bj':
                # 北京交易所
                df = ak.stock_bj_a_hist(symbol=stock_code)
            
            else:
                logger.error(f"不支持的市场: {market_code}")
                return pd.DataFrame()
            
            if df.empty:
                logger.warning(f"未获取到 {code}.{exchange} 的数据")
                return pd.DataFrame()
            
            # 重命名列
            df = self.standardize_columns(df)
            
            # 添加股票基本信息（如果还没有）
            if 'symbol' not in df.columns:
                df['symbol'] = code
            if 'market' not in df.columns:
                df['market'] = self.get_market_code(exchange)
            
            # 确保所有需要的列都存在
            df = self.ensure_all_columns(df, adjust)
            
            # 格式化数据
            df = self.format_data(df)
            
            logger.info(f"成功获取 {code}.{exchange} 共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"获取 {symbol} 数据失败: {e}", exc_info=True)
            return pd.DataFrame()
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # 应用列名映射
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        return df
    
    def ensure_all_columns(self, df: pd.DataFrame, adjust: str) -> pd.DataFrame:
        """确保DataFrame包含所有需要的列"""
        if df.empty:
            return df
        
        # 记录原始列
        original_columns = list(df.columns)
        logger.debug(f"原始列: {original_columns}")
        
        # 首先添加缺失的基础列
        for col in ['symbol', 'market']:
            if col not in df.columns:
                df[col] = ''
        
        # 确保trade_date列存在且格式正确
        if 'trade_date' not in df.columns:
            # 尝试找到日期列
            date_cols = [c for c in df.columns if 'date' in c.lower() or '时间' in c or '日期' in c]
            if date_cols:
                df['trade_date'] = df[date_cols[0]]
            else:
                # 如果没有日期列，创建一个空列
                df['trade_date'] = pd.NaT
        
        # 价格和交易量列
        price_volume_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in price_volume_cols:
            if col not in df.columns:
                df[col] = 0.0 if col != 'volume' else 0
        
        # 复权因子
        if 'adjust_factor_back' not in df.columns:
            df['adjust_factor_back'] = 1.0
        if 'adjust_factor_forward' not in df.columns:
            df['adjust_factor_forward'] = 1.0
        
        # 涨跌额和涨跌幅
        if 'change' not in df.columns:
            if 'close' in df.columns:
                df['change'] = df['close'].diff()
            else:
                df['change'] = 0.0
        
        if 'pct_change' not in df.columns:
            if 'close' in df.columns:
                df['pct_change'] = df['close'].pct_change() * 100
            else:
                df['pct_change'] = 0.0
        
        # 换手率
        if 'turnover_rate' not in df.columns:
            df['turnover_rate'] = None
        
        # 确保列的顺序
        available_columns = [col for col in self.expected_columns if col in df.columns]
        extra_columns = [col for col in df.columns if col not in available_columns]
        
        df = df[available_columns + extra_columns]
        
        logger.debug(f"处理后列: {list(df.columns)}")
        return df
    
    def format_data(self, df: pd.DataFrame) -> pd.DataFrame:
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
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df[col] = 0.0 if col != 'turnover_rate' else None
        
        # 确保volume是整数类型
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype('Int64')
        
        return df
    
    def fetch_stock_list(self, market: str = None) -> List[str]:
        """
        获取股票列表
        
        Args:
            market: 市场代码 'SH', 'SZ', 'BJ' 或 None（全部）
            
        Returns:
            股票代码列表（标准格式：'510300.SH'）
        """
        try:
            market_prefix = None
            if market:
                # 将标准市场代码转换为akshare前缀
                market_prefix = self.code_to_akshare.get(market.upper())
                if not market_prefix:
                    logger.error(f"不支持的市场代码: {market}")
                    return []
            
            # 调用akshare获取数据
            if market_prefix == 'sh':
                df = ak.stock_sh_a_spot()
            elif market_prefix == 'sz':
                df = ak.stock_sz_a_spot()
            elif market_prefix == 'bj':
                df = ak.stock_bj_a_spot()
            else:
                df = ak.stock_zh_a_spot()
            
            # 提取代码列表并转换为标准格式
            stock_list = []
            for code in df['代码'].tolist():
                code_str = str(code).zfill(6)  # 确保6位代码
                
                # 判断市场
                if code_str.startswith(('60', '68')):
                    exchange = 'SH'
                elif code_str.startswith(('00', '30')):
                    exchange = 'SZ'
                elif code_str.startswith(('43', '83', '87')):
                    exchange = 'BJ'
                else:
                    continue  # 跳过无法识别的代码
                
                stock_list.append(f"{code_str}.{exchange}")
            
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
            symbols: 股票代码列表（标准格式）
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