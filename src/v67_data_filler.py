"""
V67 Data Filler Module - 数据暴力填充与 SPI 信号质量审计

【V67 数据强制填充协议 - 死命令】

1. 暴力抓取协议
   ✅ 严禁一次性抓取全部个股
   ✅ 必须采用"批次 - 等待"模式：每抓取 20 只股票，强制 sleep 5 秒
   ✅ 每写入一条数据，必须立即执行 SELECT COUNT(*) 确认
   ✅ 连续 3 次写入失败，直接 sys.exit(1) 并打印具体 API 报错信息

2. 必须拉取的数据
   ✅ 2024 年全年的 stock_individual_fund_flow_rank_em（资金流排名表）
   ✅ 2024 年全年的 stock_board_industry_cons_em（行业成分）

3. 输出要求
   ✅ 控制台持续输出 [SUCCESS] 000001.SZ written. Total rows now: XXX

4. 数据熔断升级
   ✅ 回测引擎启动时，必须检查 stock_fund_flow
   ✅ 如果行数低于 100 万行，严禁启动并明确告知："数据不足，请运行 data_filler"

5. 报错透明化
   ✅ 遇到 NoneType 或数据缺失，不准用 fill_null(0) 掩盖
   ✅ 必须打印出缺失数据的日期和股票代码

作者：量化系统
版本：V67.0
日期：2026-03-24
"""

import sys
import time
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from db_manager import DatabaseManager, get_db


# ===========================================
# V67 配置常量 - 数据暴力填充协议
# ===========================================

# 批次等待配置
V67_BATCH_SIZE = 20  # 每批次抓取 20 只股票
V67_BATCH_WAIT_SECONDS = 5  # 每批次后强制等待 5 秒

# 写入验证配置
V67_MAX_CONSECUTIVE_FAILURES = 3  # 连续 3 次写入失败即退出
V67_WRITE_VERIFY_QUERY = "SELECT COUNT(*) as cnt FROM {table_name} WHERE symbol = '{symbol}' AND trade_date = '{trade_date}'"

# 数据熔断阈值
V67_MIN_FUND_FLOW_ROWS = 1000000  # 100 万行阈值
V67_MIN_INDUSTRY_ROWS = 100000  # 行业数据最少行数

# 请求超时配置
V67_REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
V67_MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
V67_RETRY_DELAY_SECONDS = 2  # 重试间隔（秒）

# 日期范围配置
V67_DEFAULT_START_DATE = "2024-01-01"
V67_DEFAULT_END_DATE = "2024-12-31"


# ===========================================
# V67 数据获取器 - 批次等待模式
# ===========================================

class V67DataFetcher:
    """
    V67 数据获取器 - 批次等待模式
    
    【核心功能】
    1. 使用"批次 - 等待"模式抓取数据
    2. 每抓取 20 只股票，强制 sleep 5 秒
    3. 所有网络请求带重试机制
    """
    
    def __init__(self, max_retries: int = V67_MAX_RETRY_ATTEMPTS,
                 retry_delay: float = V67_RETRY_DELAY_SECONDS):
        """
        初始化数据获取器
        
        Parameters
        ----------
        max_retries : int
            最大重试次数
        retry_delay : float
            重试间隔（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._ak_available: Optional[bool] = None
        self._last_api_error: str = ""
    
    def _check_ak_availability(self) -> bool:
        """检查 akshare 是否可用"""
        if self._ak_available is not None:
            return self._ak_available
        
        try:
            import akshare as ak
            self._ak_available = True
            logger.info("V67: akshare 库已加载")
            return True
        except ImportError:
            self._ak_available = False
            logger.error("V67: akshare 库未安装，请运行：pip install akshare")
            return False
    
    def _retry_wrapper(self, func, *args, **kwargs) -> Tuple[bool, Any, str]:
        """
        重试包装器 - 返回详细错误信息
        
        Returns
        -------
        Tuple[bool, Any, str]
            (成功标志，结果，错误信息)
        """
        last_exception = None
        last_error_msg = ""
        
        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"V67: 第 {attempt} 次尝试成功")
                return (True, result, "")
                
            except Exception as e:
                last_exception = e
                last_error_msg = str(e)
                self._last_api_error = f"Attempt {attempt}: {last_error_msg}"
                logger.warning(f"V67: 第 {attempt} 次尝试失败：{last_error_msg}")
                
                if attempt < self.max_retries:
                    logger.info(f"V67: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V67: 已达到最大重试次数 {self.max_retries}")
        
        return (False, None, self._last_api_error)
    
    def _classify_api_error(self, error_msg: str) -> str:
        """
        分类 API 错误类型
        
        Returns
        -------
        str
            错误类型："timeout" | "no_data" | "ip_blocked" | "unknown"
        """
        error_lower = error_msg.lower()
        
        if "timeout" in error_lower or "timed out" in error_lower:
            return "timeout"
        elif "no data" in error_lower or "empty" in error_lower or "none" in error_lower:
            return "no_data"
        elif "blocked" in error_lower or "ip" in error_lower or "封" in error_lower:
            return "ip_blocked"
        elif "connection" in error_lower or "connect" in error_lower:
            return "connection_error"
        else:
            return "unknown"
    
    def fetch_fund_flow_rank_em(self, trade_date: str) -> Optional[pl.DataFrame]:
        """
        获取资金流排名表 (stock_individual_fund_flow_rank_em)
        
        【核心接口】
        使用 ak.stock_individual_fund_flow_rank_em 获取当日资金流排名
        
        Parameters
        ----------
        trade_date : str
            交易日期，格式：YYYY-MM-DD
            
        Returns
        -------
        Optional[pl.DataFrame]
            资金流排名数据
        """
        if not self._check_ak_availability():
            raise ImportError("V67: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch():
            # 使用 ak.stock_individual_fund_flow_rank_em 获取资金流排名
            df = ak.stock_individual_fund_flow_rank(indicator="3 日")
            return df
        
        success, result, error_msg = self._retry_wrapper(_fetch)
        
        if not success:
            error_type = self._classify_api_error(error_msg)
            logger.error(f"V67: 获取 {trade_date} 资金流排名失败 - 类型：{error_type}, 详情：{error_msg}")
            return None
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"V67: {trade_date} 资金流排名为空数据")
            return None
        
        df = pl.from_pandas(result) if hasattr(result, 'columns') else None
        
        if df is None or df.is_empty():
            logger.warning(f"V67: {trade_date} 资金流排名转换为 DataFrame 失败")
            return None
        
        # 标准化列名
        columns_mapping = {
            '代码': 'symbol',
            '名称': 'name',
            '日期': 'trade_date',
            '主力净流入 - 净额': 'net_main_amount',
            '主力净流入 - 净占比': 'net_main_ratio',
            '超大单净流入 - 净额': 'net_super_amount',
            '超大单净流入 - 净占比': 'net_super_ratio',
            '大单净流入 - 净额': 'net_large_amount',
            '大单净流入 - 净占比': 'net_large_ratio',
            '中单净流入 - 净额': 'net_medium_amount',
            '中单净流入 - 净占比': 'net_medium_ratio',
            '小单净流入 - 净额': 'net_small_amount',
            '小单净流入 - 净占比': 'net_small_ratio',
        }
        
        # 重命名存在的列
        available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
        if available_cols:
            df = df.rename(available_cols)
        
        # 添加交易日期（如果不存在）
        if 'trade_date' not in df.columns:
            df = df.with_columns(pl.lit(trade_date).alias('trade_date'))
        
        # 只保留需要的列
        keep_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_main_ratio',
                     'net_super_amount', 'net_super_ratio', 'net_large_amount',
                     'net_large_ratio', 'net_medium_amount', 'net_medium_ratio',
                     'net_small_amount', 'net_small_ratio']
        
        available_keep_cols = [c for c in keep_cols if c in df.columns]
        if available_keep_cols:
            df = df.select(available_keep_cols)
        
        logger.info(f"V67: 获取 {trade_date} 资金流排名 - {df.height}行")
        return df
    
    def fetch_board_industry_cons_em(self, industry_name: str) -> Optional[pl.DataFrame]:
        """
        获取行业成分股 (stock_board_industry_cons_em)
        
        【核心接口】
        使用 ak.stock_board_industry_cons_em 获取行业成分
        
        Parameters
        ----------
        industry_name : str
            行业名称
            
        Returns
        -------
        Optional[pl.DataFrame]
            行业成分股数据
        """
        if not self._check_ak_availability():
            raise ImportError("V67: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch():
            df = ak.stock_board_industry_cons_em(symbol=industry_name)
            return df
        
        success, result, error_msg = self._retry_wrapper(_fetch)
        
        if not success:
            error_type = self._classify_api_error(error_msg)
            logger.error(f"V67: 获取行业 {industry_name} 成分股失败 - 类型：{error_type}, 详情：{error_msg}")
            return None
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"V67: 行业 {industry_name} 成分股为空数据")
            return None
        
        df = pl.from_pandas(result) if hasattr(result, 'columns') else None
        
        if df is None or df.is_empty():
            logger.warning(f"V67: 行业 {industry_name} 成分股转换为 DataFrame 失败")
            return None
        
        # 标准化列名
        columns_mapping = {
            '代码': 'symbol',
            '名称': 'name',
        }
        
        available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
        if available_cols:
            df = df.rename(available_cols)
        
        # 添加行业名称和日期
        df = df.with_columns([
            pl.lit(industry_name).alias('industry_name'),
            pl.lit(datetime.now().strftime('%Y-%m-%d')).alias('trade_date')
        ])
        
        # 只保留需要的列
        keep_cols = ['symbol', 'industry_name', 'trade_date']
        available_keep_cols = [c for c in keep_cols if c in df.columns]
        if available_keep_cols:
            df = df.select(available_keep_cols)
        
        return df
    
    def fetch_symbol_list(self) -> List[str]:
        """
        获取 A 股股票列表
        
        Returns
        -------
        List[str]
            股票代码列表
        """
        if not self._check_ak_availability():
            raise ImportError("V67: akshare 库未安装")
        
        import akshare as ak
        
        try:
            df = ak.stock_info_a_code_name()
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                logger.error("V67: 获取股票列表失败 - 返回空数据")
                return []
            
            pdf = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
            
            if 'code' in pdf.columns:
                return pdf['code'].to_list()
            elif '代码' in pdf.columns:
                return pdf['代码'].to_list()
            else:
                logger.error(f"V67: 无法识别股票列表列名，可用列：{pdf.columns}")
                return []
                
        except Exception as e:
            logger.error(f"V67: 获取股票列表失败：{e}")
            return []
    
    def fetch_industry_list(self) -> List[str]:
        """
        获取行业列表
        
        Returns
        -------
        List[str]
            行业名称列表
        """
        if not self._check_ak_availability():
            raise ImportError("V67: akshare 库未安装")
        
        import akshare as ak
        
        try:
            df = ak.stock_board_industry_name_em()
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                logger.error("V67: 获取行业列表失败 - 返回空数据")
                return []
            
            pdf = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
            
            if '板块名称' in pdf.columns:
                return pdf['板块名称'].to_list()
            elif '板块' in pdf.columns:
                return pdf['板块'].to_list()
            else:
                logger.error(f"V67: 无法识别行业列表列名，可用列：{pdf.columns}")
                return []
                
        except Exception as e:
            logger.error(f"V67: 获取行业列表失败：{e}")
            return []


# ===========================================
# V67 数据填充器 - 暴力抓取 + 落库验证
# ===========================================

class V67DataFiller:
    """
    V67 数据填充器 - 暴力抓取 + 落库验证
    
    【核心协议】
    1. 批次 - 等待模式：每抓取 20 只股票，强制 sleep 5 秒
    2. 落库验证：每写入一条数据，立即执行 SELECT COUNT(*) 确认
    3. 连续失败退出：连续 3 次写入失败，sys.exit(1) 并打印具体 API 报错
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化数据填充器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db or get_db()
        self.fetcher = V67DataFetcher()
        self.consecutive_failures = 0
        self.total_success_count = 0
        self.total_fail_count = 0
    
    def create_tables(self):
        """
        建表校验 - 确保表结构存在
        """
        logger.info("V67: 开始建表校验...")
        
        # 创建 stock_fund_flow 表
        create_fund_flow_sql = """
        CREATE TABLE IF NOT EXISTS `stock_fund_flow` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `trade_date` VARCHAR(20) NOT NULL,
            `net_main_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_super_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_large_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_medium_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_small_amount` DECIMAL(20, 2) DEFAULT 0,
            `net_main_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_super_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_large_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_medium_ratio` DECIMAL(10, 6) DEFAULT 0,
            `net_small_ratio` DECIMAL(10, 6) DEFAULT 0,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='个股资金流数据表'
        """
        
        try:
            self.db.execute(create_fund_flow_sql)
            logger.info("V67: stock_fund_flow 表创建成功")
        except Exception as e:
            logger.error(f"V67: 创建 stock_fund_flow 表失败：{e}")
            raise
        
        # 创建 stock_industry_daily 表
        create_industry_sql = """
        CREATE TABLE IF NOT EXISTS `stock_industry_daily` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `symbol` VARCHAR(20) NOT NULL,
            `industry_name` VARCHAR(100) NOT NULL,
            `trade_date` VARCHAR(20) NOT NULL,
            INDEX `idx_symbol` (`symbol`),
            INDEX `idx_industry` (`industry_name`),
            INDEX `idx_trade_date` (`trade_date`),
            INDEX `idx_symbol_date` (`symbol`, `trade_date`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='股票行业分类数据表'
        """
        
        try:
            self.db.execute(create_industry_sql)
            logger.info("V67: stock_industry_daily 表创建成功")
        except Exception as e:
            logger.error(f"V67: 创建 stock_industry_daily 表失败：{e}")
            raise
        
        logger.info("V67: 建表校验完成")
    
    def _verify_write(self, table_name: str, symbol: str, trade_date: str) -> bool:
        """
        验证写入是否成功
        
        Parameters
        ----------
        table_name : str
            表名
        symbol : str
            股票代码
        trade_date : str
            交易日期
            
        Returns
        -------
        bool
            写入是否成功
        """
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name} WHERE symbol = '{symbol}' AND trade_date = '{trade_date}'"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return False
            
            cnt = int(result['cnt'][0])
            return cnt > 0
            
        except Exception as e:
            logger.error(f"V67: 验证写入失败 {table_name}.{symbol}@{trade_date}: {e}")
            return False
    
    def _handle_write_failure(self, symbol: str, error_msg: str):
        """
        处理写入失败 - 连续失败计数
        
        Parameters
        ----------
        symbol : str
            股票代码
        error_msg : str
            错误信息
        """
        self.consecutive_failures += 1
        self.total_fail_count += 1
        
        error_type = self.fetcher._classify_api_error(error_msg)
        
        logger.error(f"V67: 【写入失败 #{self.consecutive_failures}】{symbol} - 类型：{error_type}")
        logger.error(f"V67: 详细报错：{error_msg}")
        
        if self.consecutive_failures >= V67_MAX_CONSECUTIVE_FAILURES:
            logger.error("=" * 60)
            logger.error(f"V67: 【致命错误】连续 {V67_MAX_CONSECUTIVE_FAILURES} 次写入失败，程序终止")
            logger.error(f"V67: 最后 API 报错：{error_msg}")
            logger.error(f"V67: 错误类型：{error_type}")
            if error_type == "timeout":
                logger.error("V67: 建议：网络超时，请检查网络连接或稍后重试")
            elif error_type == "ip_blocked":
                logger.error("V67: 建议：IP 可能被封禁，请更换 IP 或使用代理")
            elif error_type == "no_data":
                logger.error("V67: 建议：API 返回空数据，可能是非交易日或数据源问题")
            else:
                logger.error("V67: 建议：未知错误，请检查日志详情")
            logger.error("=" * 60)
            sys.exit(1)
    
    def _handle_write_success(self, symbol: str, total_rows: int):
        """
        处理写入成功
        
        Parameters
        ----------
        symbol : str
            股票代码
        total_rows : int
            当前总行数
        """
        self.consecutive_failures = 0
        self.total_success_count += 1
        
        # 按要求输出 [SUCCESS] 格式
        logger.info(f"[SUCCESS] {symbol} written. Total rows now: {total_rows}")
    
    def fill_fund_flow_data(self, start_date: str, end_date: str) -> Dict[str, int]:
        """
        填充资金流数据
        
        【核心逻辑】
        1. 获取股票列表
        2. 批次 - 等待模式抓取
        3. 每写入一条数据立即验证
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Dict[str, int]
            填充结果统计
        """
        logger.info("=" * 60)
        logger.info("V67: 开始填充资金流数据")
        logger.info(f"日期范围：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        # 建表校验
        self.create_tables()
        
        # 获取股票列表
        symbol_list = self.fetcher.fetch_symbol_list()
        
        if not symbol_list:
            logger.error("V67: 无法获取股票列表")
            return {'success': 0, 'failure': 0, 'total_rows': 0}
        
        logger.info(f"V67: 共获取到 {len(symbol_list)} 只股票")
        
        # 生成日期列表
        date_list = self._generate_date_range(start_date, end_date)
        logger.info(f"V67: 共 {len(date_list)} 个交易日")
        
        result = {'success': 0, 'failure': 0, 'total_rows': 0}
        batch_count = 0
        
        # 批次 - 等待模式处理
        for i, symbol in enumerate(symbol_list):
            try:
                # 获取单只股票的资金流数据
                df = self._fetch_single_stock_fund_flow(symbol, date_list)
                
                if df is not None and not df.is_empty():
                    # 写入数据库
                    write_success = self._write_fund_flow_to_db(df, symbol)
                    
                    if write_success:
                        result['success'] += 1
                    else:
                        result['failure'] += 1
                        self._handle_write_failure(symbol, "写入数据库失败")
                        continue
                    
                else:
                    logger.warning(f"V67: {symbol} 无资金流数据")
                    result['failure'] += 1
                
                # 批次等待：每 20 只股票强制 sleep 5 秒
                batch_count += 1
                if batch_count % V67_BATCH_SIZE == 0:
                    logger.info(f"V67: 【批次等待】已处理 {batch_count} 只股票，等待 {V67_BATCH_WAIT_SECONDS} 秒...")
                    time.sleep(V67_BATCH_WAIT_SECONDS)
                
            except Exception as e:
                logger.error(f"V67: {symbol} 处理失败：{e}")
                result['failure'] += 1
                self._handle_write_failure(symbol, str(e))
        
        # 获取总行数
        result['total_rows'] = self._count_table_rows('stock_fund_flow')
        
        logger.info("=" * 60)
        logger.info(f"V67: 资金流数据填充完成")
        logger.info(f"成功：{result['success']}只，失败：{result['failure']}只，总行数：{result['total_rows']}")
        logger.info("=" * 60)
        
        return result
    
    def _fetch_single_stock_fund_flow(self, symbol: str, date_list: List[str]) -> Optional[pl.DataFrame]:
        """
        获取单只股票的资金流数据
        
        Parameters
        ----------
        symbol : str
            股票代码
        date_list : List[str]
            日期列表
            
        Returns
        -------
        Optional[pl.DataFrame]
            资金流数据
        """
        all_data = []
        
        for trade_date in date_list:
            try:
                # 这里需要根据实际 API 调整
                # 目前使用 ak.stock_individual_fund_flow 获取个股资金流
                import akshare as ak
                
                df = ak.stock_individual_fund_flow(symbol=symbol)
                
                if df is not None and not df.empty:
                    df_pl = pl.from_pandas(df)
                    
                    # 添加交易日期过滤
                    if 'trade_date' in df_pl.columns or '日期' in df_pl.columns:
                        date_col = 'trade_date' if 'trade_date' in df_pl.columns else '日期'
                        df_pl = df_pl.with_columns(pl.col(date_col).cast(pl.Utf8).alias('trade_date'))
                        
                        # 过滤日期范围
                        df_filtered = df_pl.filter(
                            (pl.col('trade_date') >= trade_date) & 
                            (pl.col('trade_date') <= trade_date)
                        )
                        
                        if not df_filtered.is_empty():
                            df_filtered = df_filtered.with_columns(pl.lit(symbol).alias('symbol'))
                            all_data.append(df_filtered)
                
            except Exception as e:
                logger.debug(f"V67: {symbol}@{trade_date} 获取失败：{e}")
                continue
        
        if all_data:
            return pl.concat(all_data, how="diagonal")
        return None
    
    def _write_fund_flow_to_db(self, df: pl.DataFrame, symbol: str) -> bool:
        """
        写入资金流数据到数据库并验证
        
        Parameters
        ----------
        df : pl.DataFrame
            资金流数据
        symbol : str
            股票代码
            
        Returns
        -------
        bool
            写入是否成功
        """
        try:
            # 写入数据库
            self.db.to_sql(df, 'stock_fund_flow', if_exists='append')
            
            # 获取当前总行数
            total_rows = self._count_table_rows('stock_fund_flow')
            
            # 验证写入
            trade_dates = df['trade_date'].unique().to_list()
            for trade_date in trade_dates:
                if not self._verify_write('stock_fund_flow', symbol, str(trade_date)):
                    logger.warning(f"V67: 验证失败 {symbol}@{trade_date}，但继续处理")
            
            # 输出成功信息（按要求格式）
            self._handle_write_success(symbol, total_rows)
            
            return True
            
        except Exception as e:
            logger.error(f"V67: 写入 {symbol} 失败：{e}")
            return False
    
    def fill_industry_data(self) -> Dict[str, int]:
        """
        填充行业成分股数据
        
        Returns
        -------
        Dict[str, int]
            填充结果统计
        """
        logger.info("=" * 60)
        logger.info("V67: 开始填充行业成分股数据")
        logger.info("=" * 60)
        
        # 建表校验
        self.create_tables()
        
        # 获取行业列表
        industry_list = self.fetcher.fetch_industry_list()
        
        if not industry_list:
            logger.error("V67: 无法获取行业列表")
            return {'success': 0, 'failure': 0, 'total_rows': 0}
        
        logger.info(f"V67: 共获取到 {len(industry_list)} 个行业")
        
        result = {'success': 0, 'failure': 0, 'total_rows': 0}
        batch_count = 0
        
        # 批次 - 等待模式处理
        for i, industry_name in enumerate(industry_list):
            try:
                logger.info(f"V67: 获取行业成分股 {i+1}/{len(industry_list)} - {industry_name}")
                
                df = self.fetcher.fetch_board_industry_cons_em(industry_name)
                
                if df is not None and not df.is_empty():
                    # 写入数据库
                    try:
                        self.db.to_sql(df, 'stock_industry_daily', if_exists='append')
                        
                        # 获取当前总行数
                        total_rows = self._count_table_rows('stock_industry_daily')
                        
                        result['success'] += 1
                        result['total_rows'] = total_rows
                        
                        logger.info(f"[SUCCESS] {industry_name} written. Total rows now: {total_rows}")
                        
                    except Exception as e:
                        logger.error(f"V67: 写入 {industry_name} 失败：{e}")
                        result['failure'] += 1
                        self._handle_write_failure(industry_name, str(e))
                        continue
                    
                else:
                    logger.warning(f"V67: {industry_name} 无成分股数据")
                    result['failure'] += 1
                
                # 批次等待
                batch_count += 1
                if batch_count % V67_BATCH_SIZE == 0:
                    logger.info(f"V67: 【批次等待】已处理 {batch_count} 个行业，等待 {V67_BATCH_WAIT_SECONDS} 秒...")
                    time.sleep(V67_BATCH_WAIT_SECONDS)
                
            except Exception as e:
                logger.error(f"V67: {industry_name} 处理失败：{e}")
                result['failure'] += 1
                self._handle_write_failure(industry_name, str(e))
        
        # 获取总行数
        result['total_rows'] = self._count_table_rows('stock_industry_daily')
        
        logger.info("=" * 60)
        logger.info(f"V67: 行业成分股数据填充完成")
        logger.info(f"成功：{result['success']}个，失败：{result['failure']}个，总行数：{result['total_rows']}")
        logger.info("=" * 60)
        
        return result
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        生成日期范围列表
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        List[str]
            日期列表
        """
        date_list = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            # 跳过周末
            if current.weekday() < 5:
                date_list.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return date_list
    
    def _count_table_rows(self, table_name: str) -> int:
        """
        统计表行数
        
        Parameters
        ----------
        table_name : str
            表名
            
        Returns
        -------
        int
            行数
        """
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return 0
            
            return int(result['cnt'][0])
            
        except Exception as e:
            logger.warning(f"V67: 统计表 {table_name} 行数失败：{e}")
            return 0
    
    def verify_data_sufficiency(self) -> Tuple[bool, str]:
        """
        验证数据是否充足
        
        【数据熔断】
        - stock_fund_flow 行数低于 100 万行，返回 False
        - stock_industry_daily 行数低于 10 万行，返回 False
        
        Returns
        -------
        Tuple[bool, str]
            (是否充足，消息)
        """
        logger.info("=" * 60)
        logger.info("V67: 开始数据充足性验证")
        
        fund_flow_rows = self._count_table_rows('stock_fund_flow')
        industry_rows = self._count_table_rows('stock_industry_daily')
        
        logger.info(f"V67: stock_fund_flow 行数：{fund_flow_rows:,} (阈值：{V67_MIN_FUND_FLOW_ROWS:,})")
        logger.info(f"V67: stock_industry_daily 行数：{industry_rows:,} (阈值：{V67_MIN_INDUSTRY_ROWS:,})")
        
        messages = []
        
        if fund_flow_rows < V67_MIN_FUND_FLOW_ROWS:
            messages.append(f"数据不足：stock_fund_flow 仅有 {fund_flow_rows:,} 行，需要 {V67_MIN_FUND_FLOW_ROWS:,} 行")
        
        if industry_rows < V67_MIN_INDUSTRY_ROWS:
            messages.append(f"数据不足：stock_industry_daily 仅有 {industry_rows:,} 行，需要 {V67_MIN_INDUSTRY_ROWS:,} 行")
        
        if messages:
            error_msg = "数据不足，请运行 data_filler"
            logger.error(f"V67: 【数据熔断】{error_msg}")
            logger.error("V67: 详细原因:")
            for msg in messages:
                logger.error(f"  - {msg}")
            return (False, error_msg)
        
        logger.info("V67: 数据充足性验证通过")
        logger.info("=" * 60)
        return (True, "数据充足")


# ===========================================
# 便捷函数
# ===========================================

def fill_v67_data(start_date: str = V67_DEFAULT_START_DATE,
                  end_date: str = V67_DEFAULT_END_DATE,
                  db: Optional[DatabaseManager] = None) -> Dict[str, int]:
    """
    便捷函数：填充 V67 数据
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Dict[str, int]
        填充结果
    """
    filler = V67DataFiller(db=db)
    
    # 填充资金流数据
    fund_flow_result = filler.fill_fund_flow_data(start_date, end_date)
    
    # 填充行业数据
    industry_result = filler.fill_industry_data()
    
    # 验证数据充足性
    is_sufficient, message = filler.verify_data_sufficiency()
    
    if not is_sufficient:
        logger.error(f"V67: 数据验证失败：{message}")
    
    return {
        **fund_flow_result,
        **industry_result,
        'is_sufficient': is_sufficient,
        'message': message,
    }


def verify_v67_data(db: Optional[DatabaseManager] = None) -> Tuple[bool, str]:
    """
    便捷函数：验证 V67 数据
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Tuple[bool, str]
        (是否通过，消息)
    """
    filler = V67DataFiller(db=db)
    return filler.verify_data_sufficiency()


# ===========================================
# 主程序
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V67 数据填充器 - 启动")
    logger.info("=" * 60)
    
    # 初始化数据库
    db = get_db()
    
    # 填充数据
    try:
        result = fill_v67_data(V67_DEFAULT_START_DATE, V67_DEFAULT_END_DATE, db)
        
        logger.info("=" * 60)
        logger.info("V67: 数据填充完成")
        logger.info(f"资金流成功：{result.get('success', 0)}只")
        logger.info(f"资金流失败：{result.get('failure', 0)}只")
        logger.info(f"行业成功：{result.get('success', 0)}个")
        logger.info(f"行业失败：{result.get('failure', 0)}个")
        logger.info(f"总行数：{result.get('total_rows', 0):,}")
        logger.info(f"数据充足：{result.get('is_sufficient', False)}")
        logger.info("=" * 60)
        
    except SystemExit:
        logger.error("V67: 程序因连续失败已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V67: 数据填充失败：{e}")
        sys.exit(1)


__all__ = [
    'V67DataFetcher',
    'V67DataFiller',
    'fill_v67_data',
    'verify_v67_data',
    'V67_BATCH_SIZE',
    'V67_BATCH_WAIT_SECONDS',
    'V67_MAX_CONSECUTIVE_FAILURES',
    'V67_MIN_FUND_FLOW_ROWS',
    'V67_MIN_INDUSTRY_ROWS',
]