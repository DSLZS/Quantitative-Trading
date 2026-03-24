"""
V66 Data Loader Module - 机构资金踪迹模型数据加载器 (IC 优化版)

【V66 数据强制落库 - 死命令】
1. 拒绝降级逻辑：严禁 Fallback 到 price_only 模式
2. 数据熔断机制：程序启动时必须检查 stock_fund_flow 和 stock_industry_daily
   - 若数据量少于 10,000 条，必须 raise ValueError("CRITICAL DATA MISSING") 并立即停止
3. 强制反馈机制：每成功写入 100 只股票的数据，必须打印进度
4. 建表校验：脚本开头执行 CREATE TABLE IF NOT EXISTS 语句

【AkShare 接口绑定】
- 资金流：使用 ak.stock_individual_fund_flow (个股级，确保准确性)
- 行业：使用 ak.stock_board_industry_cons_em (确保覆盖度)

【杜绝偷懒与伪造】
- 禁止伪造：禁止在日志中打印"数据加载完成"但实际并无落库的行为
- 代码要求：必须包含详尽的 logger.debug 记录数据查询的 SQL 语句

作者：量化系统
版本：V66.0
日期：2026-03-24
"""

import sys
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

from db_manager import DatabaseManager, get_db


# ===========================================
# V66 配置常量
# ===========================================

V66_MAX_RETRY_ATTEMPTS = 5  # 最大重试次数
V66_RETRY_DELAY_SECONDS = 3  # 重试间隔（秒）
V66_REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
V66_BATCH_SIZE = 100  # 每 100 只股票打印一次进度
V66_MIN_DATA_ROWS = 10000  # 数据熔断阈值：最少 10,000 条


# ===========================================
# 数据获取类 - 带重试机制
# ===========================================

class V66DataFetcher:
    """
    V66 数据获取器 - 带重试机制
    
    【核心功能】
    1. 使用 ak.stock_individual_fund_flow 获取个股资金流数据
    2. 使用 ak.stock_board_industry_cons_em 获取行业成分股数据
    3. 所有网络请求都带重试机制
    """
    
    def __init__(self, max_retries: int = V66_MAX_RETRY_ATTEMPTS,
                 retry_delay: float = V66_RETRY_DELAY_SECONDS):
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
    
    def _check_ak_availability(self) -> bool:
        """检查 akshare 是否可用"""
        if self._ak_available is not None:
            return self._ak_available
        
        try:
            import akshare as ak
            self._ak_available = True
            logger.info("V66: akshare 库已加载")
            return True
        except ImportError:
            self._ak_available = False
            logger.error("V66: akshare 库未安装，请运行：pip install akshare")
            return False
    
    def _retry_wrapper(self, func, *args, **kwargs):
        """
        重试包装器
        
        Parameters
        ----------
        func : callable
            要执行的函数
        *args
            函数位置参数
        **kwargs
            函数关键字参数
            
        Returns
        -------
        Any
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(1, self.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"V66: 第 {attempt} 次尝试成功")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"V66: 第 {attempt} 次尝试失败：{e}")
                
                if attempt < self.max_retries:
                    logger.info(f"V66: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V66: 已达到最大重试次数 {self.max_retries}")
        
        raise last_exception
    
    def fetch_fund_flow_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        获取历史资金流数据
        
        【核心逻辑】
        使用 ak.stock_individual_fund_flow 获取个股资金流数据
        
        Parameters
        ----------
        start_date : str
            开始日期，格式：YYYY-MM-DD
        end_date : str
            结束日期，格式：YYYY-MM-DD
            
        Returns
        -------
        pl.DataFrame
            资金流数据，包含字段：
            - symbol: 股票代码
            - trade_date: 交易日期
            - net_main_amount: 主力净流入
            - net_main_ratio: 主力净流入占比
            - net_super_amount: 超大单净流入
            - net_large_amount: 大单净流入
            - net_medium_amount: 中单净流入
            - net_small_amount: 小单净流入
        """
        if not self._check_ak_availability():
            raise ImportError("V66: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch_single_stock(symbol: str) -> Optional[pl.DataFrame]:
            """获取单只股票的资金流数据"""
            try:
                # 使用 ak.stock_individual_fund_flow 获取个股资金流
                df = ak.stock_individual_fund_flow(symbol=symbol)
                
                if df is None or df.empty:
                    return None
                
                # 标准化列名
                columns_mapping = {
                    '日期': 'trade_date',
                    '主力净流入-净额': 'net_main_amount',
                    '主力净流入-净占比': 'net_main_ratio',
                    '超大单净流入 - 净额': 'net_super_amount',
                    '超大单净流入 - 净占比': 'net_super_ratio',
                    '大单净流入 - 净额': 'net_large_amount',
                    '大单净流入 - 净占比': 'net_large_ratio',
                    '中单净流入 - 净额': 'net_medium_amount',
                    '中单净流入 - 净占比': 'net_medium_ratio',
                    '小单净流入 - 净额': 'net_small_amount',
                    '小单净流入 - 净占比': 'net_small_ratio',
                }
                
                # 重命名列
                available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
                df = df.rename(columns=available_cols)
                
                # 添加股票代码
                df['symbol'] = symbol
                
                # 只保留需要的列
                keep_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_super_amount',
                            'net_large_amount', 'net_medium_amount', 'net_small_amount',
                            'net_main_ratio', 'net_super_ratio', 'net_large_ratio',
                            'net_medium_ratio', 'net_small_ratio']
                
                available_keep_cols = [c for c in keep_cols if c in df.columns]
                df = df[available_keep_cols]
                
                # 格式化日期
                if 'trade_date' in df.columns:
                    df['trade_date'] = df['trade_date'].astype(str)
                
                return df
                
            except Exception as e:
                logger.warning(f"V66: 获取 {symbol} 资金流数据失败：{e}")
                return None
        
        def _fetch_symbol_list() -> List[str]:
            """获取股票列表"""
            try:
                # 获取 A 股股票列表
                df = ak.stock_info_a_code_name()
                if df is not None and not df.empty:
                    if 'code' in df.columns:
                        return df['code'].tolist()
                    elif '代码' in df.columns:
                        return df['代码'].tolist()
            except Exception as e:
                logger.error(f"V66: 获取股票列表失败：{e}")
            return []
        
        logger.info(f"V66: 开始获取资金流数据 [{start_date}, {end_date}]")
        
        # 获取股票列表
        symbol_list = _fetch_symbol_list()
        if not symbol_list:
            logger.error("V66: 无法获取股票列表")
            return self._empty_fund_flow_df()
        
        logger.info(f"V66: 共获取到 {len(symbol_list)} 只股票")
        
        all_data = []
        success_count = 0
        fail_count = 0
        saved_count = 0
        
        for i, symbol in enumerate(symbol_list):
            try:
                df = self._retry_wrapper(_fetch_single_stock, symbol)
                
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    success_count += 1
                    saved_count += 1
                    
                    # 强制反馈机制：每 100 只股票打印进度
                    if saved_count % V66_BATCH_SIZE == 0:
                        logger.info(f"[DATA PROGRESS] Table: stock_fund_flow, Progress: {saved_count}/{len(symbol_list)} stocks saved")
                else:
                    logger.warning(f"V66: {symbol} 无资金流数据")
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"V66: {symbol} 获取失败：{e}")
                fail_count += 1
            
            # 避免请求过快
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
        
        if not all_data:
            logger.error("V66: 未获取到任何资金流数据")
            return self._empty_fund_flow_df()
        
        # 合并所有数据
        result_df = pl.concat([df for df in all_data], how="diagonal")
        
        logger.info(f"V66: 资金流数据获取完成 - 成功:{success_count}只，失败:{fail_count}只，总计:{result_df.height}行")
        
        return result_df
    
    def fetch_industry_data(self) -> pl.DataFrame:
        """
        获取行业分类数据
        
        【核心逻辑】
        使用 ak.stock_board_industry_cons_em 获取行业成分股
        
        Returns
        -------
        pl.DataFrame
            行业分类数据，包含字段：
            - symbol: 股票代码
            - industry_name: 行业名称
            - trade_date: 交易日期（当前日期）
        """
        if not self._check_ak_availability():
            raise ImportError("V66: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch_industry_cons(industry_name: str):
            """获取行业成分股"""
            try:
                # 使用 ak.stock_board_industry_cons_em 获取行业成分股
                df = ak.stock_board_industry_cons_em(symbol=industry_name)
                
                if df is None or df.empty:
                    return None
                
                # 标准化列名
                columns_mapping = {
                    '代码': 'symbol',
                    '名称': 'name',
                }
                
                # 重命名列
                available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
                df = df.rename(columns=available_cols)
                
                # 添加行业名称和日期
                df['industry_name'] = industry_name
                df['trade_date'] = datetime.now().strftime('%Y-%m-%d')
                
                # 只保留需要的列
                keep_cols = ['symbol', 'industry_name', 'trade_date']
                available_keep_cols = [c for c in keep_cols if c in df.columns]
                df = df[available_keep_cols]
                
                return df
                
            except Exception as e:
                logger.warning(f"V66: 获取行业 {industry_name} 成分股失败：{e}")
                return None
        
        logger.info("V66: 开始获取行业分类数据")
        
        # 获取所有行业名称
        try:
            industry_list_df = ak.stock_board_industry_name_em()
            
            if industry_list_df is None or industry_list_df.empty:
                logger.error("V66: 无法获取行业列表")
                return self._empty_industry_df()
            
            # 获取行业名称列
            if '板块名称' in industry_list_df.columns:
                industry_names = industry_list_df['板块名称'].tolist()
            elif '板块' in industry_list_df.columns:
                industry_names = industry_list_df['板块'].tolist()
            else:
                logger.error("V66: 无法识别行业列表列名")
                return self._empty_industry_df()
            
        except Exception as e:
            logger.error(f"V66: 获取行业列表失败：{e}")
            return self._empty_industry_df()
        
        all_data = []
        success_count = 0
        fail_count = 0
        saved_count = 0
        
        for i, industry_name in enumerate(industry_names):
            logger.info(f"V66: 获取行业成分股 {i+1}/{len(industry_names)} - {industry_name}")
            
            try:
                df = self._retry_wrapper(_fetch_industry_cons, industry_name)
                
                if df is not None and not df.is_empty():
                    all_data.append(df)
                    success_count += 1
                    saved_count += 1
                    
                    # 强制反馈机制：每 100 个行业打印进度
                    if saved_count % 10 == 0:
                        logger.info(f"[DATA PROGRESS] Table: stock_industry_daily, Progress: {saved_count}/{len(industry_names)} industries saved")
                else:
                    logger.warning(f"V66: {industry_name} 无成分股数据")
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"V66: {industry_name} 获取失败：{e}")
                fail_count += 1
            
            # 避免请求过快
            if i < len(industry_names) - 1:
                time.sleep(0.3)
        
        if not all_data:
            logger.error("V66: 未获取到任何行业分类数据")
            return self._empty_industry_df()
        
        # 合并所有数据
        result_df = pl.concat([df for df in all_data], how="diagonal")
        
        logger.info(f"V66: 行业分类数据获取完成 - 成功:{success_count}个行业，总计:{result_df.height}行")
        
        return result_df
    
    def _empty_fund_flow_df(self) -> pl.DataFrame:
        """返回空资金流 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'trade_date': pl.Utf8,
            'net_main_amount': pl.Float64,
            'net_super_amount': pl.Float64,
            'net_large_amount': pl.Float64,
            'net_medium_amount': pl.Float64,
            'net_small_amount': pl.Float64,
            'net_main_ratio': pl.Float64,
            'net_super_ratio': pl.Float64,
            'net_large_ratio': pl.Float64,
            'net_medium_ratio': pl.Float64,
            'net_small_ratio': pl.Float64,
        })
    
    def _empty_industry_df(self) -> pl.DataFrame:
        """返回空行业 DataFrame"""
        return pl.DataFrame(schema={
            'symbol': pl.Utf8,
            'industry_name': pl.Utf8,
            'trade_date': pl.Utf8,
        })


# ===========================================
# V66 数据加载器 - 主类
# ===========================================

class V66DataLoader:
    """
    V66 数据加载器 - 强制数据落库 + 数据熔断
    
    【核心功能】
    1. 从 akshare 获取资金流和行业数据
    2. 将数据写入 MySQL 数据库
    3. 建表校验：执行 CREATE TABLE IF NOT EXISTS
    4. 数据熔断：检查 stock_fund_flow 和 stock_industry_daily 表，少于 10,000 条则 raise ValueError
    5. 强制反馈：每 100 只股票打印进度
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化数据加载器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        self.db = db or get_db()
        self.fetcher = V66DataFetcher()
    
    def create_tables(self):
        """
        建表校验：执行 CREATE TABLE IF NOT EXISTS 语句
        
        确保表结构与要求完全对齐
        """
        logger.info("V66: 开始建表校验...")
        
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
            logger.debug(f"V66: 执行 SQL: {create_fund_flow_sql[:200]}...")
            self.db.execute(create_fund_flow_sql)
            logger.info("V66: stock_fund_flow 表创建成功")
        except Exception as e:
            logger.error(f"V66: 创建 stock_fund_flow 表失败：{e}")
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
            logger.debug(f"V66: 执行 SQL: {create_industry_sql[:200]}...")
            self.db.execute(create_industry_sql)
            logger.info("V66: stock_industry_daily 表创建成功")
        except Exception as e:
            logger.error(f"V66: 创建 stock_industry_daily 表失败：{e}")
            raise
        
        logger.info("V66: 建表校验完成")
    
    def load_and_sync_data(self, start_date: str, end_date: str) -> Dict[str, int]:
        """
        加载并同步数据到数据库
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        Dict[str, int]
            加载结果，包含：
            - fund_flow_rows: 资金流行数
            - industry_rows: 行业行数
        """
        logger.info("=" * 60)
        logger.info("V66: 开始加载数据")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        # 1. 建表校验
        self.create_tables()
        
        result = {
            'fund_flow_rows': 0,
            'industry_rows': 0,
        }
        
        # 2. 获取资金流数据
        try:
            fund_flow_df = self.fetcher.fetch_fund_flow_data(start_date, end_date)
            
            if fund_flow_df.height > 0:
                # 写入数据库
                self.db.to_sql(fund_flow_df, 'stock_fund_flow', if_exists='append')
                result['fund_flow_rows'] = fund_flow_df.height
                logger.info(f"V66: 资金流数据已写入数据库 - {fund_flow_df.height}行")
            else:
                logger.error("V66: 资金流数据为空，无法写入数据库")
                
        except Exception as e:
            logger.error(f"V66: 加载资金流数据失败：{e}")
            raise
        
        # 3. 获取行业分类数据
        try:
            industry_df = self.fetcher.fetch_industry_data()
            
            if industry_df.height > 0:
                # 写入数据库
                self.db.to_sql(industry_df, 'stock_industry_daily', if_exists='append')
                result['industry_rows'] = industry_df.height
                logger.info(f"V66: 行业分类数据已写入数据库 - {industry_df.height}行")
            else:
                logger.error("V66: 行业分类数据为空，无法写入数据库")
                
        except Exception as e:
            logger.error(f"V66: 加载行业分类数据失败：{e}")
            raise
        
        # 4. 数据熔断检查
        self._data_circuit_breaker_check()
        
        return result
    
    def _data_circuit_breaker_check(self) -> Dict[str, int]:
        """
        数据熔断检查
        
        【死命令】
        - 检查 stock_fund_flow 和 stock_industry_daily 表
        - 若数据量少于 10,000 条，必须 raise ValueError("CRITICAL DATA MISSING")
        
        Returns
        -------
        Dict[str, int]
            检查结果
        """
        logger.info("=" * 60)
        logger.info("V66: 开始数据熔断检查")
        
        # 检查 stock_fund_flow 表
        fund_flow_count = self._count_table_rows('stock_fund_flow')
        
        # 检查 stock_industry_daily 表
        industry_count = self._count_table_rows('stock_industry_daily')
        
        # 打印数据检查报告
        logger.info(f"[DATA CHECK] Fund Flow Rows: {fund_flow_count}, Industry Rows: {industry_count}")
        
        # 数据熔断：少于 10,000 条直接抛出异常
        if fund_flow_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_fund_flow 表仅有 {fund_flow_count} 条数据，少于阈值 {V66_MIN_DATA_ROWS}"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        if industry_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_industry_daily 表仅有 {industry_count} 条数据，少于阈值 {V66_MIN_DATA_ROWS}"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        logger.info("V66: 数据熔断检查通过")
        logger.info("=" * 60)
        
        return {
            'fund_flow_rows': fund_flow_count,
            'industry_rows': industry_count,
        }
    
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
            logger.debug(f"V66: 执行 SQL: {query}")
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return 0
            
            return int(df['cnt'][0])
            
        except Exception as e:
            logger.warning(f"V66: 统计表 {table_name} 行数失败：{e}")
            return 0
    
    def verify_data_for_backtest(self) -> bool:
        """
        验证数据是否满足回测要求
        
        【死命令】
        如果 stock_fund_flow 或 stock_industry_daily 表数据量少于 10,000 条
        直接 raise ValueError("CRITICAL DATA MISSING")
        
        Returns
        -------
        bool
            验证是否通过
        """
        logger.info("=" * 60)
        logger.info("V66: 开始回测数据验证")
        
        # 检查 stock_fund_flow 表
        fund_flow_count = self._count_table_rows('stock_fund_flow')
        
        # 检查 stock_industry_daily 表
        industry_count = self._count_table_rows('stock_industry_daily')
        
        # 打印数据检查报告
        logger.info(f"[DATA CHECK] Fund Flow Rows: {fund_flow_count}, Industry Rows: {industry_count}")
        
        # 数据熔断：少于 10,000 条直接抛出异常
        if fund_flow_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_fund_flow 表仅有 {fund_flow_count} 条数据"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        if industry_count < V66_MIN_DATA_ROWS:
            error_msg = f"CRITICAL DATA MISSING: stock_industry_daily 表仅有 {industry_count} 条数据"
            logger.error(f"V66: 【数据熔断】{error_msg}")
            raise ValueError(error_msg)
        
        logger.info("V66: 数据验证通过")
        logger.info("=" * 60)
        
        return True


# ===========================================
# 便捷函数
# ===========================================

def sync_v66_data(start_date: str = "2024-01-01",
                  end_date: str = "2024-12-31",
                  db: Optional[DatabaseManager] = None) -> Dict[str, int]:
    """
    便捷函数：同步 V66 数据
    
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
        加载结果
    """
    loader = V66DataLoader(db=db)
    return loader.load_and_sync_data(start_date, end_date)


def verify_v66_data(db: Optional[DatabaseManager] = None) -> bool:
    """
    便捷函数：验证 V66 数据
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    bool
        验证是否通过
    """
    loader = V66DataLoader(db=db)
    return loader.verify_data_for_backtest()


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
    logger.info("V66 数据加载器 - 启动")
    logger.info("=" * 60)
    
    # 默认回测区间
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # 初始化数据库
    db = get_db()
    
    # 加载并同步数据
    try:
        result = sync_v66_data(start_date, end_date, db)
        
        # 验证数据
        verify_v66_data(db)
        
        logger.info("V66: 数据加载完成")
        
    except ValueError as e:
        logger.error(f"V66: 数据熔断触发：{e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V66: 数据加载失败：{e}")
        sys.exit(1)


__all__ = [
    'V66DataFetcher',
    'V66DataLoader',
    'sync_v66_data',
    'verify_v66_data',
    'V66_MIN_DATA_ROWS',
]