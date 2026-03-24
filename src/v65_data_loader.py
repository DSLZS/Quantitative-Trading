"""
V65 Data Loader Module - 机构资金踪迹模型数据加载器

【V65 数据强制落库 - 死命令】
1. 拒绝自修复：严禁在缺少 stock_fund_flow 和 stock_industry_daily 的情况下启动回测
2. 如果数据为空，程序必须直接 sys.exit(1) 并报错
3. 数据获取脚本必须带重试机制

【核心功能】
1. 使用 ak.stock_individual_fund_flow_rank_em 获取历史资金流
2. 使用 ak.stock_board_industry_cons_em 获取行业分类
3. 自检逻辑：回测前必须查询数据库，统计两个表的行数
4. 在控制台打印：[DATA CHECK] Fund Flow Rows: XXX, Industry Rows: YYY

作者：量化系统
版本：V65.0
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
# V65 配置常量
# ===========================================

V65_MAX_RETRY_ATTEMPTS = 5  # 最大重试次数
V65_RETRY_DELAY_SECONDS = 3  # 重试间隔（秒）
V65_REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
V65_BATCH_SIZE = 500  # 批量写入数据库的大小


# ===========================================
# 数据获取类 - 带重试机制
# ===========================================

class V65DataFetcher:
    """
    V65 数据获取器 - 带重试机制
    
    【核心功能】
    1. 使用 akshare API 获取资金流数据
    2. 使用 akshare API 获取行业分类数据
    3. 所有网络请求都带重试机制
    """
    
    def __init__(self, max_retries: int = V65_MAX_RETRY_ATTEMPTS,
                 retry_delay: float = V65_RETRY_DELAY_SECONDS):
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
            logger.info("V65: akshare 库已加载")
            return True
        except ImportError:
            self._ak_available = False
            logger.error("V65: akshare 库未安装，请运行：pip install akshare")
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
                    logger.info(f"V65: 第 {attempt} 次尝试成功")
                return result
                
            except Exception as e:
                last_exception = e
                logger.warning(f"V65: 第 {attempt} 次尝试失败：{e}")
                
                if attempt < self.max_retries:
                    logger.info(f"V65: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V65: 已达到最大重试次数 {self.max_retries}")
        
        raise last_exception
    
    def fetch_fund_flow_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """
        获取历史资金流数据
        
        【核心逻辑】
        使用 ak.stock_individual_fund_flow_rank_em 获取历史资金流
        
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
            - net_super_amount: 超大单净流入
            - net_large_amount: 大单净流入
            - net_medium_amount: 中单净流入
            - net_small_amount: 小单净流入
        """
        if not self._check_ak_availability():
            raise ImportError("V65: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch_batch(trade_date: str):
            """获取单日资金流数据"""
            try:
                # 使用 stock_individual_fund_flow_rank_em 获取资金流排名数据
                df = ak.stock_individual_fund_flow_rank_em(trade_date=trade_date)
                
                if df is None or df.empty:
                    return None
                
                # 标准化列名
                columns_mapping = {
                    '代码': 'symbol',
                    '名称': 'name',
                    '主力净流入-净额': 'net_main_amount',
                    '主力净流入-净占比': 'net_main_ratio',
                    '超大单净流入-净额': 'net_super_amount',
                    '超大单净流入-净占比': 'net_super_ratio',
                    '大单净流入-净额': 'net_large_amount',
                    '大单净流入-净占比': 'net_large_ratio',
                    '中单净流入-净额': 'net_medium_amount',
                    '中单净流入-净占比': 'net_medium_ratio',
                    '小单净流入-净额': 'net_small_amount',
                    '小单净流入-净占比': 'net_small_ratio',
                }
                
                # 重命名列
                available_cols = {k: v for k, v in columns_mapping.items() if k in df.columns}
                df = df.rename(columns=available_cols)
                
                # 添加交易日期
                df['trade_date'] = trade_date
                
                # 只保留需要的列
                keep_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_super_amount',
                            'net_large_amount', 'net_medium_amount', 'net_small_amount',
                            'net_main_ratio', 'net_super_ratio', 'net_large_ratio',
                            'net_medium_ratio', 'net_small_ratio']
                
                available_keep_cols = [c for c in keep_cols if c in df.columns]
                df = df[available_keep_cols]
                
                return df
                
            except Exception as e:
                logger.warning(f"V65: 获取 {trade_date} 资金流数据失败：{e}")
                return None
        
        logger.info(f"V65: 开始获取资金流数据 [{start_date}, {end_date}]")
        
        # 生成交易日期列表
        trade_dates = self._generate_trade_dates(start_date, end_date)
        
        all_data = []
        success_count = 0
        fail_count = 0
        
        for i, trade_date in enumerate(trade_dates):
            logger.info(f"V65: 获取资金流数据 {i+1}/{len(trade_dates)} - {trade_date}")
            
            try:
                df = self._retry_wrapper(_fetch_batch, trade_date)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    success_count += 1
                else:
                    logger.warning(f"V65: {trade_date} 无资金流数据")
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"V65: {trade_date} 获取失败：{e}")
                fail_count += 1
            
            # 避免请求过快
            if i < len(trade_dates) - 1:
                time.sleep(0.5)
        
        if not all_data:
            logger.error("V65: 未获取到任何资金流数据")
            return self._empty_fund_flow_df()
        
        # 合并所有数据
        result_df = pl.from_pandas(pl.concat([pl.from_pandas(df) for df in all_data], how="diagonal"))
        
        logger.info(f"V65: 资金流数据获取完成 - 成功:{success_count}天，失败:{fail_count}天，总计:{result_df.height}行")
        
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
            raise ImportError("V65: akshare 库未安装")
        
        import akshare as ak
        
        def _fetch_industry_cons(industry_name: str):
            """获取行业成分股"""
            try:
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
                logger.warning(f"V65: 获取行业 {industry_name} 成分股失败：{e}")
                return None
        
        logger.info("V65: 开始获取行业分类数据")
        
        # 获取所有行业名称
        try:
            industry_list_df = ak.stock_board_industry_name_em()
            
            if industry_list_df is None or industry_list_df.empty:
                logger.error("V65: 无法获取行业列表")
                return self._empty_industry_df()
            
            # 获取行业名称列
            if '板块名称' in industry_list_df.columns:
                industry_names = industry_list_df['板块名称'].tolist()
            elif '板块' in industry_list_df.columns:
                industry_names = industry_list_df['板块'].tolist()
            else:
                logger.error("V65: 无法识别行业列表列名")
                return self._empty_industry_df()
            
        except Exception as e:
            logger.error(f"V65: 获取行业列表失败：{e}")
            return self._empty_industry_df()
        
        all_data = []
        success_count = 0
        fail_count = 0
        
        for i, industry_name in enumerate(industry_names):
            logger.info(f"V65: 获取行业成分股 {i+1}/{len(industry_names)} - {industry_name}")
            
            try:
                df = self._retry_wrapper(_fetch_industry_cons, industry_name)
                
                if df is not None and not df.empty:
                    all_data.append(df)
                    success_count += 1
                else:
                    logger.warning(f"V65: {industry_name} 无成分股数据")
                    fail_count += 1
                    
            except Exception as e:
                logger.error(f"V65: {industry_name} 获取失败：{e}")
                fail_count += 1
            
            # 避免请求过快
            if i < len(industry_names) - 1:
                time.sleep(0.5)
        
        if not all_data:
            logger.error("V65: 未获取到任何行业分类数据")
            return self._empty_industry_df()
        
        # 合并所有数据
        result_df = pl.from_pandas(pl.concat([pl.from_pandas(df) for df in all_data], how="diagonal"))
        
        logger.info(f"V65: 行业分类数据获取完成 - 成功:{success_count}个行业，总计:{result_df.height}行")
        
        return result_df
    
    def _generate_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        生成交易日期列表
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
            
        Returns
        -------
        List[str]
            交易日期列表
        """
        # 简单实现：生成所有日期（不区分节假日）
        # 实际使用时应该从中国股市交易日历获取
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        dates = []
        current = start
        
        while current <= end:
            # 排除周末
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
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
# V65 数据加载器 - 主类
# ===========================================

class V65DataLoader:
    """
    V65 数据加载器 - 强制数据落库
    
    【核心功能】
    1. 从 akshare 获取资金流和行业数据
    2. 将数据写入 MySQL 数据库
    3. 数据自检：检查 stock_fund_flow 和 stock_industry_daily 表是否有数据
    4. 打印数据检查报告：[DATA CHECK] Fund Flow Rows: XXX, Industry Rows: YYY
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
        self.fetcher = V65DataFetcher()
    
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
        logger.info("V65: 开始加载数据")
        logger.info(f"回测区间：[{start_date}, {end_date}]")
        logger.info("=" * 60)
        
        result = {
            'fund_flow_rows': 0,
            'industry_rows': 0,
        }
        
        # 1. 获取资金流数据
        try:
            fund_flow_df = self.fetcher.fetch_fund_flow_data(start_date, end_date)
            
            if fund_flow_df.height > 0:
                # 写入数据库
                self.db.to_sql(fund_flow_df, 'stock_fund_flow', if_exists='append')
                result['fund_flow_rows'] = fund_flow_df.height
                logger.info(f"V65: 资金流数据已写入数据库 - {fund_flow_df.height}行")
            else:
                logger.error("V65: 资金流数据为空，无法写入数据库")
                
        except Exception as e:
            logger.error(f"V65: 加载资金流数据失败：{e}")
            raise
        
        # 2. 获取行业分类数据
        try:
            industry_df = self.fetcher.fetch_industry_data()
            
            if industry_df.height > 0:
                # 写入数据库
                self.db.to_sql(industry_df, 'stock_industry_daily', if_exists='append')
                result['industry_rows'] = industry_df.height
                logger.info(f"V65: 行业分类数据已写入数据库 - {industry_df.height}行")
            else:
                logger.error("V65: 行业分类数据为空，无法写入数据库")
                
        except Exception as e:
            logger.error(f"V65: 加载行业分类数据失败：{e}")
            raise
        
        # 3. 数据自检
        self._data_integrity_check()
        
        return result
    
    def _data_integrity_check(self) -> Dict[str, int]:
        """
        数据完整性检查
        
        Returns
        -------
        Dict[str, int]
            检查结果
        """
        logger.info("=" * 60)
        logger.info("V65: 开始数据完整性检查")
        
        # 检查 stock_fund_flow 表
        fund_flow_count = self._count_table_rows('stock_fund_flow')
        
        # 检查 stock_industry_daily 表
        industry_count = self._count_table_rows('stock_industry_daily')
        
        # 打印数据检查报告
        logger.info(f"[DATA CHECK] Fund Flow Rows: {fund_flow_count}, Industry Rows: {industry_count}")
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
            df = self.db.read_sql(query)
            
            if df.is_empty():
                return 0
            
            return int(df['cnt'][0])
            
        except Exception as e:
            logger.warning(f"V65: 统计表 {table_name} 行数失败：{e}")
            return 0
    
    def verify_data_for_backtest(self) -> bool:
        """
        验证数据是否满足回测要求
        
        【死命令】
        如果 stock_fund_flow 或 stock_industry_daily 表为空，直接 sys.exit(1)
        
        Returns
        -------
        bool
            验证是否通过
        """
        logger.info("=" * 60)
        logger.info("V65: 开始回测数据验证")
        
        # 检查 stock_fund_flow 表
        fund_flow_count = self._count_table_rows('stock_fund_flow')
        
        # 检查 stock_industry_daily 表
        industry_count = self._count_table_rows('stock_industry_daily')
        
        # 打印数据检查报告
        logger.info(f"[DATA CHECK] Fund Flow Rows: {fund_flow_count}, Industry Rows: {industry_count}")
        
        # 死命令：数据为空直接退出
        if fund_flow_count == 0:
            logger.error("V65: 【致命错误】stock_fund_flow 表为空，无法启动回测！")
            logger.error("V65: 请先运行数据加载脚本：python src/v65_data_loader.py")
            sys.exit(1)
        
        if industry_count == 0:
            logger.error("V65: 【致命错误】stock_industry_daily 表为空，无法启动回测！")
            logger.error("V65: 请先运行数据加载脚本：python src/v65_data_loader.py")
            sys.exit(1)
        
        logger.info("V65: 数据验证通过")
        logger.info("=" * 60)
        
        return True


# ===========================================
# 便捷函数
# ===========================================

def sync_v65_data(start_date: str = "2024-01-01",
                  end_date: str = "2024-12-31",
                  db: Optional[DatabaseManager] = None) -> Dict[str, int]:
    """
    便捷函数：同步 V65 数据
    
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
    loader = V65DataLoader(db=db)
    return loader.load_and_sync_data(start_date, end_date)


def verify_v65_data(db: Optional[DatabaseManager] = None) -> bool:
    """
    便捷函数：验证 V65 数据
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    bool
        验证是否通过
    """
    loader = V65DataLoader(db=db)
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
    logger.info("V65 数据加载器 - 启动")
    logger.info("=" * 60)
    
    # 默认回测区间
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    # 初始化数据库
    db = get_db()
    
    # 加载并同步数据
    try:
        result = sync_v65_data(start_date, end_date, db)
        
        # 验证数据
        verify_v65_data(db)
        
        logger.info("V65: 数据加载完成")
        
    except Exception as e:
        logger.error(f"V65: 数据加载失败：{e}")
        sys.exit(1)


__all__ = [
    'V65DataFetcher',
    'V65DataLoader',
    'sync_v65_data',
    'verify_v65_data',
]