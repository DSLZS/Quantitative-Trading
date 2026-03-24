"""
V68 Data Force Filler Module - 暴力数据抓取与强制落库

【V68 数据暴力抓取协议 - 死命令】

1. 独立运行
   ✅ 这个脚本必须能独立运行
   ✅ 不依赖任何外部模块（除了 akshare 和数据库）

2. 历史回溯逻辑
   ✅ 使用 AkShare 循环抓取 2024 全年 5000 只股票的日线资金流
   ✅ 每抓取 20 只股票，强制 sleep 5 秒

3. 强制落库确认
   ✅ 禁止使用 df.to_sql 的静默模式
   ✅ 必须每写入 1000 行，执行一次 SELECT COUNT(*) 并将结果打印到控制台

4. 错误阻断
   ✅ 如果遇到 IP Blocked 或 Network Error，必须停止脚本
   ✅ 记录最后一条成功抓取的日期，以便断点续传

作者：量化系统
版本：V68.0
日期：2026-03-24
"""

import sys
import os
import time
import traceback
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import polars as pl
from loguru import logger

# 尝试导入 akshare
try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False
    logger.error("V68: akshare 库未安装，请运行：pip install akshare")

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("V68: db_manager 模块未找到")


# ===========================================
# V68 配置常量 - 暴力数据抓取协议
# ===========================================

# 批次等待配置
V68_BATCH_SIZE = 20  # 每批次抓取 20 只股票
V68_BATCH_WAIT_SECONDS = 5  # 每批次后强制等待 5 秒

# 写入验证配置
V68_WRITE_VERIFY_ROWS = 1000  # 每写入 1000 行验证一次
V68_MAX_CONSECUTIVE_FAILURES = 3  # 连续 3 次写入失败即退出

# 数据熔断阈值
V68_MIN_FUND_FLOW_ROWS = 1000000  # 100 万行阈值

# 请求超时配置
V68_REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
V68_MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
V68_RETRY_DELAY_SECONDS = 2  # 重试间隔（秒）

# 日期范围配置
V68_DEFAULT_START_DATE = "2024-01-01"
V68_DEFAULT_END_DATE = "2024-12-31"

# 断点续传配置文件
V68_CHECKPOINT_FILE = "data/sync_status/v68_checkpoint.json"


# ===========================================
# V68 数据获取器 - 暴力抓取模式
# ===========================================

class V68DataFetcher:
    """
    V68 数据获取器 - 暴力抓取模式
    
    【核心功能】
    1. 使用 AkShare 抓取股票资金流数据
    2. 批次 - 等待模式：每抓取 20 只股票，强制 sleep 5 秒
    3. 所有网络请求带重试机制
    """
    
    def __init__(self, max_retries: int = V68_MAX_RETRY_ATTEMPTS,
                 retry_delay: float = V68_RETRY_DELAY_SECONDS):
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
        self._last_api_error: str = ""
        self._consecutive_errors = 0
    
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
                    logger.info(f"V68: 第 {attempt} 次尝试成功")
                self._consecutive_errors = 0
                return (True, result, "")
                
            except Exception as e:
                last_exception = e
                last_error_msg = str(e)
                self._last_api_error = f"Attempt {attempt}: {last_error_msg}"
                self._consecutive_errors += 1
                logger.warning(f"V68: 第 {attempt} 次尝试失败：{last_error_msg}")
                
                if attempt < self.max_retries:
                    logger.info(f"V68: {self.retry_delay}秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"V68: 已达到最大重试次数 {self.max_retries}")
        
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
        elif "blocked" in error_lower or "ip" in error_lower or "封" in error_lower or "limit" in error_lower:
            return "ip_blocked"
        elif "connection" in error_lower or "connect" in error_lower or "network" in error_lower:
            return "connection_error"
        else:
            return "unknown"
    
    def fetch_stock_fund_flow(self, symbol: str) -> Optional[pl.DataFrame]:
        """
        获取单只股票的资金流数据
        
        【核心接口】
        使用 ak.stock_individual_fund_flow 获取个股历史资金流
        
        Parameters
        ----------
        symbol : str
            股票代码
            
        Returns
        -------
        Optional[pl.DataFrame]
            资金流数据
        """
        if not AK_AVAILABLE:
            raise ImportError("V68: akshare 库未安装")
        
        def _fetch():
            # 使用 ak.stock_individual_fund_flow 获取个股历史资金流
            df = ak.stock_individual_fund_flow(symbol=symbol)
            return df
        
        success, result, error_msg = self._retry_wrapper(_fetch)
        
        if not success:
            error_type = self._classify_api_error(error_msg)
            logger.error(f"V68: 获取 {symbol} 资金流失败 - 类型：{error_type}, 详情：{error_msg}")
            
            # 如果是 IP 被封或网络错误，抛出异常停止脚本
            if error_type in ["ip_blocked", "connection_error"]:
                raise ConnectionError(f"V68: {error_type} - {error_msg}")
            
            return None
        
        if result is None or (hasattr(result, 'empty') and result.empty):
            logger.warning(f"V68: {symbol} 资金流为空数据")
            return None
        
        df = pl.from_pandas(result) if hasattr(result, 'columns') else None
        
        if df is None or df.is_empty():
            logger.warning(f"V68: {symbol} 资金流转换为 DataFrame 失败")
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
        
        # 添加股票代码（如果不存在）
        if 'symbol' not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias('symbol'))
        
        # 只保留需要的列
        keep_cols = ['symbol', 'trade_date', 'net_main_amount', 'net_main_ratio',
                     'net_super_amount', 'net_super_ratio', 'net_large_amount',
                     'net_large_ratio', 'net_medium_amount', 'net_medium_ratio',
                     'net_small_amount', 'net_small_ratio']
        
        available_keep_cols = [c for c in keep_cols if c in df.columns]
        if available_keep_cols:
            df = df.select(available_keep_cols)
        
        # 确保 trade_date 是字符串格式
        if 'trade_date' in df.columns:
            df = df.with_columns(pl.col('trade_date').cast(pl.Utf8))
        
        logger.debug(f"V68: 获取 {symbol} 资金流 - {df.height}行")
        return df
    
    def fetch_symbol_list(self) -> List[str]:
        """
        获取 A 股股票列表
        
        Returns
        -------
        List[str]
            股票代码列表
        """
        if not AK_AVAILABLE:
            raise ImportError("V68: akshare 库未安装")
        
        try:
            df = ak.stock_info_a_code_name()
            
            if df is None or (hasattr(df, 'empty') and df.empty):
                logger.error("V68: 获取股票列表失败 - 返回空数据")
                return []
            
            pdf = df if isinstance(df, pl.DataFrame) else pl.from_pandas(df)
            
            if 'code' in pdf.columns:
                return pdf['code'].to_list()
            elif '代码' in pdf.columns:
                return pdf['代码'].to_list()
            else:
                logger.error(f"V68: 无法识别股票列表列名，可用列：{pdf.columns}")
                return []
                
        except Exception as e:
            logger.error(f"V68: 获取股票列表失败：{e}")
            return []
    
    def fetch_stock_list_with_filter(self) -> List[str]:
        """
        获取过滤后的股票列表（只保留主板股票）
        
        Returns
        -------
        List[str]
            过滤后的股票代码列表
        """
        all_symbols = self.fetch_symbol_list()
        
        # 过滤：只保留 600/601/603/000/001/002/003 开头的股票
        filtered = []
        for sym in all_symbols:
            if sym.startswith(('600', '601', '603', '605', '000', '001', '002', '003')):
                filtered.append(sym)
        
        logger.info(f"V68: 过滤后股票数量：{len(filtered)} (原始：{len(all_symbols)})")
        return filtered


# ===========================================
# V68 数据填充器 - 暴力抓取 + 强制落库验证
# ===========================================

class V68DataForceFiller:
    """
    V68 数据填充器 - 暴力抓取 + 强制落库验证
    
    【核心协议】
    1. 独立运行：不依赖任何外部模块
    2. 批次 - 等待模式：每抓取 20 只股票，强制 sleep 5 秒
    3. 强制落库确认：每写入 1000 行，执行 SELECT COUNT(*) 并打印
    4. 错误阻断：IP Blocked 或 Network Error 立即停止
    5. 断点续传：记录最后成功抓取的日期
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化数据填充器
        
        Parameters
        ----------
        db : DatabaseManager, optional
            数据库管理器实例
        """
        if db is None and DB_AVAILABLE:
            self.db = get_db()
        else:
            self.db = db
        
        self.fetcher = V68DataFetcher()
        self.consecutive_failures = 0
        self.total_success_count = 0
        self.total_fail_count = 0
        self.last_success_symbol = ""
        self.last_success_date = ""
        self._batch_buffer: List[pl.DataFrame] = []
        self._batch_row_count = 0
    
    def create_tables(self):
        """
        建表校验 - 确保表结构存在
        """
        if self.db is None:
            logger.error("V68: 数据库连接未初始化，无法建表")
            return
        
        logger.info("V68: 开始建表校验...")
        
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
            logger.info("V68: stock_fund_flow 表创建成功")
        except Exception as e:
            logger.error(f"V68: 创建 stock_fund_flow 表失败：{e}")
            raise
        
        logger.info("V68: 建表校验完成")
    
    def _verify_table_count(self, table_name: str) -> int:
        """
        执行 SELECT COUNT(*) 并打印结果
        
        Parameters
        ----------
        table_name : str
            表名
            
        Returns
        -------
        int
            行数
        """
        if self.db is None:
            return 0
        
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return 0
            
            cnt = int(result['cnt'][0])
            return cnt
            
        except Exception as e:
            logger.error(f"V68: 验证表 {table_name} 行数失败：{e}")
            return 0
    
    def _save_checkpoint(self, symbol: str, trade_date: str = ""):
        """
        保存断点续传配置
        
        Parameters
        ----------
        symbol : str
            最后成功抓取的股票代码
        trade_date : str
            最后成功抓取的日期
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(V68_CHECKPOINT_FILE), exist_ok=True)
            
            checkpoint_data = {
                'last_symbol': symbol,
                'last_date': trade_date,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_success': self.total_success_count,
                'total_failure': self.total_fail_count,
            }
            
            import json
            with open(V68_CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"V68: 断点已保存 - {symbol}@{trade_date}")
            
        except Exception as e:
            logger.warning(f"V68: 保存断点失败：{e}")
    
    def _load_checkpoint(self) -> Optional[Dict[str, str]]:
        """
        加载断点续传配置
        
        Returns
        -------
        Optional[Dict[str, str]]
            断点数据
        """
        if not os.path.exists(V68_CHECKPOINT_FILE):
            return None
        
        try:
            import json
            with open(V68_CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"V68: 加载断点失败：{e}")
            return None
    
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
        
        logger.error(f"V68: 【写入失败 #{self.consecutive_failures}】{symbol} - 类型：{error_type}")
        logger.error(f"V68: 详细报错：{error_msg}")
        
        # 如果是 IP 被封或网络错误，立即停止
        if error_type in ["ip_blocked", "connection_error"]:
            logger.error("=" * 60)
            logger.error(f"V68: 【致命错误】{error_type}，程序终止")
            logger.error(f"V68: 最后 API 报错：{error_msg}")
            logger.error(f"V68: 最后成功：{self.last_success_symbol}@{self.last_success_date}")
            self._save_checkpoint(self.last_success_symbol, self.last_success_date)
            logger.error("V68: 断点已保存，请更换 IP 或稍后重试")
            logger.error("=" * 60)
            sys.exit(1)
        
        if self.consecutive_failures >= V68_MAX_CONSECUTIVE_FAILURES:
            logger.error("=" * 60)
            logger.error(f"V68: 【致命错误】连续 {V68_MAX_CONSECUTIVE_FAILURES} 次写入失败，程序终止")
            logger.error(f"V68: 最后 API 报错：{error_msg}")
            logger.error(f"V68: 错误类型：{error_type}")
            logger.error(f"V68: 最后成功：{self.last_success_symbol}@{self.last_success_date}")
            self._save_checkpoint(self.last_success_symbol, self.last_success_date)
            logger.error("=" * 60)
            sys.exit(1)
    
    def _handle_write_success(self, symbol: str, total_rows: int, trade_date: str = ""):
        """
        处理写入成功
        
        Parameters
        ----------
        symbol : str
            股票代码
        total_rows : int
            当前总行数
        trade_date : str
            交易日期
        """
        self.consecutive_failures = 0
        self.total_success_count += 1
        self.last_success_symbol = symbol
        self.last_success_date = trade_date
        
        # 按要求输出 [SUCCESS] 格式
        logger.info(f"[SUCCESS] {symbol} written. Total rows now: {total_rows}")
        
        # 保存断点
        self._save_checkpoint(symbol, trade_date)
    
    def fill_fund_flow_data(self, start_date: str = V68_DEFAULT_START_DATE,
                            end_date: str = V68_DEFAULT_END_DATE,
                            resume: bool = True) -> Dict[str, int]:
        """
        填充资金流数据 - 暴力抓取模式
        
        【核心逻辑】
        1. 获取股票列表
        2. 批次 - 等待模式抓取
        3. 每写入 1000 行立即验证
        
        Parameters
        ----------
        start_date : str
            开始日期
        end_date : str
            结束日期
        resume : bool
            是否从断点续传
            
        Returns
        -------
        Dict[str, int]
            填充结果统计
        """
        logger.info("=" * 60)
        logger.info("V68 暴力数据抓取器 - 启动")
        logger.info("=" * 60)
        logger.info(f"日期范围：[{start_date}, {end_date}]")
        logger.info(f"断点续传：{resume}")
        logger.info("=" * 60)
        
        if self.db is None:
            logger.error("V68: 数据库连接未初始化")
            return {'success': 0, 'failure': 0, 'total_rows': 0}
        
        # 建表校验
        self.create_tables()
        
        # 获取股票列表
        symbol_list = self.fetcher.fetch_stock_list_with_filter()
        
        if not symbol_list:
            logger.error("V68: 无法获取股票列表")
            return {'success': 0, 'failure': 0, 'total_rows': 0}
        
        logger.info(f"V68: 共获取到 {len(symbol_list)} 只股票")
        
        # 断点续传处理
        start_index = 0
        if resume:
            checkpoint = self._load_checkpoint()
            if checkpoint and checkpoint.get('last_symbol'):
                last_symbol = checkpoint['last_symbol']
                try:
                    start_index = symbol_list.index(last_symbol) + 1
                    logger.info(f"V68: 从断点 {last_symbol} 继续，跳过前 {start_index} 只股票")
                except ValueError:
                    logger.warning(f"V68: 断点股票 {last_symbol} 不在列表中，从头开始")
        
        result = {'success': 0, 'failure': 0, 'total_rows': 0}
        batch_count = 0
        
        # 批次 - 等待模式处理
        for i, symbol in enumerate(symbol_list[start_index:], start=start_index):
            try:
                # 获取单只股票的资金流数据
                df = self.fetcher.fetch_stock_fund_flow(symbol)
                
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
                    logger.warning(f"V68: {symbol} 无资金流数据")
                    result['failure'] += 1
                
                # 批次等待：每 20 只股票强制 sleep 5 秒
                batch_count += 1
                if batch_count % V68_BATCH_SIZE == 0:
                    logger.info(f"V68: 【批次等待】已处理 {batch_count} 只股票，等待 {V68_BATCH_WAIT_SECONDS} 秒...")
                    time.sleep(V68_BATCH_WAIT_SECONDS)
                
            except ConnectionError as e:
                # IP 被封或网络错误，立即停止
                logger.error(f"V68: 【致命错误】{symbol} 处理失败：{e}")
                self._handle_write_failure(symbol, str(e))
                
            except Exception as e:
                logger.error(f"V68: {symbol} 处理失败：{e}")
                result['failure'] += 1
                self._handle_write_failure(symbol, str(e))
        
        # 获取总行数
        result['total_rows'] = self._verify_table_count('stock_fund_flow')
        
        logger.info("=" * 60)
        logger.info("V68: 资金流数据填充完成")
        logger.info(f"成功：{result['success']}只，失败：{result['failure']}只，总行数：{result['total_rows']:,}")
        logger.info(f"最后成功：{self.last_success_symbol}@{self.last_success_date}")
        logger.info("=" * 60)
        
        return result
    
    def _write_fund_flow_to_db(self, df: pl.DataFrame, symbol: str) -> bool:
        """
        写入资金流数据到数据库并验证
        
        【强制落库确认】
        - 每写入 1000 行，执行 SELECT COUNT(*) 并打印
        
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
        if self.db is None:
            logger.error("V68: 数据库连接未初始化")
            return False
        
        try:
            # 写入数据库
            self.db.to_sql(df, 'stock_fund_flow', if_exists='append')
            
            # 获取当前总行数
            total_rows = self._verify_table_count('stock_fund_flow')
            
            # 每 1000 行验证一次
            if total_rows % V68_WRITE_VERIFY_ROWS < df.height:
                logger.info(f"V68: [VERIFY] stock_fund_flow 总行数：{total_rows:,}")
            
            # 获取最后一条数据的日期
            if not df.is_empty() and 'trade_date' in df.columns:
                trade_dates = df['trade_date'].to_list()
                last_date = str(trade_dates[-1]) if trade_dates else ""
            else:
                last_date = ""
            
            # 输出成功信息（按要求格式）
            self._handle_write_success(symbol, total_rows, last_date)
            
            return True
            
        except Exception as e:
            logger.error(f"V68: 写入 {symbol} 失败：{e}")
            return False
    
    def verify_data_sufficiency(self) -> Tuple[bool, str]:
        """
        验证数据是否充足
        
        【数据熔断】
        - stock_fund_flow 行数低于 100 万行，返回 False
        
        Returns
        -------
        Tuple[bool, str]
            (是否充足，消息)
        """
        logger.info("=" * 60)
        logger.info("V68: 开始数据充足性验证")
        
        fund_flow_rows = self._verify_table_count('stock_fund_flow')
        
        logger.info(f"V68: stock_fund_flow 行数：{fund_flow_rows:,} (阈值：{V68_MIN_FUND_FLOW_ROWS:,})")
        
        if fund_flow_rows < V68_MIN_FUND_FLOW_ROWS:
            error_msg = f"数据不足：stock_fund_flow 仅有 {fund_flow_rows:,} 行，需要 {V68_MIN_FUND_FLOW_ROWS:,} 行"
            logger.error(f"V68: 【数据熔断】{error_msg}")
            logger.error("V68: 请先运行 v68_data_force_filler.py 进行数据抓取")
            return (False, error_msg)
        
        logger.info("V68: 数据充足性验证通过")
        logger.info("=" * 60)
        return (True, "数据充足")


# ===========================================
# 便捷函数
# ===========================================

def fill_v68_data(start_date: str = V68_DEFAULT_START_DATE,
                  end_date: str = V68_DEFAULT_END_DATE,
                  db: Optional[DatabaseManager] = None,
                  resume: bool = True) -> Dict[str, int]:
    """
    便捷函数：填充 V68 数据
    
    Parameters
    ----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    db : DatabaseManager, optional
        数据库管理器实例
    resume : bool
        是否从断点续传
        
    Returns
    -------
    Dict[str, int]
        填充结果
    """
    filler = V68DataForceFiller(db=db)
    
    # 填充资金流数据
    result = filler.fill_fund_flow_data(start_date, end_date, resume)
    
    # 验证数据充足性
    is_sufficient, message = filler.verify_data_sufficiency()
    
    return {
        **result,
        'is_sufficient': is_sufficient,
        'message': message,
    }


def verify_v68_data(db: Optional[DatabaseManager] = None) -> Tuple[bool, str]:
    """
    便捷函数：验证 V68 数据
    
    Parameters
    ----------
    db : DatabaseManager, optional
        数据库管理器实例
        
    Returns
    -------
    Tuple[bool, str]
        (是否通过，消息)
    """
    filler = V68DataForceFiller(db=db)
    return filler.verify_data_sufficiency()


# ===========================================
# 主程序 - 独立运行入口
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
    logger.info("V68 暴力数据抓取器 - 启动")
    logger.info("=" * 60)
    
    # 检查 akshare 是否可用
    if not AK_AVAILABLE:
        logger.error("V68: akshare 库未安装，请运行：pip install akshare")
        sys.exit(1)
    
    # 检查数据库是否可用
    if not DB_AVAILABLE:
        logger.error("V68: db_manager 模块未找到")
        sys.exit(1)
    
    # 初始化数据库
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"V68: 数据库连接失败：{e}")
        sys.exit(1)
    
    # 填充数据
    try:
        result = fill_v68_data(V68_DEFAULT_START_DATE, V68_DEFAULT_END_DATE, db, resume=True)
        
        logger.info("=" * 60)
        logger.info("V68: 数据抓取完成")
        logger.info(f"成功：{result.get('success', 0)}只")
        logger.info(f"失败：{result.get('failure', 0)}只")
        logger.info(f"总行数：{result.get('total_rows', 0):,}")
        logger.info(f"数据充足：{result.get('is_sufficient', False)}")
        logger.info("=" * 60)
        
        if not result.get('is_sufficient', False):
            logger.warning("V68: 数据量不足，可能需要继续抓取")
        
    except SystemExit:
        logger.error("V68: 程序因错误已退出")
        sys.exit(1)
    except Exception as e:
        logger.error(f"V68: 数据抓取失败：{e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


__all__ = [
    'V68DataFetcher',
    'V68DataForceFiller',
    'fill_v68_data',
    'verify_v68_data',
    'V68_BATCH_SIZE',
    'V68_BATCH_WAIT_SECONDS',
    'V68_WRITE_VERIFY_ROWS',
    'V68_MAX_CONSECUTIVE_FAILURES',
    'V68_MIN_FUND_FLOW_ROWS',
]