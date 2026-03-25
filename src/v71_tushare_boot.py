"""
V71 Tushare Boot Module - 基于 Tushare 的数据强制补完计划

【V71 核心特性】
1. 读取 config/settings.yaml 中的 TUSHARE_CONFIG 配置
2. 调用 Tushare API 获取 2024 年全年数据
3. 实现限频处理、断点续传、实时监控
4. 数据入库后自动输出统计报表

【数据映射规则】
1. stock_fund_flow 表：
   - trade_date (20240102) -> date (2024-01-02)
   - ts_code -> stock_code
   - net_mf_amount -> net_main_inflow (单位：万元)

2. stock_industry_daily 表：
   - trade_date -> date
   - ts_code -> industry_code
   - close -> close
   - pct_chg -> pct_chg
   - amount -> net_main_inflow (行业维度以成交额表征)

作者：量化系统
版本：V71.0
日期：2026-03-25
"""

import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import tushare as ts
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))
from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================

# 目标时间范围
V71_START_DATE = "2024-01-01"
V71_END_DATE = "2024-12-31"

# 限频配置
V71_REQUEST_DELAY = 6.0  # 请求间隔（秒）- 增加以避免 sw_daily 接口限频（每分钟 10 次）
V71_RATE_LIMIT_SLEEP = 60  # 触发限频后休眠时间（秒）
V71_MAX_RETRIES = 3  # 最大重试次数

# 日志配置
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================
# 限频器 - RateLimiter
# ===========================================

class RateLimiter:
    """
    限频器 - 处理 Tushare API 的频率限制
    
    【核心功能】
    1. 检测限频错误
    2. 自动休眠 60 秒后重试
    3. 记录限频次数
    """
    
    def __init__(self, sleep_seconds: int = V71_RATE_LIMIT_SLEEP):
        """
        初始化限频器
        
        Parameters
        ----------
        sleep_seconds : int
            触发限频后休眠的秒数
        """
        self.sleep_seconds = sleep_seconds
        self.rate_limit_count = 0
        self.last_request_time = 0.0
    
    def check_rate_limit(self, error_msg: str) -> bool:
        """
        检查是否是限频错误
        
        Parameters
        ----------
        error_msg : str
            错误消息
        
        Returns
        -------
        bool
            是否是限频错误
        """
        rate_limit_keywords = [
            "抱歉，您每分钟访问超过限制",
            "访问频率过高",
            "rate limit",
            "too many requests",
            "429",
        ]
        error_lower = error_msg.lower()
        return any(keyword in error_lower for keyword in rate_limit_keywords)
    
    def handle_rate_limit(self, retry_count: int = 0) -> None:
        """
        处理限频 - 自动休眠
        
        Parameters
        ----------
        retry_count : int
            当前重试次数
        """
        self.rate_limit_count += 1
        logger.warning(f"⚠️  触发限频！第 {self.rate_limit_count} 次，休眠 {self.sleep_seconds} 秒后继续...")
        time.sleep(self.sleep_seconds)
    
    def wait_between_requests(self) -> None:
        """请求间等待"""
        elapsed = time.time() - self.last_request_time
        if elapsed < V71_REQUEST_DELAY:
            sleep_time = V71_REQUEST_DELAY - elapsed
            time.sleep(sleep_time)
        self.last_request_time = time.time()


# ===========================================
# Tushare 数据抓取器
# ===========================================

class V71TushareFetcher:
    """
    V71 Tushare 数据抓取器
    
    【核心功能】
    1. 调用 Tushare API 获取资金流和行业数据
    2. 字段映射转换
    3. 限频处理
    4. 断点续传
    """
    
    def __init__(self, token: str, db: Optional[DatabaseManager] = None):
        """
        初始化 Tushare 抓取器
        
        Parameters
        ----------
        token : str
            Tushare API Token
        db : Optional[DatabaseManager]
            数据库管理器实例
        """
        self.token = token
        self.db = db
        self.pro = ts.pro_api(token)
        self.rate_limiter = RateLimiter()
        
        # 统计数据
        self.stats = {
            "total_days": 0,
            "success_days": 0,
            "failed_days": 0,
            "total_rows_fund_flow": 0,
            "total_rows_industry": 0,
        }
    
    def _convert_date_format(self, date_str: str, to_format: str = "%Y-%m-%d") -> str:
        """
        转换日期格式
        
        Parameters
        ----------
        date_str : str
            日期字符串
        to_format : str
            目标格式
        
        Returns
        -------
        str
            转换后的日期字符串
        """
        # Tushare 返回 YYYYMMDD 格式
        if len(date_str) == 8:
            dt = datetime.strptime(date_str, "%Y%m%d")
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime(to_format)
    
    def get_existing_dates(self, table_name: str) -> set:
        """
        获取数据库中已存在的日期（断点续传）
        
        Parameters
        ----------
        table_name : str
            表名
        
        Returns
        -------
        set
            已存在的日期集合
        """
        if self.db is None:
            return set()
        
        try:
            # 两个表都使用 trade_date 字段
            date_field = "trade_date"
            query = f"SELECT DISTINCT {date_field} FROM {table_name}"
            df = self.db.read_sql(query)
            if len(df) > 0:
                return set(df[date_field].to_list())
        except Exception as e:
            logger.debug(f"获取已存在日期失败（表可能不存在）: {e}")
        
        return set()
    
    def fetch_moneyflow(self, trade_date: str) -> Optional[pl.DataFrame]:
        """
        获取个股资金流数据
        
        调用接口：pro.moneyflow(trade_date='YYYYMMDD')
        
        数据库表 stock_fund_flow 字段：
        - symbol: 股票代码
        - trade_date: 交易日期
        - net_main_amount: 主力资金净流入（万元）
        - net_main_rate: 主力净流入占比
        
        Parameters
        ----------
        trade_date : str
            交易日期（YYYYMMDD 格式）
        
        Returns
        -------
        Optional[pl.DataFrame]
            资金流数据
        """
        for retry in range(V71_MAX_RETRIES):
            try:
                self.rate_limiter.wait_between_requests()
                
                df = self.pro.moneyflow(trade_date=trade_date)
                
                if df is None or len(df) == 0:
                    logger.debug(f"moneyflow 返回空数据：{trade_date}")
                    return None
                
                # 字段映射 - 适配数据库实际字段
                records = []
                for _, row in df.iterrows():
                    record = {
                        "symbol": str(row["ts_code"]),
                        "trade_date": self._convert_date_format(str(row["trade_date"])),
                        "net_main_amount": float(row["net_mf_amount"]) if row.get("net_mf_amount") is not None else 0.0,
                        "net_main_rate": float(row["net_mf_ratio"]) if row.get("net_mf_ratio") is not None else 0.0,
                    }
                    records.append(record)
                
                result_df = pl.DataFrame(records)
                logger.debug(f"moneyflow 成功：{trade_date}, {len(result_df)} 行")
                return result_df
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是限频错误
                if self.rate_limiter.check_rate_limit(error_msg):
                    self.rate_limiter.handle_rate_limit(retry)
                    continue
                
                logger.error(f"fetch_moneyflow 失败 ({trade_date}): {error_msg}")
                if retry >= V71_MAX_RETRIES - 1:
                    return None
        
        return None
    
    def fetch_sw_daily(self, trade_date: str) -> Optional[pl.DataFrame]:
        """
        获取行业日线数据
        
        调用接口：pro.sw_daily(trade_date='YYYYMMDD')
        
        数据库表 stock_industry_daily 字段：
        - symbol: 行业代码
        - trade_date: 交易日期
        - industry_name: 行业名称
        - industry_code: 行业代码
        
        Parameters
        ----------
        trade_date : str
            交易日期（YYYYMMDD 格式）
        
        Returns
        -------
        Optional[pl.DataFrame]
            行业日线数据
        """
        for retry in range(V71_MAX_RETRIES):
            try:
                self.rate_limiter.wait_between_requests()
                
                df = self.pro.sw_daily(trade_date=trade_date)
                
                if df is None or len(df) == 0:
                    logger.debug(f"sw_daily 返回空数据：{trade_date}")
                    return None
                
                # 字段映射 - 适配数据库实际字段
                records = []
                for _, row in df.iterrows():
                    record = {
                        "symbol": str(row["ts_code"]),
                        "trade_date": self._convert_date_format(str(row["trade_date"])),
                        "industry_name": str(row.get("name", "")),
                        "industry_code": str(row["ts_code"]),
                    }
                    records.append(record)
                
                result_df = pl.DataFrame(records)
                logger.debug(f"sw_daily 成功：{trade_date}, {len(result_df)} 行")
                return result_df
                
            except Exception as e:
                error_msg = str(e)
                
                # 检查是否是限频错误
                if self.rate_limiter.check_rate_limit(error_msg):
                    self.rate_limiter.handle_rate_limit(retry)
                    continue
                
                logger.error(f"fetch_sw_daily 失败 ({trade_date}): {error_msg}")
                if retry >= V71_MAX_RETRIES - 1:
                    return None
        
        return None
    
    def save_to_db(self, df: pl.DataFrame, table_name: str) -> int:
        """
        保存数据到数据库
        
        Parameters
        ----------
        df : pl.DataFrame
            要保存的数据
        table_name : str
            表名
        
        Returns
        -------
        int
            保存的行数
        """
        if self.db is None:
            logger.error("数据库连接未初始化")
            return 0
        
        if df.is_empty():
            logger.warning("数据为空，跳过保存")
            return 0
        
        try:
            self.db.to_sql(df, table_name, if_exists="append")
            logger.info(f"成功保存 {len(df)} 条数据到 {table_name}")
            return len(df)
        except Exception as e:
            logger.error(f"保存数据失败：{e}")
            return 0
    
    def sync_fund_flow(self, start_date: str, end_date: str) -> Tuple[int, int]:
        """
        同步个股资金流数据
        
        Parameters
        ----------
        start_date : str
            开始日期（YYYY-MM-DD）
        end_date : str
            结束日期（YYYY-MM-DD）
        
        Returns
        -------
        Tuple[int, int]
            (成功天数，总行数)
        """
        logger.info(f"开始同步 stock_fund_flow: {start_date} -> {end_date}")
        
        # 获取已存在的日期
        existing_dates = self.get_existing_dates("stock_fund_flow")
        if existing_dates:
            logger.info(f"stock_fund_flow 已存在 {len(existing_dates)} 天的数据，将跳过")
        
        # 生成日期列表
        date_list = self._generate_trade_dates(start_date, end_date)
        dates_to_sync = [d for d in date_list if d not in existing_dates]
        
        if not dates_to_sync:
            logger.info("✓ stock_fund_flow 数据已完整，无需同步")
            return 0, 0
        
        logger.info(f"需要同步 {len(dates_to_sync)} 天的数据")
        
        success_count = 0
        total_rows = 0
        
        for i, date in enumerate(dates_to_sync):
            # 转换为 YYYYMMDD 格式用于 API 调用
            api_date = date.replace("-", "")
            
            df = self.fetch_moneyflow(api_date)
            
            if df is not None and not df.is_empty():
                rows = self.save_to_db(df, "stock_fund_flow")
                total_rows += rows
                success_count += 1
                self.stats["total_rows_fund_flow"] += rows
                
                # 实时监控输出
                print(f"[SUCCESS] {date}: 写入 {rows} 行 | 累计已写入 {self.stats['total_rows_fund_flow']} 行")
            else:
                logger.warning(f"跳过 {date}（无数据或失败）")
        
        return success_count, total_rows
    
    def sync_industry_daily(self, start_date: str, end_date: str) -> Tuple[int, int]:
        """
        同步行业日线数据
        
        Parameters
        ----------
        start_date : str
            开始日期（YYYY-MM-DD）
        end_date : str
            结束日期（YYYY-MM-DD）
        
        Returns
        -------
        Tuple[int, int]
            (成功天数，总行数)
        """
        logger.info(f"开始同步 stock_industry_daily: {start_date} -> {end_date}")
        
        # 获取已存在的日期
        existing_dates = self.get_existing_dates("stock_industry_daily")
        if existing_dates:
            logger.info(f"stock_industry_daily 已存在 {len(existing_dates)} 天的数据，将跳过")
        
        # 生成日期列表
        date_list = self._generate_trade_dates(start_date, end_date)
        dates_to_sync = [d for d in date_list if d not in existing_dates]
        
        if not dates_to_sync:
            logger.info("✓ stock_industry_daily 数据已完整，无需同步")
            return 0, 0
        
        logger.info(f"需要同步 {len(dates_to_sync)} 天的数据")
        
        success_count = 0
        total_rows = 0
        
        for i, date in enumerate(dates_to_sync):
            # 转换为 YYYYMMDD 格式用于 API 调用
            api_date = date.replace("-", "")
            
            df = self.fetch_sw_daily(api_date)
            
            if df is not None and not df.is_empty():
                rows = self.save_to_db(df, "stock_industry_daily")
                total_rows += rows
                success_count += 1
                self.stats["total_rows_industry"] += rows
                
                # 实时监控输出
                print(f"[SUCCESS] {date}: 写入 {rows} 行 | 累计已写入 {self.stats['total_rows_industry']} 行")
            else:
                logger.warning(f"跳过 {date}（无数据或失败）")
        
        return success_count, total_rows
    
    def _generate_trade_dates(self, start_date: str, end_date: str) -> List[str]:
        """
        生成交易日列表（跳过周末）
        
        Parameters
        ----------
        start_date : str
            开始日期（YYYY-MM-DD）
        end_date : str
            结束日期（YYYY-MM-DD）
        
        Returns
        -------
        List[str]
            日期列表
        """
        dates = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current <= end:
            # 跳过周末（0=周一，6=周日）
            if current.weekday() < 5:
                dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates


# ===========================================
# 统计报表生成器
# ===========================================

class V71StatsReporter:
    """
    V71 统计报表生成器
    
    【核心功能】
    1. 执行 SQL 统计
    2. 输出月度数据表
    """
    
    def __init__(self, db: DatabaseManager):
        """
        初始化统计报表生成器
        
        Parameters
        ----------
        db : DatabaseManager
            数据库管理器实例
        """
        self.db = db
    
    def generate_monthly_stats(self, table_name: str) -> Optional[pl.DataFrame]:
        """
        生成月度统计表
        
        Parameters
        ----------
        table_name : str
            表名
        
        Returns
        -------
        Optional[pl.DataFrame]
            月度统计数据
        """
        try:
            # 两个表都使用 trade_date 字段
            date_field = "trade_date"
            
            query = f"""
                SELECT 
                    SUBSTR({date_field}, 1, 7) as month,
                    COUNT(*) as record_count
                FROM {table_name}
                WHERE {date_field} >= '2024-01-01' AND {date_field} <= '2024-12-31'
                GROUP BY SUBSTR({date_field}, 1, 7)
                ORDER BY month
            """
            df = self.db.read_sql(query)
            return df
        except Exception as e:
            logger.error(f"生成月度统计失败：{e}")
            return None
    
    def print_stats_table(self, table_name: str) -> None:
        """
        打印统计表
        
        Parameters
        ----------
        table_name : str
            表名
        """
        logger.info("=" * 60)
        logger.info(f"{table_name} - 2024 年月度统计表")
        logger.info("=" * 60)
        
        df = self.generate_monthly_stats(table_name)
        
        if df is not None and len(df) > 0:
            print(f"\n{'='*60}")
            print(f"{table_name} - 2024 年月度数据条数统计")
            print(f"{'='*60}")
            print(f"{'月份':<12} {'记录条数':>15}")
            print(f"{'-'*60}")
            
            total = 0
            for row in df.to_dicts():
                month = str(row["month"])
                count = int(row["record_count"])
                total += count
                print(f"{month:<12} {count:>15,}")
            
            print(f"{'-'*60}")
            print(f"{'总计':<12} {total:>15,}")
            print(f"{'='*60}\n")
        else:
            logger.warning(f"{table_name} 暂无数据")


# ===========================================
# 主运行函数
# ===========================================

def load_tushare_config() -> Dict[str, Any]:
    """
    从 config/settings.yaml 加载 Tushare 配置
    
    注意：TUSHARE_CONFIG 是 Python 字典格式，需要使用 exec 解析
    
    Returns
    -------
    Dict[str, Any]
        配置字典
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 查找 TUSHARE_CONFIG 字典
    config_dict = {}
    
    # 使用简单解析方式，查找 TUSHARE_CONFIG 行
    lines = content.split('\n')
    in_config = False
    config_lines = []
    
    for line in lines:
        if line.strip().startswith('TUSHARE_CONFIG'):
            in_config = True
            # 提取字典内容
            if '=' in line:
                dict_part = line.split('=', 1)[1].strip()
                config_lines.append(dict_part)
        elif in_config:
            config_lines.append(line)
            if line.strip() == '}':
                break
    
    config_str = '\n'.join(config_lines)
    
    # 安全地评估字典
    try:
        # 使用 ast.literal_eval 安全解析
        import ast
        config_dict = ast.literal_eval(config_str)
    except Exception as e:
        # 如果解析失败，尝试直接 exec
        exec_globals = {}
        exec(f"_tushare_config = {config_str}", exec_globals)
        config_dict = exec_globals.get("_tushare_config", {})
    
    return config_dict


def run_v71_tushare_boot() -> Dict[str, Any]:
    """
    运行 V71 Tushare 数据补完计划
    
    Returns
    -------
    Dict[str, Any]
        运行结果
    """
    logger.info("=" * 80)
    logger.info("V71 Tushare Boot - 数据强制补完计划启动")
    logger.info("=" * 80)
    
    result = {
        "success": False,
        "fund_flow_days": 0,
        "fund_flow_rows": 0,
        "industry_days": 0,
        "industry_rows": 0,
        "error": None,
    }
    
    try:
        # 1. 加载配置
        logger.info("步骤 1: 加载 Tushare 配置...")
        tushare_config = load_tushare_config()
        
        if not tushare_config:
            raise ValueError("TUSHARE_CONFIG 未找到，请检查 config/settings.yaml")
        
        token = tushare_config.get("token")
        if not token:
            raise ValueError("Tushare Token 未配置")
        
        logger.info(f"✓ Token 已加载：{token[:10]}...{token[-4:]}")
        
        # 2. 初始化数据库
        logger.info("步骤 2: 初始化数据库连接...")
        db = DatabaseManager()
        db.connect()
        logger.info("✓ 数据库连接已建立")
        
        # 3. 初始化抓取器
        logger.info("步骤 3: 初始化 Tushare 抓取器...")
        fetcher = V71TushareFetcher(token=token, db=db)
        
        # 4. 同步个股资金流数据
        logger.info("步骤 4: 开始同步 stock_fund_flow...")
        fund_flow_days, fund_flow_rows = fetcher.sync_fund_flow(
            start_date=V71_START_DATE,
            end_date=V71_END_DATE,
        )
        result["fund_flow_days"] = fund_flow_days
        result["fund_flow_rows"] = fund_flow_rows
        
        # 5. 同步行业日线数据
        logger.info("步骤 5: 开始同步 stock_industry_daily...")
        industry_days, industry_rows = fetcher.sync_industry_daily(
            start_date=V71_START_DATE,
            end_date=V71_END_DATE,
        )
        result["industry_days"] = industry_days
        result["industry_rows"] = industry_rows
        
        # 6. 生成统计报表
        logger.info("步骤 6: 生成统计报表...")
        reporter = V71StatsReporter(db)
        
        print("\n" + "=" * 80)
        print("【V71 数据补完计划 - 最终统计】")
        print("=" * 80)
        
        reporter.print_stats_table("stock_fund_flow")
        reporter.print_stats_table("stock_industry_daily")
        
        # 7. 限频统计
        if fetcher.rate_limiter.rate_limit_count > 0:
            logger.info(f"限频统计：共触发 {fetcher.rate_limiter.rate_limit_count} 次限频等待")
        
        result["success"] = True
        
    except Exception as e:
        result["error"] = f"运行异常：{e}"
        logger.error(traceback.format_exc())
    
    logger.info("=" * 80)
    logger.info(f"V71 运行完成：成功={result['success']}")
    if result["error"]:
        logger.error(f"错误：{result['error']}")
    logger.info(f"个股资金流：{result['fund_flow_days']} 天，{result['fund_flow_rows']} 行")
    logger.info(f"行业日线：{result['industry_days']} 天，{result['industry_rows']} 行")
    logger.info("=" * 80)
    
    return result


# ===========================================
# CLI 入口
# ===========================================

if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        LOG_DIR / "v71_tushare_boot_{time:YYYYMMDD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
    )
    
    result = run_v71_tushare_boot()
    
    sys.exit(0 if result["success"] else 1)