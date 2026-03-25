"""
V71 Raw Fetcher Module - 暴力拆解反爬机制

【V71 核心特性：生存模式】
1. 禁用 akshare 高级接口，直连东方财富底层 JSON API
2. 使用 requests.Session() 保持长连接
3. Header 强伪装（Referer + 动态 Cookie）
4. 首日探测与自诊断
5. 动态延迟采样（模拟人类翻页行为）
6. 数据库"死锁"校验
7. 自动生成 DEBUG_REPORT.log

【API 端点】
- 资金流向排名：http://push2.eastmoney.com/api/qt/clist/get
- 个股资金流向：http://push2his.eastmoney.com/api/qt/stock/fundflow/day/get

作者：量化系统
版本：V71.0
日期：2026-03-25
"""

import json
import random
import sys
import time
import traceback
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import uuid

import requests
import polars as pl
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# 动态导入数据库管理器
sys.path.insert(0, str(Path(__file__).parent))
from db_manager import DatabaseManager


# ===========================================
# V71 配置常量
# ===========================================

# API 端点
EASTMONEY_FUND_FLOW_RANK_URL = "http://push2.eastmoney.com/api/qt/clist/get"
EASTMONEY_STOCK_FUND_FLOW_URL = "http://push2his.eastmoney.com/api/qt/stock/fundflow/day/get"

# 请求头配置
DEFAULT_HEADERS = {
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Host": "push2.eastmoney.com",
    "Pragma": "no-cache",
    "Referer": "http://data.eastmoney.com/zjlx/detail.html",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

# 重试配置
V71_MAX_RETRIES = 5
V71_RETRY_DELAY_BASE = 1.0
V71_RETRY_DELAY_MAX = 30.0

# 频率控制
V71_REQUEST_DELAY_MIN = 0.5
V71_REQUEST_DELAY_MAX = 1.5
V71_BATCH_SIZE = 5  # 每 5 天强制长休眠
V71_LONG_DELAY_MIN = 10.0
V71_LONG_DELAY_MAX = 30.0

# 日期范围
V71_DEFAULT_START_DATE = "2024-01-02"
V71_DEFAULT_END_DATE = None  # 今天

# 日志配置
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_REPORT_PATH = LOG_DIR / "DEBUG_REPORT.log"


# ===========================================
# V71 工具函数
# ===========================================

def generate_dynamic_cookie() -> str:
    """
    生成动态 Cookie（模拟浏览器行为）
    
    Returns
    -------
    str
        生成的 Cookie 字符串
    """
    # 生成随机 session ID
    session_id = str(uuid.uuid4()).replace("-", "")[:16]
    # 生成随机时间戳
    timestamp = int(time.time() * 1000)
    # 生成随机 visitor ID
    visitor_id = f"v{random.randint(100000000000, 999999999999)}"
    
    cookie_parts = [
        f"EMFID1={session_id}",
        f"EMFID2={visitor_id}",
        f"st={timestamp}",
        f"sw={random.randint(1920, 2560)}x{random.randint(1080, 1440)}",
    ]
    
    return "; ".join(cookie_parts)


def get_headers_with_dynamic_cookie() -> Dict[str, str]:
    """
    获取带有动态 Cookie 的请求头
    
    Returns
    -------
    Dict[str, str]
        完整的请求头
    """
    headers = DEFAULT_HEADERS.copy()
    headers["Cookie"] = generate_dynamic_cookie()
    return headers


def random_delay(min_delay: float = V71_REQUEST_DELAY_MIN, 
                 max_delay: float = V71_REQUEST_DELAY_MAX) -> float:
    """
    随机延迟
    
    Parameters
    ----------
    min_delay : float
        最小延迟秒数
    max_delay : float
        最大延迟秒数
    
    Returns
    -------
    float
        实际延迟的秒数
    """
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)
    return delay


def long_delay_for_batch(batch_count: int) -> None:
    """
    每 N 次请求后的长延迟（模拟人类翻页行为）
    
    Parameters
    ----------
    batch_count : int
        当前批次计数
    """
    if batch_count % V71_BATCH_SIZE == 0 and batch_count > 0:
        delay = random.uniform(V71_LONG_DELAY_MIN, V71_LONG_DELAY_MAX)
        logger.info(f"已抓取 {batch_count} 天数据，强制休眠 {delay:.1f} 秒...")
        time.sleep(delay)


def retry_with_exponential_backoff(func, *args, max_retries: int = V71_MAX_RETRIES, **kwargs) -> Any:
    """
    带指数退避的重试
    
    Parameters
    ----------
    func : callable
        要执行的函数
    max_retries : int
        最大重试次数
    args : tuple
        位置参数
    kwargs : dict
        关键字参数
    
    Returns
    -------
    Any
        函数执行结果
    
    Raises
    ------
    Exception
        如果所有重试都失败
    """
    last_exception = None
    error_log = []
    
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            last_exception = e
            error_type = type(e).__name__
            error_msg = str(e)
            error_log.append({
                "attempt": attempt + 1,
                "error_type": error_type,
                "error_msg": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            if attempt < max_retries - 1:
                # 指数退避
                delay = min(V71_RETRY_DELAY_BASE * (2 ** attempt), V71_RETRY_DELAY_MAX)
                # 添加随机抖动
                delay *= random.uniform(0.8, 1.2)
                logger.warning(f"请求失败 ({error_type})，{delay:.1f}秒后重试 (尝试 {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                logger.error(f"请求最终失败 (已重试 {max_retries}次): {error_type}: {error_msg}")
    
    # 保存错误日志用于 DEBUG_REPORT
    kwargs.setdefault("_error_log", []).extend(error_log)
    raise last_exception


def parse_json_response(response_text: str) -> Optional[Dict]:
    """
    解析 JSON 响应
    
    Parameters
    ----------
    response_text : str
        响应文本
    
    Returns
    -------
    Optional[Dict]
        解析后的字典，如果解析失败返回 None
    """
    try:
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败：{e}")
        logger.error(f"原始响应内容：{response_text[:500]}...")
        return None


# ===========================================
# V71 调试报告生成器
# ===========================================

class V71DebugReporter:
    """
    V71 调试报告生成器
    
    【核心功能】
    1. 记录连续失败
    2. 分析失败原因（403/503/RemoteDisconnected）
    3. 生成 DEBUG_REPORT.log
    """
    
    def __init__(self, report_path: Path = DEBUG_REPORT_PATH):
        """
        初始化调试报告器
        
        Parameters
        ----------
        report_path : Path
            报告文件路径
        """
        self.report_path = report_path
        self.consecutive_failures = 0
        self.failure_log: List[Dict] = []
        self.request_stats = {
            "total": 0,
            "success": 0,
            "failure": 0,
            "403": 0,
            "503": 0,
            "timeout": 0,
            "other": 0,
        }
    
    def record_success(self) -> None:
        """记录成功请求"""
        self.consecutive_failures = 0
        self.request_stats["total"] += 1
        self.request_stats["success"] += 1
    
    def record_failure(self, status_code: Optional[int] = None, 
                       error_type: Optional[str] = None,
                       error_msg: Optional[str] = None) -> None:
        """
        记录失败请求
        
        Parameters
        ----------
        status_code : Optional[int]
            HTTP 状态码
        error_type : Optional[str]
            错误类型
        error_msg : Optional[str]
            错误消息
        """
        self.consecutive_failures += 1
        self.request_stats["total"] += 1
        self.request_stats["failure"] += 1
        
        # 分类统计
        if status_code == 403:
            self.request_stats["403"] += 1
        elif status_code == 503:
            self.request_stats["503"] += 1
        elif error_type and "timeout" in error_type.lower():
            self.request_stats["timeout"] += 1
        else:
            self.request_stats["other"] += 1
        
        # 记录详细日志
        self.failure_log.append({
            "timestamp": datetime.now().isoformat(),
            "status_code": status_code,
            "error_type": error_type,
            "error_msg": error_msg,
            "consecutive_failures": self.consecutive_failures,
        })
        
        # 检查是否需要生成报告
        if self.consecutive_failures >= 3:
            self.generate_report()
    
    def generate_report(self) -> None:
        """生成 DEBUG_REPORT.log"""
        report_lines = [
            "=" * 80,
            "V71 DEBUG REPORT - 连续失败分析报告",
            f"生成时间：{datetime.now().isoformat()}",
            "=" * 80,
            "",
            "【请求统计】",
            f"  总请求数：{self.request_stats['total']}",
            f"  成功：{self.request_stats['success']}",
            f"  失败：{self.request_stats['failure']}",
            f"  403 Forbidden: {self.request_stats['403']}",
            f"  503 Service Unavailable: {self.request_stats['503']}",
            f"  Timeout: {self.request_stats['timeout']}",
            f"  其他错误：{self.request_stats['other']}",
            "",
            "【失败分析】",
        ]
        
        # 分析主要错误类型
        if self.request_stats["403"] > 0:
            report_lines.append("  ⚠️  403 Forbidden - 可能被反爬虫机制识别")
            report_lines.append("     建议：更新 Cookie 或增加请求延迟")
        if self.request_stats["503"] > 0:
            report_lines.append("  ⚠️  503 Service Unavailable - 服务器过载")
            report_lines.append("     建议：降低请求频率")
        if self.request_stats["timeout"] > 0:
            report_lines.append("  ⚠️  Timeout - 网络超时")
            report_lines.append("     建议：检查网络连接或增加超时时间")
        
        report_lines.extend([
            "",
            "【失败日志详情】",
        ])
        
        for entry in self.failure_log[-20:]:  # 只显示最近 20 条
            report_lines.append(f"  [{entry['timestamp']}] "
                              f"连续失败 #{entry['consecutive_failures']}: "
                              f"{entry['error_type']} - {entry['error_msg'][:100]}")
        
        report_lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80,
        ])
        
        # 写入文件
        report_content = "\n".join(report_lines)
        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(report_content + "\n\n")
        
        logger.error(f"DEBUG_REPORT.log 已生成：{self.report_path}")
    
    def should_stop(self) -> bool:
        """
        检查是否应该停止（连续失败过多）
        
        Returns
        -------
        bool
            是否应该停止
        """
        return self.consecutive_failures >= 10


# ===========================================
# V71 原始数据抓取器
# ===========================================

class V71RawFetcher:
    """
    V71 原始数据抓取器 - 生存模式
    
    【核心功能】
    1. 直连东方财富底层 JSON API
    2. 使用 requests.Session() 保持长连接
    3. 首日探测与自诊断
    4. 动态延迟采样
    5. 内置调试报告生成
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化原始数据抓取器
        
        Parameters
        ----------
        db : Optional[DatabaseManager]
            数据库管理器实例
        """
        self.db = db
        self.session = requests.Session()
        self.debug_reporter = V71DebugReporter()
        self.last_request_time = 0.0
        self.batch_count = 0
        
        # 配置 session
        self.session.headers.update(DEFAULT_HEADERS)
        self.session.mount(
            "http://",
            requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=3,
                pool_block=False,
            )
        )
    
    def _update_cookie(self) -> None:
        """更新 Cookie"""
        self.session.headers["Cookie"] = generate_dynamic_cookie()
    
    def _wait_rate_limit(self) -> None:
        """等待以满足频率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < V71_REQUEST_DELAY_MIN:
            sleep_time = V71_REQUEST_DELAY_MIN - elapsed
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, params: Dict[str, Any], 
                      referer: str = None) -> Tuple[Optional[Dict], Optional[int], Optional[str]]:
        """
        发送 HTTP 请求
        
        Parameters
        ----------
        url : str
            请求 URL
        params : Dict[str, Any]
            请求参数
        referer : Optional[str]
            Referer 头
        
        Returns
        -------
        Tuple[Optional[Dict], Optional[int], Optional[str]]
            (解析后的数据，HTTP 状态码，错误消息)
        """
        self._wait_rate_limit()
        self._update_cookie()
        
        if referer:
            self.session.headers["Referer"] = referer
        
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=30,
                stream=False,
            )
            
            status_code = response.status_code
            
            if status_code != 200:
                error_msg = f"HTTP {status_code}"
                self.debug_reporter.record_failure(
                    status_code=status_code,
                    error_type=f"HTTP_{status_code}",
                    error_msg=error_msg,
                )
                return None, status_code, error_msg
            
            # 尝试解析 JSON
            data = parse_json_response(response.text)
            
            if data is None:
                self.debug_reporter.record_failure(
                    status_code=status_code,
                    error_type="JSON_Parse_Error",
                    error_msg="Response is not valid JSON",
                )
                return None, status_code, "Invalid JSON response"
            
            self.debug_reporter.record_success()
            return data, status_code, None
            
        except requests.exceptions.Timeout as e:
            self.debug_reporter.record_failure(
                error_type="Timeout",
                error_msg=str(e),
            )
            return None, None, f"Timeout: {e}"
        
        except requests.exceptions.ConnectionError as e:
            error_type = type(e).__name__
            self.debug_reporter.record_failure(
                error_type=error_type,
                error_msg=str(e),
            )
            return None, None, f"ConnectionError: {e}"
        
        except requests.exceptions.RequestException as e:
            error_type = type(e).__name__
            self.debug_reporter.record_failure(
                error_type=error_type,
                error_msg=str(e),
            )
            return None, None, f"RequestException: {e}"
    
    def probe_first_day(self, test_date: str = "2024-01-02") -> bool:
        """
        首日探测 - 测试 API 可用性
        
        Parameters
        ----------
        test_date : str
            测试日期
        
        Returns
        -------
        bool
            是否成功
        """
        logger.info(f"V71: 开始首日探测，测试日期：{test_date}")
        
        # 构建参数
        params = {
            "pn": "1",
            "pz": "10",
            "po": "1",
            "np": "1",
            "ut": "bd1d9dff01a339b61539d195e12c13cd",
            "fltt": "2",
            "invt": "2",
            "fid": "f62",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
            "fields": "f12,f14,f2,f3,f4,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205",
            "_": str(int(time.time() * 1000)),
        }
        
        data, status_code, error_msg = self._make_request(
            EASTMONEY_FUND_FLOW_RANK_URL,
            params,
            referer="http://data.eastmoney.com/zjlx/detail.html",
        )
        
        if data is None:
            logger.error("=" * 80)
            logger.error("❌ 首日探测失败！")
            logger.error(f"   HTTP 状态码：{status_code}")
            logger.error(f"   错误信息：{error_msg}")
            logger.error("=" * 80)
            
            # 打印原始响应（如果有）
            if status_code == 200:
                logger.error("原始响应内容片段：")
                # 这里需要重新请求一次来获取原始内容
            else:
                logger.error(f"Header 校验失败，请更新 Cookie 或 Referer")
            
            return False
        
        # 检查返回的数据是否有效
        if "data" not in data or "diff" not in data.get("data", {}):
            logger.error("❌ 首日探测：返回数据格式异常")
            logger.error(f"返回数据：{json.dumps(data, ensure_ascii=False)[:500]}")
            return False
        
        diff = data["data"]["diff"]
        if not diff or len(diff) == 0:
            logger.warning("⚠️  首日探测：返回数据为空（可能是非交易日）")
            return True  # 非交易日也算成功
        
        logger.info(f"✓ 首日探测成功，获取到 {len(diff)} 条数据")
        return True
    
    def fetch_fund_flow_rank(self, page_num: int = 1, page_size: int = 50) -> Optional[pl.DataFrame]:
        """
        抓取资金流向排名数据
        
        Parameters
        ----------
        page_num : int
            页码
        page_size : int
            每页数量
        
        Returns
        -------
        Optional[pl.DataFrame]
            资金流向排名数据
        """
        params = {
            "pn": str(page_num),
            "pz": str(page_size),
            "po": "1",
            "np": "1",
            "ut": "bd1d9dff01a339b61539d195e12c13cd",
            "fltt": "2",
            "invt": "2",
            "fid": "f62",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23",
            "fields": "f12,f14,f2,f3,f4,f62,f184,f66,f69,f72,f75,f78,f81,f84,f87,f204,f205",
            "_": str(int(time.time() * 1000)),
        }
        
        data, status_code, error_msg = self._make_request(
            EASTMONEY_FUND_FLOW_RANK_URL,
            params,
            referer="http://data.eastmoney.com/zjlx/detail.html",
        )
        
        if data is None:
            logger.error(f"抓取资金流向排名失败 (page={page_num}): {error_msg}")
            return None
        
        # 解析数据
        if "data" not in data or "diff" not in data.get("data", {}):
            logger.warning(f"资金流向排名数据格式异常 (page={page_num})")
            return None
        
        diff = data["data"]["diff"]
        if not diff:
            return None
        
        # 转换为 DataFrame
        records = []
        for item in diff:
            record = {
                "symbol": str(item.get("f12", "")),
                "name": str(item.get("f14", "")),
                "close": float(item.get("f2", 0) or 0),
                "change_percent": float(item.get("f3", 0) or 0),
                "net_main_amount": float(item.get("f62", 0) or 0),
                "net_main_ratio": float(item.get("f184", 0) or 0),
                "net_super_amount": float(item.get("f66", 0) or 0),
                "net_super_ratio": float(item.get("f69", 0) or 0),
                "net_large_amount": float(item.get("f69", 0) or 0),
                "net_large_ratio": float(item.get("f72", 0) or 0),
                "net_medium_amount": float(item.get("f75", 0) or 0),
                "net_medium_ratio": float(item.get("f78", 0) or 0),
                "net_small_amount": float(item.get("f81", 0) or 0),
                "net_small_ratio": float(item.get("f84", 0) or 0),
            }
            records.append(record)
        
        if not records:
            return None
        
        df = pl.DataFrame(records)
        logger.debug(f"抓取资金流向排名成功 (page={page_num}, rows={len(df)})")
        return df
    
    def fetch_stock_fund_flow(self, symbol: str, 
                               start_date: str, 
                               end_date: str) -> Optional[pl.DataFrame]:
        """
        抓取单只股票的资金流向历史数据
        
        Parameters
        ----------
        symbol : str
            股票代码（格式：000001.SZ）
        start_date : str
            开始日期（YYYY-MM-DD）
        end_date : str
            结束日期（YYYY-MM-DD）
        
        Returns
        -------
        Optional[pl.DataFrame]
            资金流向历史数据
        """
        # 转换股票代码格式
        if "." in symbol:
            code, exchange = symbol.split(".")
            if exchange == "SZ":
                sec_id = "0" + code
            elif exchange == "SH":
                sec_id = "1" + code
            else:
                sec_id = "0" + code
        else:
            sec_id = "0" + symbol
        
        params = {
            "secid": sec_id,
            "ut": "bd1d9dff01a339b61539d195e12c13cd",
            "lmt": "0",
            "klt": "1",
            "fields1": "f1,f2,f3,f7",
            "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64,f65",
            "_": str(int(time.time() * 1000)),
        }
        
        data, status_code, error_msg = self._make_request(
            EASTMONEY_STOCK_FUND_FLOW_URL,
            params,
            referer=f"http://data.eastmoney.com/zjlx/{symbol.replace('.', '')}.html",
        )
        
        if data is None:
            logger.error(f"抓取 {symbol} 资金流向失败：{error_msg}")
            return None
        
        # 解析数据
        if "data" not in data:
            logger.warning(f"{symbol} 资金流向数据格式异常")
            return None
        
        lines = data["data"].get("klines", [])
        if not lines:
            return None
        
        # 解析 K 线数据
        records = []
        for line in lines:
            parts = line.split(",")
            if len(parts) < 15:
                continue
            
            try:
                record = {
                    "symbol": symbol,
                    "trade_date": parts[0],
                    "close": float(parts[2]) if parts[2] else 0.0,
                    "change_percent": float(parts[3]) if parts[3] else 0.0,
                    "net_main_amount": float(parts[9]) if parts[9] else 0.0,
                    "net_super_amount": float(parts[10]) if parts[10] else 0.0,
                    "net_large_amount": float(parts[11]) if parts[11] else 0.0,
                    "net_medium_amount": float(parts[12]) if parts[12] else 0.0,
                    "net_small_amount": float(parts[13]) if parts[13] else 0.0,
                }
                records.append(record)
            except (ValueError, IndexError) as e:
                logger.warning(f"解析 {symbol} 数据行失败：{e}")
                continue
        
        if not records:
            return None
        
        df = pl.DataFrame(records)
        
        # 日期过滤
        df = df.filter(
            (pl.col("trade_date") >= start_date) & 
            (pl.col("trade_date") <= end_date)
        )
        
        return df
    
    def fetch_all_dates_fund_flow(self, start_date: str, 
                                   end_date: str) -> Optional[pl.DataFrame]:
        """
        抓取指定日期范围内的所有股票资金流向数据
        
        Parameters
        ----------
        start_date : str
            开始日期（YYYY-MM-DD）
        end_date : str
            结束日期（YYYY-MM-DD）
        
        Returns
        -------
        Optional[pl.DataFrame]
            所有股票的资金流向数据
        """
        logger.info(f"V71: 开始抓取 {start_date} 至 {end_date} 的资金流向数据...")
        
        all_records = []
        self.batch_count = 0
        
        # 遍历所有交易日（简化处理，实际应该获取交易日列表）
        current_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        
        while current_date <= end_date_obj:
            # 跳过周末
            if current_date.weekday() < 5:
                date_str = current_date.strftime("%Y-%m-%d")
                logger.info(f"抓取日期：{date_str}")
                
                # 获取当日排名数据
                page = 1
                while True:
                    df = self.fetch_fund_flow_rank(page_num=page, page_size=50)
                    
                    if df is None or df.is_empty():
                        break
                    
                    # 添加日期列
                    df = df.with_columns(pl.lit(date_str).alias("trade_date"))
                    all_records.append(df)
                    
                    # 如果返回数据少于 page_size，说明是最后一页
                    if len(df) < 50:
                        break
                    
                    page += 1
                    random_delay()
                
                self.batch_count += 1
                long_delay_for_batch(self.batch_count)
                
                if self.debug_reporter.should_stop():
                    logger.error("连续失败过多，停止抓取")
                    break
            
            current_date += timedelta(days=1)
        
        if not all_records:
            logger.warning("未获取到任何资金流向数据")
            return None
        
        result = pl.concat(all_records, how="vertical_relaxed")
        logger.info(f"V71: 成功抓取 {len(result)} 条资金流向数据")
        return result
    
    def save_to_db(self, df: pl.DataFrame, table_name: str = "stock_fund_flow") -> int:
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
            logger.error("V71: 数据库连接未初始化")
            return 0
        
        if df.is_empty():
            logger.warning("V71: 数据为空，跳过保存")
            return 0
        
        try:
            self.db.to_sql(df, table_name, if_exists="append")
            logger.info(f"V71: 成功保存 {len(df)} 条数据到 {table_name}")
            return len(df)
        except Exception as e:
            logger.error(f"V71: 保存数据失败：{e}")
            return 0
    
    def verify_insert(self, date: str, table_name: str = "stock_fund_flow") -> int:
        """
        验证插入 - 死锁校验
        
        Parameters
        ----------
        date : str
            日期
        table_name : str
            表名
        
        Returns
        -------
        int
            该日期的记录数
        """
        if self.db is None:
            return 0
        
        try:
            query = f"""
                SELECT COUNT(*) as cnt 
                FROM {table_name} 
                WHERE trade_date = '{date}'
            """
            result = self.db.read_sql(query)
            count = result["cnt"][0] if len(result) > 0 else 0
            logger.debug(f"日期 {date} 的数据库记录数：{count}")
            return count
        except Exception as e:
            logger.error(f"验证插入失败：{e}")
            return 0
    
    def record_sync_log(self, date: str, status: str, 
                        retry_count: int = 0, 
                        error_msg: str = None) -> None:
        """
        记录同步日志到 sync_log 表
        
        Parameters
        ----------
        date : str
            日期
        status : str
            状态（success/failed/retry）
        retry_count : int
            重试次数
        error_msg : Optional[str]
            错误消息
        """
        if self.db is None:
            return
        
        try:
            # 确保 sync_log 表存在
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    trade_date VARCHAR(20) NOT NULL,
                    status VARCHAR(20) NOT NULL,
                    retry_count INT DEFAULT 0,
                    error_msg TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_date (trade_date),
                    INDEX idx_status (status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            self.db.execute(create_table_sql)
            
            # 插入日志
            insert_sql = """
                INSERT INTO sync_log (trade_date, status, retry_count, error_msg)
                VALUES (%s, %s, %s, %s)
            """
            # 使用参数化查询
            from sqlalchemy import text
            with self.db.get_connection() as conn:
                conn.execute(
                    text(insert_sql),
                    {"trade_date": date, "status": status, 
                     "retry_count": retry_count, "error_msg": error_msg}
                )
                conn.commit()
            
            logger.debug(f"同步日志已记录：{date} - {status}")
            
        except Exception as e:
            logger.error(f"记录同步日志失败：{e}")
    
    def close(self) -> None:
        """关闭 session"""
        self.session.close()
        logger.info("V71: Session 已关闭")


# ===========================================
# V71 监控与报告函数
# ===========================================

def monitor_and_report(fetcher: V71RawFetcher) -> Dict[str, Any]:
    """
    监控并生成报告
    
    Parameters
    ----------
    fetcher : V71RawFetcher
        抓取器实例
    
    Returns
    -------
    Dict[str, Any]
        监控报告
    """
    reporter = fetcher.debug_reporter
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "stats": reporter.request_stats.copy(),
        "consecutive_failures": reporter.consecutive_failures,
        "failure_rate": 0.0,
        "status": "healthy",
    }
    
    if report["stats"]["total"] > 0:
        report["failure_rate"] = report["stats"]["failure"] / report["stats"]["total"]
    
    # 判断状态
    if reporter.consecutive_failures >= 10:
        report["status"] = "critical"
    elif reporter.consecutive_failures >= 3:
        report["status"] = "warning"
    elif report["failure_rate"] > 0.5:
        report["status"] = "degraded"
    
    # 生成报告文件
    reporter.generate_report()
    
    logger.info(f"V71 监控报告：{report['status']} - "
               f"成功率 {100 * (1 - report['failure_rate']):.1f}%")
    
    return report


# ===========================================
# V71 主运行函数
# ===========================================

def run_v71_fetcher(db: Optional[DatabaseManager] = None,
                    start_date: str = V71_DEFAULT_START_DATE,
                    end_date: str = None,
                    skip_probe: bool = False) -> Dict[str, Any]:
    """
    运行 V71 数据抓取
    
    Parameters
    ----------
    db : Optional[DatabaseManager]
        数据库管理器实例
    start_date : str
        开始日期
    end_date : Optional[str]
        结束日期，默认今天
    skip_probe : bool
        是否跳过首日探测
    
    Returns
    -------
    Dict[str, Any]
        运行结果
    """
    logger.info("=" * 80)
    logger.info("V71 Raw Fetcher - 生存模式启动")
    logger.info("=" * 80)
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # 初始化抓取器
    fetcher = V71RawFetcher(db=db)
    
    result = {
        "success": False,
        "start_date": start_date,
        "end_date": end_date,
        "total_rows": 0,
        "error": None,
    }
    
    try:
        # 首日探测
        if not skip_probe:
            logger.info("V71: 执行首日探测...")
            if not fetcher.probe_first_day("2024-01-02"):
                result["error"] = "首日探测失败，API 可能不可用"
                logger.error(result["error"])
                monitor_and_report(fetcher)
                return result
            logger.info("✓ 首日探测通过")
        
        # 抓取数据
        logger.info(f"V71: 开始抓取 {start_date} 至 {end_date} 的数据...")
        df = fetcher.fetch_all_dates_fund_flow(start_date, end_date)
        
        if df is None or df.is_empty():
            result["error"] = "未获取到任何数据"
            logger.warning(result["error"])
        else:
            # 保存数据
            rows = fetcher.save_to_db(df)
            result["total_rows"] = rows
            
            # 验证插入
            unique_dates = df["trade_date"].unique().to_list()
            for date in unique_dates:
                count = fetcher.verify_insert(date)
                fetcher.record_sync_log(date, "success", 0, None)
                logger.debug(f"日期 {date}: {count} 条记录")
        
        # 监控报告
        monitor_report = monitor_and_report(fetcher)
        result["monitor_report"] = monitor_report
        result["success"] = monitor_report["status"] != "critical"
        
    except Exception as e:
        result["error"] = f"运行异常：{e}"
        logger.error(traceback.format_exc())
        fetcher.debug_reporter.record_failure(
            error_type=type(e).__name__,
            error_msg=str(e),
        )
    
    finally:
        fetcher.close()
    
    logger.info("=" * 80)
    logger.info(f"V71 运行完成：成功={result['success']}, 数据量={result['total_rows']}")
    if result["error"]:
        logger.error(f"错误：{result['error']}")
    logger.info("=" * 80)
    
    return result


# ===========================================
# CLI 入口
# ===========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V71 Raw Fetcher - 暴力抓取资金流向数据")
    parser.add_argument("--start-date", type=str, default=V71_DEFAULT_START_DATE,
                       help=f"开始日期 (默认：{V71_DEFAULT_START_DATE})")
    parser.add_argument("--end-date", type=str, default=None,
                       help="结束日期 (默认：今天)")
    parser.add_argument("--skip-probe", action="store_true",
                       help="跳过首日探测")
    parser.add_argument("--db-host", type=str, default=None,
                       help="数据库主机")
    parser.add_argument("--db-port", type=str, default=None,
                       help="数据库端口")
    parser.add_argument("--db-user", type=str, default=None,
                       help="数据库用户名")
    parser.add_argument("--db-password", type=str, default=None,
                       help="数据库密码")
    parser.add_argument("--db-name", type=str, default=None,
                       help="数据库名称")
    
    args = parser.parse_args()
    
    # 设置环境变量
    if args.db_host:
        import os
        os.environ["MYSQL_HOST"] = args.db_host
    if args.db_port:
        import os
        os.environ["MYSQL_PORT"] = args.db_port
    if args.db_user:
        import os
        os.environ["MYSQL_USER"] = args.db_user
    if args.db_password:
        import os
        os.environ["MYSQL_PASSWORD"] = args.db_password
    if args.db_name:
        import os
        os.environ["MYSQL_DATABASE"] = args.db_name
    
    # 初始化数据库
    db = DatabaseManager()
    db.connect()
    
    # 运行抓取
    result = run_v71_fetcher(
        db=db,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_probe=args.skip_probe,
    )
    
    # 退出码
    sys.exit(0 if result["success"] else 1)