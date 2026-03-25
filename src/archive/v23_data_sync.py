"""
V23 Data Sync Module - 自动数据闭环与智能自愈

【核心功能】
1. 数据缺失自检 (check_market_data): 在回测启动时自动检测数据完整性
2. 自动补齐：检测到 000300.SH 或个股数据缺失时，立即执行数据抓取
3. 数据对齐：确保指数数据与个股交易日完全对齐

【严防甩锅】
- 检测到数据缺失时，严禁报错停止
- 必须立即生成并执行数据抓取代码
- 确保回测所需的全部时间序列完整

作者：资深量化系统架构师 (V23: 自动数据闭环与智能自愈)
版本：V23.0 (Data Auto-Healing)
日期：2026-03-18
"""

import sys
import time
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Set, Tuple

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================
DEFAULT_LOOKBACK_DAYS = 730  # 2 年
COMMIT_INTERVAL = 50  # 每 50 只股票 commit 一次
MAX_RETRIES = 3  # 最大重试次数
RANDOM_DELAY_MIN = 0.2  # 随机延迟最小值 (秒)
RANDOM_DELAY_MAX = 0.5  # 随机延迟最大值 (秒)

# 必需指数列表
REQUIRED_INDICES = [
    "000300.SH",  # 沪深 300 - 用于大盘风控
    "000905.SH",  # 中证 500
    "000001.SH",  # 上证指数
]

# 内置备用成分股列表（20 只蓝筹股）
FALLBACK_STOCKS = [
    {"symbol": "600519.SH", "name": "贵州茅台"},
    {"symbol": "300750.SZ", "name": "宁德时代"},
    {"symbol": "000858.SZ", "name": "五粮液"},
    {"symbol": "601318.SH", "name": "中国平安"},
    {"symbol": "600036.SH", "name": "招商银行"},
    {"symbol": "000333.SZ", "name": "美的集团"},
    {"symbol": "002415.SZ", "name": "海康威视"},
    {"symbol": "601888.SH", "name": "中国中免"},
    {"symbol": "600276.SH", "name": "恒瑞医药"},
    {"symbol": "601166.SH", "name": "兴业银行"},
    {"symbol": "000001.SZ", "name": "平安银行"},
    {"symbol": "000002.SZ", "name": "万科 A"},
    {"symbol": "600030.SH", "name": "中信证券"},
    {"symbol": "000651.SZ", "name": "格力电器"},
    {"symbol": "000725.SZ", "name": "京东方 A"},
    {"symbol": "002594.SZ", "name": "比亚迪"},
    {"symbol": "300059.SZ", "name": "东方财富"},
    {"symbol": "601398.SH", "name": "工商银行"},
    {"symbol": "601988.SH", "name": "中国银行"},
    {"symbol": "601857.SH", "name": "中国石油"},
]


# ===========================================
# 数据缺失自检与自愈
# ===========================================

class DataAutoHealer:
    """
    V23 数据自愈器 - 自动检测并补齐缺失数据
    
    【核心职责】
    1. 检查 000300.SH 指数数据是否完整
    2. 检查回测所需的个股数据是否完整
    3. 发现缺失时，立即执行数据抓取
    4. 确保指数与个股交易日完全对齐
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        self.synced_stocks: Set[str] = set()
        self.synced_indices: Set[str] = set()
        logger.info("DataAutoHealer initialized")
    
    def check_market_data(
        self,
        start_date: str,
        end_date: str,
        required_stocks: Optional[List[str]] = None,
    ) -> Dict[str, any]:
        """
        检查市场数据完整性并自动补齐
        
        Args:
            start_date: 回测开始日期
            end_date: 回测结束日期
            required_stocks: 必需的股票列表（可选）
        
        Returns:
            检查结果字典
        """
        logger.info("=" * 70)
        logger.info("V23 DATA AUTO-HEALING CHECK")
        logger.info("=" * 70)
        logger.info(f"Checking data for period: {start_date} to {end_date}")
        
        result = {
            "index_status": {},
            "stock_status": {},
            "missing_indices": [],
            "missing_stocks": [],
            "aligned_trading_dates": [],
            "healing_actions": [],
        }
        
        # 1. 检查指数数据
        logger.info("\n[Step 1] Checking index data...")
        for index_symbol in REQUIRED_INDICES:
            status = self._check_index_data(index_symbol, start_date, end_date)
            result["index_status"][index_symbol] = status
            if not status["is_complete"]:
                result["missing_indices"].append(index_symbol)
        
        # 2. 检查个股数据
        logger.info("\n[Step 2] Checking stock data...")
        if required_stocks is None:
            required_stocks = self._get_required_stocks(start_date, end_date)
        
        for stock in required_stocks:
            symbol = stock["symbol"] if isinstance(stock, dict) else stock
            status = self._check_stock_data(symbol, start_date, end_date)
            result["stock_status"][symbol] = status
            if not status["is_complete"]:
                result["missing_stocks"].append(symbol)
        
        # 3. 对齐交易日
        logger.info("\n[Step 3] Aligning trading dates...")
        result["aligned_trading_dates"] = self._align_trading_dates(start_date, end_date)
        
        # 4. 输出检查结果
        logger.info("\n" + "=" * 70)
        logger.info("DATA CHECK RESULT")
        logger.info("=" * 70)
        
        # 指数状态
        logger.info("\nIndex Data Status:")
        for symbol, status in result["index_status"].items():
            icon = "✅" if status["is_complete"] else "❌"
            logger.info(f"  {icon} {symbol}: {status['available_days']}/{status['expected_days']} days")
        
        # 个股状态汇总
        total_stocks = len(result["stock_status"])
        complete_stocks = sum(1 for s in result["stock_status"].values() if s["is_complete"])
        logger.info(f"\nStock Data Status:")
        logger.info(f"  Total: {total_stocks}, Complete: {complete_stocks}, Missing: {total_stocks - complete_stocks}")
        
        # 5. 如果需要修复，执行自愈
        if result["missing_indices"] or result["missing_stocks"]:
            logger.info("\n" + "=" * 70)
            logger.info("DATA HEALING REQUIRED")
            logger.info("=" * 70)
            self._execute_healing(result, start_date, end_date)
        
        return result
    
    def _check_index_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, any]:
        """检查单个指数数据完整性"""
        try:
            query = f"""
                SELECT COUNT(DISTINCT trade_date) as cnt
                FROM index_daily
                WHERE symbol = '{symbol}'
                AND trade_date >= '{start_date}'
                AND trade_date <= '{end_date}'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return {
                    "is_complete": False,
                    "available_days": 0,
                    "expected_days": self._count_expected_days(start_date, end_date),
                    "reason": "No data found in database",
                }
            
            available_days = result["cnt"][0]
            expected_days = self._count_expected_days(start_date, end_date)
            
            # 允许 10% 的缺失（节假日等）
            is_complete = available_days >= expected_days * 0.9
            
            return {
                "is_complete": is_complete,
                "available_days": available_days,
                "expected_days": expected_days,
                "reason": "" if is_complete else f"Missing {expected_days - available_days} days",
            }
            
        except Exception as e:
            return {
                "is_complete": False,
                "available_days": 0,
                "expected_days": self._count_expected_days(start_date, end_date),
                "reason": str(e),
            }
    
    def _check_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, any]:
        """检查单个股票数据完整性"""
        try:
            query = f"""
                SELECT COUNT(DISTINCT trade_date) as cnt
                FROM stock_daily
                WHERE symbol = '{symbol}'
                AND trade_date >= '{start_date}'
                AND trade_date <= '{end_date}'
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return {
                    "is_complete": False,
                    "available_days": 0,
                    "expected_days": self._count_expected_days(start_date, end_date),
                    "reason": "No data found in database",
                }
            
            available_days = result["cnt"][0]
            expected_days = self._count_expected_days(start_date, end_date)
            
            # 允许 10% 的缺失
            is_complete = available_days >= expected_days * 0.9
            
            return {
                "is_complete": is_complete,
                "available_days": available_days,
                "expected_days": expected_days,
                "reason": "" if is_complete else f"Missing {expected_days - available_days} days",
            }
            
        except Exception as e:
            return {
                "is_complete": False,
                "available_days": 0,
                "expected_days": self._count_expected_days(start_date, end_date),
                "reason": str(e),
            }
    
    def _count_expected_days(self, start_date: str, end_date: str) -> int:
        """计算预期交易日数量（约 252 天/年）"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            total_days = (end - start).days
            # 估算交易日（约 252/365 = 0.69）
            return int(total_days * 0.69)
        except Exception:
            return 100  # 默认值
    
    def _get_required_stocks(
        self,
        start_date: str,
        end_date: str,
    ) -> List[Dict[str, str]]:
        """获取回测所需的股票列表"""
        # 默认使用内置的 20 只蓝筹股
        return FALLBACK_STOCKS.copy()
    
    def _align_trading_dates(
        self,
        start_date: str,
        end_date: str,
    ) -> List[str]:
        """对齐指数和个股的交易日"""
        try:
            # 获取 000300.SH 的交易日期
            query = f"""
                SELECT DISTINCT trade_date
                FROM index_daily
                WHERE symbol = '000300.SH'
                AND trade_date >= '{start_date}'
                AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                # 如果指数数据不存在，使用股票数据推断
                query = f"""
                    SELECT DISTINCT trade_date
                    FROM stock_daily
                    WHERE trade_date >= '{start_date}'
                    AND trade_date <= '{end_date}'
                    ORDER BY trade_date
                """
                result = self.db.read_sql(query)
            
            dates = sorted([str(d) for d in result["trade_date"].to_list()])
            logger.info(f"Aligned {len(dates)} trading dates")
            return dates
            
        except Exception as e:
            logger.warning(f"Failed to align trading dates: {e}")
            return []
    
    def _execute_healing(
        self,
        check_result: Dict[str, any],
        start_date: str,
        end_date: str,
    ):
        """执行数据自愈"""
        healing_actions = []
        
        # 1. 修复指数数据
        for symbol in check_result["missing_indices"]:
            logger.info(f"\n🔧 Healing index data: {symbol}")
            try:
                rows = self._fetch_and_sync_index(symbol, start_date, end_date)
                healing_actions.append({
                    "type": "index",
                    "symbol": symbol,
                    "rows_synced": rows,
                    "status": "success" if rows > 0 else "failed",
                })
                logger.info(f"  ✅ Synced {rows} rows for {symbol}")
            except Exception as e:
                logger.error(f"  ❌ Failed to sync {symbol}: {e}")
                healing_actions.append({
                    "type": "index",
                    "symbol": symbol,
                    "rows_synced": 0,
                    "status": "failed",
                    "error": str(e),
                })
        
        # 2. 修复个股数据
        for symbol in check_result["missing_stocks"]:
            logger.info(f"\n🔧 Healing stock data: {symbol}")
            try:
                rows = self._fetch_and_sync_stock(symbol, start_date, end_date)
                healing_actions.append({
                    "type": "stock",
                    "symbol": symbol,
                    "rows_synced": rows,
                    "status": "success" if rows > 0 else "failed",
                })
                logger.info(f"  ✅ Synced {rows} rows for {symbol}")
            except Exception as e:
                logger.error(f"  ❌ Failed to sync {symbol}: {e}")
                healing_actions.append({
                    "type": "stock",
                    "symbol": symbol,
                    "rows_synced": 0,
                    "status": "failed",
                    "error": str(e),
                })
        
        check_result["healing_actions"] = healing_actions
        
        # 3. 重新检查
        logger.info("\n" + "=" * 70)
        logger.info("RE-CHECKING DATA AFTER HEALING")
        logger.info("=" * 70)
        
        # 重新检查指数
        for symbol in check_result["missing_indices"]:
            status = self._check_index_data(symbol, start_date, end_date)
            check_result["index_status"][symbol] = status
            icon = "✅" if status["is_complete"] else "❌"
            logger.info(f"  {icon} {symbol}: {status['available_days']}/{status['expected_days']} days")
        
        # 重新检查个股
        for symbol in check_result["missing_stocks"]:
            status = self._check_stock_data(symbol, start_date, end_date)
            check_result["stock_status"][symbol] = status
    
    def _fetch_and_sync_index(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """抓取并同步指数数据"""
        try:
            import akshare as ak
            
            # 转换代码格式
            ak_code = self._convert_symbol_to_ak_code(symbol)
            
            # 日期格式转换
            start_ak = start_date.replace("-", "")
            end_ak = end_date.replace("-", "")
            
            # 获取数据
            df = ak.index_zh_a_hist(
                symbol=ak_code,
                period="daily",
                start_date=start_ak,
                end_date=end_ak,
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"No data fetched for {symbol}")
                return 0
            
            # 处理数据
            df_pl = self._process_index_data(df, symbol)
            
            if df_pl.is_empty():
                return 0
            
            # 写入数据库
            rows = self.db.to_sql(df_pl, "index_daily", if_exists="append")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to fetch index data for {symbol}: {e}")
            return 0
    
    def _fetch_and_sync_stock(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> int:
        """抓取并同步个股数据"""
        try:
            import akshare as ak
            
            # 提取纯代码
            pure_code = symbol.split(".")[0]
            
            # 日期格式转换
            start_ak = start_date.replace("-", "")
            end_ak = end_date.replace("-", "")
            
            # 获取数据
            df = ak.stock_zh_a_hist(
                symbol=pure_code,
                period="daily",
                start_date=start_ak,
                end_date=end_ak,
                adjust="qfq",
            )
            
            if df is None or len(df) == 0:
                logger.warning(f"No data fetched for {symbol}")
                return 0
            
            # 处理数据
            df_pl = self._process_stock_data(df, symbol)
            
            if df_pl.is_empty():
                return 0
            
            # 写入数据库
            rows = self.db.to_sql(df_pl, "stock_daily", if_exists="append")
            return rows
            
        except Exception as e:
            logger.error(f"Failed to fetch stock data for {symbol}: {e}")
            return 0
    
    def _convert_symbol_to_ak_code(self, symbol: str) -> str:
        """将标准 symbol 转换为 AKShare 代码格式"""
        if symbol.endswith(".SH"):
            code = symbol.replace(".SH", "")
            return f"sh{code}"
        elif symbol.endswith(".SZ"):
            code = symbol.replace(".SZ", "")
            return f"sz{code}"
        return symbol
    
    def _process_index_data(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """处理指数数据"""
        df_pl = pl.from_pandas(df) if hasattr(df, 'columns') and hasattr(df, 'to_pandas') else pl.from_pandas(df)
        
        # 列名映射
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "昨收": "pre_close",
            "涨跌额": "change",
            "涨跌幅": "pct_chg",
        }
        
        for old, new in rename_map.items():
            if old in df_pl.columns:
                df_pl = df_pl.rename({old: new})
        
        # 添加 symbol 列
        df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
        
        # 日期格式标准化
        if "trade_date" in df_pl.columns:
            def format_date(x):
                if x is None:
                    return None
                s = str(x).strip().replace('-', '')
                if len(s) >= 8 and s[:8].isdigit():
                    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                return s
            
            df_pl = df_pl.with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8, strict=False)
                .map_elements(format_date, return_dtype=pl.Utf8)
                .alias("trade_date")
            )
        
        # 补齐缺失字段
        if "pre_close" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col("close").shift(1).over("symbol").alias("pre_close")
            )
        
        if "change" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                (pl.col("close") - pl.col("pre_close")).alias("change")
            )
        
        if "pct_chg" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                ((pl.col("close") / pl.col("pre_close") - 1) * 100).alias("pct_chg")
            )
        
        # 选择目标列
        target_columns = [
            "symbol", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "volume", "amount"
        ]
        
        available_columns = [c for c in target_columns if c in df_pl.columns]
        df_pl = df_pl.select(available_columns)
        
        # 类型转换
        numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount"]
        for col in numeric_columns:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        return df_pl.sort(["symbol", "trade_date"])
    
    def _process_stock_data(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """处理个股数据"""
        df_pl = pl.from_pandas(df) if hasattr(df, 'columns') and hasattr(df, 'to_pandas') else pl.from_pandas(df)
        
        # 列名映射
        rename_map = {
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "振幅": "amplitude",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
            "换手率": "turnover_rate",
        }
        
        for old, new in rename_map.items():
            if old in df_pl.columns:
                df_pl = df_pl.rename({old: new})
        
        # 添加 symbol 列
        df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
        
        # 日期格式标准化
        if "trade_date" in df_pl.columns:
            def format_date(x):
                if x is None:
                    return None
                s = str(x).strip().replace('-', '')
                if len(s) >= 8 and s[:8].isdigit():
                    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                return s
            
            df_pl = df_pl.with_columns(
                pl.col("trade_date")
                .cast(pl.Utf8, strict=False)
                .map_elements(format_date, return_dtype=pl.Utf8)
                .alias("trade_date")
            )
        
        # 补齐缺失字段
        if "pre_close" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col("close").shift(1).over("symbol").alias("pre_close")
            )
        
        if "change" not in df_pl.columns:
            df_pl = df_pl.with_columns(
                (pl.col("close") - pl.col("pre_close")).alias("change")
            )
        
        if "turnover_rate" not in df_pl.columns:
            df_pl = df_pl.with_columns(pl.lit(1.0).alias("turnover_rate"))
        
        # 选择目标列
        target_columns = [
            "symbol", "trade_date", "open", "high", "low", "close",
            "pre_close", "change", "pct_chg", "volume", "amount",
            "turnover_rate"
        ]
        
        available_columns = [c for c in target_columns if c in df_pl.columns]
        df_pl = df_pl.select(available_columns)
        
        # 类型转换
        numeric_columns = ["open", "high", "low", "close", "pre_close", "change", "pct_chg", "volume", "amount", "turnover_rate"]
        for col in numeric_columns:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        return df_pl.sort(["symbol", "trade_date"])


# ===========================================
# 便捷函数
# ===========================================

def check_market_data(
    start_date: str,
    end_date: str,
    required_stocks: Optional[List[str]] = None,
    db: Optional[DatabaseManager] = None,
) -> Dict[str, any]:
    """
    检查市场数据完整性并自动补齐
    
    这是 V23 回测启动时的必需检查函数
    
    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        required_stocks: 必需的股票列表（可选）
        db: 数据库管理器（可选）
    
    Returns:
        检查结果字典
    """
    healer = DataAutoHealer(db=db)
    return healer.check_market_data(start_date, end_date, required_stocks)


def ensure_data_complete(
    start_date: str,
    end_date: str,
    db: Optional[DatabaseManager] = None,
) -> bool:
    """
    确保数据完整性的便捷函数
    
    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        db: 数据库管理器
    
    Returns:
        True 如果数据完整，False 如果需要修复
    """
    result = check_market_data(start_date, end_date, db=db)
    
    # 检查是否所有数据都完整
    all_complete = all(
        status["is_complete"]
        for status in result["index_status"].values()
    )
    
    return all_complete


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V23 DATA SYNC - Auto-Healing Mode")
    logger.info("=" * 70)
    
    # 默认参数
    start_date = "2025-01-01"
    end_date = "2026-03-17"
    
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    
    logger.info(f"Period: {start_date} to {end_date}")
    
    # 执行数据检查与自愈
    result = check_market_data(start_date, end_date)
    
    # 输出结果
    logger.info("\n" + "=" * 70)
    logger.info("V23 DATA SYNC COMPLETE")
    logger.info("=" * 70)
    
    # 统计修复动作
    healing_actions = result.get("healing_actions", [])
    success_count = sum(1 for a in healing_actions if a.get("status") == "success")
    fail_count = sum(1 for a in healing_actions if a.get("status") == "failed")
    
    logger.info(f"Healing Actions: {len(healing_actions)}")
    logger.info(f"  ✅ Success: {success_count}")
    logger.info(f"  ❌ Failed: {fail_count}")


if __name__ == "__main__":
    main()