"""
Sync Industry and Market Cap Data - 使用 Tushare 获取申万一级行业和总市值数据。

核心功能:
    - 使用 Tushare 获取股票行业分类（industry 字段）
    - 获取总市值数据（total_mv 字段）
    - 批量更新 stock_daily 表的 industry_code 和 total_mv 字段
    - 严格的数据完整性校验（缺失率 > 1% 则退出）

使用示例:
    >>> python src/sync_industry_and_market_cap.py
"""

import sys
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


# ===========================================
# 配置常量
# ===========================================
COMMIT_INTERVAL = 500  # 每 500 只股票 commit 一次
MAX_RETRIES = 3  # 最大重试次数
RANDOM_DELAY_MIN = 0.2  # 随机延迟最小值 (秒)
RANDOM_DELAY_MAX = 0.5  # 随机延迟最大值 (秒)
INDUSTRY_MISSING_THRESHOLD = 0.01  # 行业数据缺失率阈值 (1%)
MARKET_CAP_MISSING_THRESHOLD = 0.01  # 市值数据缺失率阈值 (1%)


@dataclass
class SyncResult:
    """同步结果数据结构"""
    industry_success_count: int = 0
    industry_fail_count: int = 0
    total_stocks_updated: int = 0
    industry_coverage_rate: float = 0.0
    market_cap_coverage_rate: float = 0.0


class IndustryAndMarketCapSyncer:
    """
    行业和市值数据同步器 - 使用 Tushare 获取行业分类和总市值数据。
    
    数据源说明:
        - 行业分类：使用 Tushare stock_basic 接口的 industry 字段
        - 总市值：使用 Tushare daily_basic 接口的 total_mv 字段
    
    同步策略:
        1. 从 Tushare 获取所有股票的基本信息（含行业分类）
        2. 从 Tushare 获取所有股票的总市值
        3. 批量更新 stock_daily 表
        4. 物理校验：缺失率必须 < 1%，否则退出
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化同步器。
        
        Args:
            db: 数据库管理器实例
        """
        self.db = db or DatabaseManager.get_instance()
        self.result = SyncResult()
    
    def fetch_all_stock_info_from_tushare(self) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame]]:
        """
        从 Tushare 获取所有股票的基本信息和市值数据。
        
        Returns:
            (基本信息 DataFrame, 市值数据 DataFrame)
        """
        import tushare as ts
        
        try:
            logger.info("Fetching stock basic info from Tushare...")
            pro = ts.pro_api()
            
            # 获取股票基本信息
            basic_df = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,symbol,name,area,industry,market,list_date'
            )
            
            if basic_df is None or len(basic_df) == 0:
                logger.error("Failed to fetch stock basic info")
                return None, None
            
            basic_pl = pl.from_pandas(basic_df)
            
            logger.success(f"Fetched {len(basic_pl)} stocks basic info")
            
            # 获取市值数据 - 使用 daily_basic 获取最新市值
            logger.info("Fetching latest market cap data from Tushare...")
            
            # 获取最近 30 天的市值数据
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y%m%d")
            
            mv_df = pro.daily_basic(
                exchange='',
                start_date=start_date,
                end_date=end_date,
                fields='ts_code,trade_date,total_mv'
            )
            
            if mv_df is not None and len(mv_df) > 0:
                mv_pl = pl.from_pandas(mv_df)
                logger.success(f"Fetched {len(mv_pl)} market cap records")
                return basic_pl, mv_pl
            else:
                logger.warning("Failed to fetch market cap data")
                return basic_pl, None
            
        except Exception as e:
            logger.error(f"Failed to fetch data from Tushare: {e}")
            return None, None
    
    def batch_update_industry_and_mv(self, basic_df: pl.DataFrame, mv_df: Optional[pl.DataFrame] = None) -> int:
        """
        批量更新行业和市值数据。
        
        Args:
            basic_df: 股票基本信息 DataFrame
            mv_df: 市值数据 DataFrame（可选）
            
        Returns:
            更新的行数
        """
        total_updated = 0
        
        # 按 ts_code 分组，获取最新的市值
        if mv_df is not None:
            # 按 ts_code 和 trade_date 排序，取最新的市值
            mv_latest = mv_df.sort("trade_date", descending=True).group_by("ts_code").first()
        else:
            mv_latest = None
        
        # 遍历股票列表
        stocks = basic_df.to_dicts()
        
        for i, stock in enumerate(stocks, 1):
            ts_code = stock.get("ts_code", "")
            symbol = stock.get("symbol", "")
            industry = stock.get("industry", "")
            
            # 确定股票代码格式
            if not ts_code:
                if symbol:
                    if symbol.startswith("6"):
                        ts_code = f"{symbol}.SH"
                    else:
                        ts_code = f"{symbol}.SZ"
                else:
                    continue
            
            # 获取市值
            market_cap = None
            if mv_latest is not None:
                mv_row = mv_latest.filter(pl.col("ts_code") == ts_code)
                if len(mv_row) > 0:
                    market_cap = mv_row["total_mv"][0]
                    try:
                        # 转换为亿元
                        market_cap = float(market_cap) / 100000000
                    except (ValueError, TypeError):
                        market_cap = None
            
            # 构建 UPDATE 语句
            set_parts = []
            
            if industry:
                set_parts.append(f"`industry_code` = '{industry}'")
            else:
                set_parts.append("`industry_code` = NULL")
            
            if market_cap is not None:
                set_parts.append(f"`total_mv` = {market_cap}")
            else:
                set_parts.append("`total_mv` = NULL")
            
            if set_parts:
                update_sql = f"""
                    UPDATE `stock_daily`
                    SET {', '.join(set_parts)}
                    WHERE `symbol` = '{ts_code}'
                """
                
                try:
                    rows = self.db.execute(update_sql)
                    total_updated += rows
                except Exception as e:
                    logger.debug(f"Failed to update {ts_code}: {e}")
            
            # 进度显示
            if i % 100 == 0:
                progress = i / len(stocks) * 100
                logger.info(f"[{i}/{len(stocks)}] ({progress:.1f}%) Updated {total_updated} rows so far")
            
            # 定期 commit
            if i % COMMIT_INTERVAL == 0:
                self.db.execute("COMMIT")
                logger.info(f"📌 Checkpoint committed after {i} stocks")
            
            # 随机延迟
            if i < len(stocks):
                time.sleep(random.uniform(RANDOM_DELAY_MIN, RANDOM_DELAY_MAX))
        
        # 最终 commit
        self.db.execute("COMMIT")
        
        return total_updated
    
    def sync_all_data(self) -> int:
        """
        同步所有数据。
        
        Returns:
            更新的行数
        """
        logger.info("=" * 60)
        logger.info("INDUSTRY & MARKET CAP SYNC - 行业和市值数据同步")
        logger.info("=" * 60)
        
        # 获取数据
        basic_df, mv_df = self.fetch_all_stock_info_from_tushare()
        
        if basic_df is None:
            logger.error("Failed to fetch stock data")
            return 0
        
        logger.info(f"Processing {len(basic_df)} stocks...")
        
        # 批量更新
        total_updated = self.batch_update_industry_and_mv(basic_df, mv_df)
        
        self.result.total_stocks_updated = total_updated
        
        logger.info("=" * 60)
        logger.info(f"SYNC COMPLETE")
        logger.info(f"  Total rows updated: {total_updated}")
        logger.info("=" * 60)
        
        return total_updated
    
    def verify_data_completeness(self) -> Tuple[bool, Dict]:
        """
        验证数据完整性。
        
        Returns:
            (是否通过校验，验证结果字典)
        """
        logger.info("=" * 60)
        logger.info("DATA COMPLETENESS VERIFICATION - 数据完整性校验")
        logger.info("=" * 60)
        
        # 检查 industry_code 缺失率
        industry_query = """
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN `industry_code` IS NULL OR `industry_code` = '' OR `industry_code` = 'UNKNOWN' THEN 1 ELSE 0 END) as missing_count
            FROM stock_daily
        """
        industry_result = self.db.read_sql(industry_query)
        
        if industry_result.is_empty():
            logger.error("Failed to query industry_code statistics")
            return False, {"error": "Query failed"}
        
        industry_total = int(industry_result["total_count"][0])
        industry_missing = int(industry_result["missing_count"][0])
        industry_missing_rate = (industry_missing / industry_total * 100) if industry_total > 0 else 100.0
        
        logger.info(f"Industry Code Statistics:")
        logger.info(f"  Total records: {industry_total:,}")
        logger.info(f"  Missing records: {industry_missing:,}")
        logger.info(f"  Missing rate: {industry_missing_rate:.4f}%")
        
        # 检查 total_mv 缺失率
        mv_query = """
            SELECT 
                COUNT(*) as total_count,
                SUM(CASE WHEN `total_mv` IS NULL OR `total_mv` = 0 THEN 1 ELSE 0 END) as missing_count
            FROM stock_daily
        """
        mv_result = self.db.read_sql(mv_query)
        
        if mv_result.is_empty():
            logger.error("Failed to query total_mv statistics")
            return False, {"error": "Query failed"}
        
        mv_total = int(mv_result["total_count"][0])
        mv_missing = int(mv_result["missing_count"][0])
        mv_missing_rate = (mv_missing / mv_total * 100) if mv_total > 0 else 100.0
        
        logger.info(f"Market Cap Statistics:")
        logger.info(f"  Total records: {mv_total:,}")
        logger.info(f"  Missing records: {mv_missing:,}")
        logger.info(f"  Missing rate: {mv_missing_rate:.4f}%")
        
        # 物理校验：缺失率必须 < 1%
        industry_passed = industry_missing_rate < (INDUSTRY_MISSING_THRESHOLD * 100)
        mv_passed = mv_missing_rate < (MARKET_CAP_MISSING_THRESHOLD * 100)
        
        logger.info("=" * 60)
        logger.info("VERIFICATION RESULT")
        logger.info("=" * 60)
        
        if industry_passed:
            logger.success(f"✅ Industry code missing rate ({industry_missing_rate:.4f}%) < {INDUSTRY_MISSING_THRESHOLD * 100}%")
        else:
            logger.error(f"❌ Industry code missing rate ({industry_missing_rate:.4f}%) >= {INDUSTRY_MISSING_THRESHOLD * 100}%")
        
        if mv_passed:
            logger.success(f"✅ Market cap missing rate ({mv_missing_rate:.4f}%) < {MARKET_CAP_MISSING_THRESHOLD * 100}%")
        else:
            logger.error(f"❌ Market cap missing rate ({mv_missing_rate:.4f}%) >= {MARKET_CAP_MISSING_THRESHOLD * 100}%")
        
        verification_result = {
            "industry_code": {
                "total": industry_total,
                "missing": industry_missing,
                "missing_rate": industry_missing_rate,
                "threshold": INDUSTRY_MISSING_THRESHOLD * 100,
                "passed": industry_passed
            },
            "total_mv": {
                "total": mv_total,
                "missing": mv_missing,
                "missing_rate": mv_missing_rate,
                "threshold": MARKET_CAP_MISSING_THRESHOLD * 100,
                "passed": mv_passed
            },
            "overall_passed": industry_passed and mv_passed
        }
        
        return verification_result["overall_passed"], verification_result
    
    def generate_report(self, output_path: str = None) -> str:
        """
        生成同步报告。
        
        Args:
            output_path: 报告输出路径
            
        Returns:
            str: 报告内容
        """
        from pathlib import Path
        
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"sync_industry_mv_{timestamp}.md"
        
        # 获取验证结果
        passed, verification = self.verify_data_completeness()
        
        report_lines = [
            "# Industry and Market Cap Sync Report",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "## 同步结果汇总",
            f"",
        ]
        
        if passed:
            report_lines.append("✅ **数据完整性校验通过**")
        else:
            report_lines.append("❌ **数据完整性校验失败**")
        
        report_lines.append("")
        report_lines.append("### 行业数据同步")
        report_lines.append("")
        report_lines.append(f"- 成功处理的行业数：{self.result.industry_success_count}")
        report_lines.append(f"- 失败的行业数：{self.result.industry_fail_count}")
        report_lines.append(f"- 更新的记录数：{self.result.total_stocks_updated}")
        report_lines.append("")
        report_lines.append("### 市值数据同步")
        report_lines.append("")
        report_lines.append(f"- 总市值覆盖率：{verification.get('total_mv', {}).get('missing_rate', 100):.2f}%")
        report_lines.append("")
        report_lines.append("### 数据完整性校验")
        report_lines.append("")
        report_lines.append("| 字段 | 总记录数 | 缺失记录数 | 缺失率 | 阈值 | 状态 |")
        report_lines.append("|------|----------|------------|--------|------|------|")
        
        for field_name, field_data in verification.items():
            if field_name in ["industry_code", "total_mv"]:
                status = "✅" if field_data.get("passed", False) else "❌"
                report_lines.append(
                    f"| {field_name} | {field_data.get('total', 0):,} | "
                    f"{field_data.get('missing', 0):,} | {field_data.get('missing_rate', 0):.4f}% | "
                    f"{field_data.get('threshold', 0):.2f}% | {status} |"
                )
        
        report_lines.append("")
        report_lines.append("## 详细统计")
        report_lines.append("")
        report_lines.append("```json")
        import json
        report_lines.append(json.dumps(verification, indent=2, ensure_ascii=False))
        report_lines.append("```")
        report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"报告已保存至：{output_path}")
        return report_content
    
    def run_full_sync(self) -> bool:
        """
        运行完整同步流程。
        
        Returns:
            是否成功
        """
        logger.info("=" * 60)
        logger.info("FULL SYNC PROCESS - 完整同步流程")
        logger.info("=" * 60)
        
        # 步骤 1: 同步所有数据
        logger.info("Step 1: Syncing industry and market cap data...")
        try:
            rows = self.sync_all_data()
            logger.info(f"Sync completed: {rows} rows updated")
        except Exception as e:
            logger.error(f"Sync failed: {e}")
            return False
        
        # 步骤 2: 验证数据完整性
        logger.info("Step 2: Verifying data completeness...")
        passed, verification = self.verify_data_completeness()
        
        if not passed:
            logger.error("=" * 60)
            logger.error("CRITICAL: Data completeness verification FAILED")
            logger.error("Missing rate exceeds threshold (1%)")
            logger.error("Program will exit as per physical verification requirements")
            logger.error("=" * 60)
            return False
        
        logger.success("=" * 60)
        logger.success("FULL SYNC COMPLETED SUCCESSFULLY")
        logger.success("=" * 60)
        
        return True


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 初始化同步器
    syncer = IndustryAndMarketCapSyncer()
    
    # 运行完整同步
    success = syncer.run_full_sync()
    
    # 生成报告
    report_path = syncer.generate_report()
    
    # 返回退出码
    if success:
        logger.success("Sync completed successfully")
        sys.exit(0)
    else:
        logger.error("Sync failed - data completeness verification did not pass")
        logger.error("Please check the network connection and try again")
        sys.exit(1)


if __name__ == "__main__":
    main()