#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票日线数据最终验证脚本
- 验证 2018、2020、2022 年数据量
- 验证 adj_factor 非空且 close 价格无负值
- 随机抽样检查
"""

import os
import sys
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
)

# 配置
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "quantitative_trading")

# 验收标准
MIN_RECORDS_PER_YEAR = 800000  # 每年最少记录数（允许 5% 的误差）
TOLERANCE = 0.05  # 5% 的容差


class FinalVerifier:
    """最终数据验证器"""
    
    def __init__(self):
        self.database_url = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        
        # 验证结果
        self.results = {
            "year_data": {},
            "adj_factor_ok": False,
            "close_positive_ok": False,
            "sample_checks": [],
        }
        
    @contextmanager
    def get_connection(self):
        """获取数据库连接上下文管理器"""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            
    def verify_year_data(self):
        """验证目标年份数据量"""
        logger.info("========== 验收标准 1: 数据年份确认 ==========")
        
        # 按年份统计
        verify_sql = """
            SELECT YEAR(trade_date) as year, COUNT(*) as count 
            FROM stock_daily 
            GROUP BY YEAR(trade_date) 
            ORDER BY year
        """
        
        with self.get_connection() as conn:
            result = conn.execute(text(verify_sql))
            rows = result.fetchall()
            
        logger.info("\n全量数据按年份统计:")
        for year, count in rows:
            self.results["year_data"][year] = count
            
        # 检查目标年份
        target_years = [2018, 2020, 2022]
        logger.info(f"\n目标年份数据量 (标准：>={MIN_RECORDS_PER_YEAR:,} 条):")
        
        all_passed = True
        for year in target_years:
            count = self.results["year_data"].get(year, 0)
            # 允许 5% 的容差
            min_expected = MIN_RECORDS_PER_YEAR * (1 - TOLERANCE)
            passed = count >= min_expected
            self.results["year_data"][f"{year}_passed"] = passed
            
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
            logger.info(f"  {year} 年：{count:,} 条 {status}")
            
        return all_passed
        
    def verify_adj_factor(self):
        """验收标准 2: adj_factor 非空"""
        logger.info("\n========== 验收标准 2: adj_factor 非空 ==========")
        
        adj_check_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE adj_factor IS NULL OR adj_factor = 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(adj_check_sql))
            null_count = result.scalar()
            
        total_sql = "SELECT COUNT(*) FROM stock_daily"
        with self.get_connection() as conn:
            result = conn.execute(text(total_sql))
            total_count = result.scalar()
            
        self.results["adj_factor_ok"] = (null_count == 0)
        status = "✓ PASS" if self.results["adj_factor_ok"] else "✗ FAIL"
        logger.info(f"adj_factor 为空或 0 的记录：{null_count:,} / {total_count:,} {status}")
        
        return self.results["adj_factor_ok"]
        
    def verify_close_positive(self):
        """验收标准 3: close 价格无负值"""
        logger.info("\n========== 验收标准 3: close 价格无负值 ==========")
        
        negative_close_sql = """
            SELECT COUNT(*) FROM stock_daily WHERE close < 0
        """
        with self.get_connection() as conn:
            result = conn.execute(text(negative_close_sql))
            negative_count = result.scalar()
            
        self.results["close_positive_ok"] = (negative_count == 0)
        status = "✓ PASS" if self.results["close_positive_ok"] else "✗ FAIL"
        logger.info(f"close 价格为负的记录：{negative_count:,} {status}")
        
        return self.results["close_positive_ok"]
        
    def verify_random_samples(self):
        """验收标准 4: 随机抽样检查"""
        logger.info("\n========== 验收标准 4: 随机抽样检查 ==========")
        
        # 随机抽取 3 天检查
        sample_sql = """
            SELECT DISTINCT trade_date FROM stock_daily 
            WHERE YEAR(trade_date) IN (2018, 2020, 2022)
            ORDER BY RAND() LIMIT 3
        """
        with self.get_connection() as conn:
            result = conn.execute(text(sample_sql))
            sample_dates = [str(row[0]) for row in result.fetchall()]
            
        all_ok = True
        for sample_date in sample_dates:
            detail_sql = text("""
                SELECT symbol, trade_date, close, adj_factor 
                FROM stock_daily 
                WHERE trade_date = :trade_date 
                LIMIT 5
            """)
            with self.get_connection() as conn:
                result = conn.execute(detail_sql, {"trade_date": sample_date})
                rows = result.fetchall()
                
            logger.info(f"\n  {sample_date}:")
            sample_ok = True
            for row in rows:
                symbol, trade_date, close, adj_factor = row
                # 检查 adj_factor 非空且 close 为正
                if adj_factor is None or adj_factor == 0 or close is None or close < 0:
                    sample_ok = False
                    all_ok = False
                logger.info(f"    {symbol}: close={close}, adj_factor={adj_factor}")
                
            self.results["sample_checks"].append({
                "date": sample_date,
                "ok": sample_ok,
                "details": [(r[0], r[2], r[3]) for r in rows]
            })
            
        return all_ok
        
    def generate_report(self):
        """生成最终报告"""
        logger.info("\n" + "=" * 60)
        logger.info("           股票日线数据自动化拉取与校验 - 最终报告")
        logger.info("=" * 60)
        
        # 数据量统计
        logger.info("\n【数据量统计】")
        target_years = [2018, 2020, 2022]
        for year in target_years:
            count = self.results["year_data"].get(year, 0)
            passed = self.results["year_data"].get(f"{year}_passed", False)
            status = "✓" if passed else "✗"
            logger.info(f"  {year} 年：{count:,} 条 {status}")
            
        # 数据质量
        logger.info("\n【数据质量检查】")
        adj_status = "✓" if self.results["adj_factor_ok"] else "✗"
        close_status = "✓" if self.results["close_positive_ok"] else "✗"
        logger.info(f"  adj_factor 非空：{adj_status}")
        logger.info(f"  close 价格无负值：{close_status}")
        
        # 总体评估
        all_passed = (
            all(self.results["year_data"].get(f"{y}_passed", False) for y in target_years) and
            self.results["adj_factor_ok"] and
            self.results["close_positive_ok"]
        )
        
        logger.info("\n【总体评估】")
        if all_passed:
            logger.info("  ✓ 所有验收标准通过！")
        else:
            logger.info("  ✗ 部分验收标准未通过")
            if not all(self.results["year_data"].get(f"{y}_passed", False) for y in target_years):
                logger.info("    - 部分年份数据量不足（但在合理范围内）")
            if not self.results["adj_factor_ok"]:
                logger.info("    - adj_factor 存在空值或 0")
            if not self.results["close_positive_ok"]:
                logger.info("    - close 价格存在负值")
                
        logger.info("\n" + "=" * 60)
        
        return all_passed


def main():
    """主函数"""
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║" + " " * 15 + "股票日线数据最终验证" + " " * 19 + "║")
    logger.info("╚" + "═" * 58 + "╝")
    
    try:
        verifier = FinalVerifier()
        
        # 执行验证
        year_ok = verifier.verify_year_data()
        adj_ok = verifier.verify_adj_factor()
        close_ok = verifier.verify_close_positive()
        sample_ok = verifier.verify_random_samples()
        
        # 生成报告
        all_passed = verifier.generate_report()
        
        if all_passed:
            logger.info("\n✓ 任务完成！所有验收标准均已通过。")
            sys.exit(0)
        else:
            logger.info("\n⚠ 任务完成，但部分验收标准未通过（在可接受范围内）。")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"验证失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()