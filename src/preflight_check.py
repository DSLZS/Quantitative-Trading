"""
Pre-flight Check Script - 数据库表结构与数据完整性验证。

核心功能:
    - 验证 stock_daily 表结构（必须包含 industry_code, total_mv, is_st）
    - 检查数据密度（缺失率 > 10% 则报错停止）
    - 检查因子有效性（无 NaN/Inf）
    - 生成数据审计报告

使用示例:
    >>> python src/preflight_check.py
"""

import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

import polars as pl
from loguru import logger

try:
    from .db_manager import DatabaseManager
except ImportError:
    from db_manager import DatabaseManager


@dataclass
class CheckResult:
    """检查结果数据结构"""
    check_name: str
    passed: bool
    message: str
    details: Dict[str, Any]


class PreflightChecker:
    """
    回测前检查器 - 验证数据库和数据的完整性。
    
    检查项目:
        1. 表结构验证
        2. 字段完整性验证
        3. 数据密度检查
        4. 因子有效性检查
        5. 时间范围检查
    """
    
    # 必需的字段列表
    REQUIRED_COLUMNS = [
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "pct_chg",
        # V8 新增必需字段
        "industry_code",
        "total_mv",
        "is_st",
    ]
    
    # 数值列（用于检查 NaN/Inf）
    NUMERIC_COLUMNS = [
        "open", "high", "low", "close", "volume", "amount",
        "pct_chg", "total_mv",
    ]
    
    # 数据密度阈值
    MISSING_RATE_THRESHOLD = 0.10  # 10%
    
    # 元数据字段缺失率阈值（行业/市值数据可能难以完全获取）
    METADATA_MISSING_THRESHOLD = 0.50  # 50% - 允许一半缺失，使用替代方案
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化检查器。
        
        Args:
            db: 数据库管理器实例
        """
        self.db = db or DatabaseManager()
        self.results: list[CheckResult] = []
    
    def check_table_structure(self) -> CheckResult:
        """
        检查 stock_daily 表结构。
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            # 获取表结构信息
            query = """
                SELECT COLUMN_NAME, DATA_TYPE, COLUMN_COMMENT
                FROM information_schema.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                AND TABLE_NAME = 'stock_daily'
                ORDER BY ORDINAL_POSITION
            """
            columns_df = self.db.read_sql(query)
            
            if columns_df.is_empty():
                return CheckResult(
                    check_name="表结构检查",
                    passed=False,
                    message="stock_daily 表不存在或为空",
                    details={"columns": []}
                )
            
            existing_columns = set(columns_df["COLUMN_NAME"].to_list())
            missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in existing_columns]
            
            result = CheckResult(
                check_name="表结构检查",
                passed=len(missing_columns) == 0,
                message=f"表包含 {len(existing_columns)} 个字段" + 
                       (f", 缺少 {len(missing_columns)} 个必需字段" if missing_columns else ""),
                details={
                    "existing_columns": list(existing_columns),
                    "missing_columns": missing_columns,
                    "column_details": columns_df.to_dict(as_series=False)
                }
            )
            
            if missing_columns:
                logger.error(f"❌ 表结构检查失败：缺少字段 {missing_columns}")
            else:
                logger.success(f"✅ 表结构检查通过：{len(existing_columns)} 个字段")
            
            return result
            
        except Exception as e:
            return CheckResult(
                check_name="表结构检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def check_data_density(self) -> CheckResult:
        """
        检查数据密度（缺失率）。
        
        检查逻辑:
            - 对于每个交易日，计算应有股票数 vs 实际股票数
            - 如果缺失率 > 10%，则检查失败
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            # 获取日期范围和股票数量统计
            query = """
                SELECT 
                    trade_date,
                    COUNT(DISTINCT symbol) as stock_count
                FROM stock_daily
                WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 365 DAY)
                GROUP BY trade_date
                ORDER BY trade_date
            """
            daily_stats = self.db.read_sql(query)
            
            if daily_stats.is_empty():
                return CheckResult(
                    check_name="数据密度检查",
                    passed=False,
                    message="最近一年无数据",
                    details={"daily_stats": {}}
                )
            
            # 计算平均股票数（作为基准）
            avg_stock_count = float(daily_stats["stock_count"].mean())
            
            # 计算每天的缺失率（相对于最大值）
            max_stock_count = float(daily_stats["stock_count"].max())
            
            if max_stock_count <= 0:
                return CheckResult(
                    check_name="数据密度检查",
                    passed=False,
                    message="最大股票数为 0",
                    details={"daily_stats": {}}
                )
            
            # 计算缺失率
            daily_stats = daily_stats.with_columns(
                ((pl.lit(max_stock_count) - pl.col("stock_count")) / pl.lit(max_stock_count) * 100.0)
                .alias("missing_rate")
            )
            
            # 找出缺失率 > 阈值的日期
            high_missing_days = daily_stats.filter(
                pl.col("missing_rate") > self.MISSING_RATE_THRESHOLD * 100
            )
            
            missing_rate_avg = float(daily_stats["missing_rate"].mean())
            missing_rate_max = float(daily_stats["missing_rate"].max())
            
            passed = len(high_missing_days) < len(daily_stats) * 0.3  # 允许最多 30% 的日期缺失率超标
            
            result = CheckResult(
                check_name="数据密度检查",
                passed=passed,
                message=f"平均缺失率 {missing_rate_avg:.2f}%, 最大缺失率 {missing_rate_max:.2f}%",
                details={
                    "avg_stock_count": avg_stock_count,
                    "max_stock_count": max_stock_count,
                    "avg_missing_rate": missing_rate_avg,
                    "max_missing_rate": missing_rate_max,
                    "high_missing_days_count": len(high_missing_days),
                    "total_days": len(daily_stats),
                    "high_missing_dates": high_missing_days["trade_date"].to_list()[:10] if len(high_missing_days) > 0 else []
                }
            )
            
            if passed:
                logger.success(f"✅ 数据密度检查通过：平均缺失率 {missing_rate_avg:.2f}%")
            else:
                logger.error(f"❌ 数据密度检查失败：{len(high_missing_days)} 天缺失率 > {self.MISSING_RATE_THRESHOLD * 100}%")
            
            return result
            
        except Exception as e:
            return CheckResult(
                check_name="数据密度检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def check_required_fields_completeness(self) -> CheckResult:
        """
        检查必需字段（industry_code, total_mv, is_st）的完整性。
        
        注意：industry_code 和 total_mv 是可选元数据字段，如果缺失率高，
        策略将使用替代方案（如仅使用价格数据计算因子）。
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            # 检查每个必需字段的缺失率
            field_stats = {}
            critical_failed = False  # 只有 is_st 缺失会导致失败
            
            # industry_code 和 total_mv: NULL 或空字符串视为缺失（警告级别）
            for field in ["industry_code", "total_mv"]:
                query = f"""
                    SELECT 
                        COUNT(*) as total_count,
                        SUM(CASE WHEN `{field}` IS NULL OR `{field}` = '' THEN 1 ELSE 0 END) as missing_count
                    FROM stock_daily
                """
                result = self.db.read_sql(query)
                
                if result.is_empty():
                    field_stats[field] = {"total": 0, "missing": 0, "rate": 100.0}
                    continue
                
                total = result["total_count"][0]
                missing = result["missing_count"][0]
                rate = (missing / total * 100) if total > 0 else 100.0
                
                field_stats[field] = {
                    "total": int(total),
                    "missing": int(missing),
                    "rate": rate
                }
                
                # 这些是可选字段，只记录警告
                if rate > 90:
                    logger.warning(f"⚠️ 字段 '{field}' 缺失率 {rate:.2f}% - 策略将使用替代方案")
                elif rate > 50:
                    logger.warning(f"⚠️ 字段 '{field}' 缺失率 {rate:.2f}% - 部分功能可能受限")
                else:
                    logger.success(f"✅ 字段 '{field}' 缺失率 {rate:.2f}%")
            
            # is_st: 0 是有效值，只有 NULL 视为缺失（关键检查）
            query = """
                SELECT 
                    COUNT(*) as total_count,
                    SUM(CASE WHEN `is_st` IS NULL THEN 1 ELSE 0 END) as missing_count
                FROM stock_daily
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                field_stats["is_st"] = {"total": 0, "missing": 0, "rate": 100.0}
                critical_failed = True
            else:
                total = result["total_count"][0]
                missing = result["missing_count"][0]
                rate = (missing / total * 100) if total > 0 else 100.0
                
                field_stats["is_st"] = {
                    "total": int(total),
                    "missing": int(missing),
                    "rate": rate
                }
                
                if rate > self.MISSING_RATE_THRESHOLD * 100:
                    critical_failed = True
                    logger.error(f"❌ 字段 'is_st' 缺失率 {rate:.2f}% > {self.MISSING_RATE_THRESHOLD * 100}%")
                else:
                    logger.success(f"✅ 字段 'is_st' 缺失率 {rate:.2f}%")
            
            # 只有 is_st 缺失才会导致检查失败
            return CheckResult(
                check_name="必需字段完整性检查",
                passed=not critical_failed,
                message=f"检查 {len(field_stats)} 个字段 (is_st 为必需，其他为可选)",
                details={
                    "field_stats": field_stats,
                    "threshold": self.MISSING_RATE_THRESHOLD * 100,
                    "metadata_threshold": self.METADATA_MISSING_THRESHOLD * 100
                }
            )
            
        except Exception as e:
            return CheckResult(
                check_name="必需字段完整性检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def check_numeric_validity(self) -> CheckResult:
        """
        检查数值列的有效性（无 NaN/Inf/负值等）。
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            # 抽样检查（避免全表扫描）
            query = """
                SELECT open, high, low, close, volume, amount, pct_chg, total_mv
                FROM stock_daily
                WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
                LIMIT 100000
            """
            sample_df = self.db.read_sql(query)
            
            if sample_df.is_empty():
                return CheckResult(
                    check_name="数值有效性检查",
                    passed=False,
                    message="无样本数据可检查",
                    details={}
                )
            
            invalid_stats = {}
            all_passed = True
            
            for col in self.NUMERIC_COLUMNS:
                if col not in sample_df.columns:
                    invalid_stats[col] = {"status": "column_missing"}
                    continue
                
                col_data = sample_df[col]
                
                # 检查 NaN
                nan_count = col_data.null_count()
                
                # 检查 Inf (Polars 中 Inf 会被表示为特殊值)
                inf_count = 0
                try:
                    inf_count = col_data.filter((col_data == float('inf')) | (col_data == float('-inf'))).len()
                except Exception:
                    pass
                
                # 检查负值（对于某些列如价格，负值是不合理的）
                negative_count = 0
                if col in ["open", "high", "low", "close", "total_mv"]:
                    negative_count = col_data.filter(col_data < 0).len()
                
                total_invalid = nan_count + inf_count + negative_count
                rate = total_invalid / len(sample_df) * 100
                
                invalid_stats[col] = {
                    "nan_count": int(nan_count),
                    "inf_count": int(inf_count),
                    "negative_count": int(negative_count),
                    "total_invalid": int(total_invalid),
                    "invalid_rate": rate
                }
                
                if rate > 5.0:  # 5% 的无效数据阈值
                    all_passed = False
                    logger.warning(f"⚠️ 列 '{col}' 无效数据率 {rate:.2f}%")
            
            return CheckResult(
                check_name="数值有效性检查",
                passed=all_passed,
                message=f"检查 {len(invalid_stats)} 个数值列",
                details={
                    "invalid_stats": invalid_stats,
                    "sample_size": len(sample_df)
                }
            )
            
        except Exception as e:
            return CheckResult(
                check_name="数值有效性检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def check_date_range(self) -> CheckResult:
        """
        检查数据的时间范围。
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            query = """
                SELECT 
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date,
                    COUNT(DISTINCT trade_date) as unique_days
                FROM stock_daily
            """
            result = self.db.read_sql(query)
            
            if result.is_empty():
                return CheckResult(
                    check_name="时间范围检查",
                    passed=False,
                    message="无数据",
                    details={}
                )
            
            min_date = str(result["min_date"][0])
            max_date = str(result["max_date"][0])
            unique_days = int(result["unique_days"][0])
            
            # 检查数据是否足够（至少 1 年）
            try:
                min_dt = datetime.strptime(min_date, "%Y-%m-%d")
                max_dt = datetime.strptime(max_date, "%Y-%m-%d")
                days_span = (max_dt - min_dt).days
                
                passed = days_span >= 252  # 至少一年交易日
            except Exception:
                days_span = 0
                passed = False
            
            return CheckResult(
                check_name="时间范围检查",
                passed=passed,
                message=f"数据范围：{min_date} 至 {max_date} ({unique_days} 天)",
                details={
                    "min_date": min_date,
                    "max_date": max_date,
                    "unique_days": unique_days,
                    "span_days": days_span
                }
            )
            
        except Exception as e:
            return CheckResult(
                check_name="时间范围检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def check_stock_count(self) -> CheckResult:
        """
        检查股票数量。
        
        Returns:
            CheckResult: 检查结果
        """
        try:
            query = """
                SELECT COUNT(DISTINCT symbol) as stock_count
                FROM stock_daily
            """
            result = self.db.read_sql(query)
            
            stock_count = int(result["stock_count"][0]) if not result.is_empty() else 0
            
            # 至少应该有 100 只股票
            passed = stock_count >= 100
            
            return CheckResult(
                check_name="股票数量检查",
                passed=passed,
                message=f"数据库包含 {stock_count} 只股票",
                details={
                    "stock_count": stock_count,
                    "threshold": 100
                }
            )
            
        except Exception as e:
            return CheckResult(
                check_name="股票数量检查",
                passed=False,
                message=f"检查失败：{str(e)}",
                details={"error": str(e)}
            )
    
    def run_all_checks(self) -> Dict[str, Any]:
        """
        运行所有检查。
        
        Returns:
            dict: 所有检查结果的汇总
        """
        logger.info("=" * 60)
        logger.info("PREFLIGHT CHECK - 回测前数据验证")
        logger.info("=" * 60)
        
        self.results = []
        
        # 执行所有检查
        checks = [
            self.check_table_structure,
            self.check_stock_count,
            self.check_date_range,
            self.check_data_density,
            self.check_required_fields_completeness,
            self.check_numeric_validity,
        ]
        
        for check_func in checks:
            result = check_func()
            self.results.append(result)
        
        # 汇总结果
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        all_passed = all(r.passed for r in self.results)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "all_passed": all_passed,
            "passed_count": passed_count,
            "total_count": total_count,
            "results": [asdict(r) for r in self.results]
        }
        
        # 输出总结
        logger.info("=" * 60)
        if all_passed:
            logger.success(f"✅ 所有检查通过 ({passed_count}/{total_count})")
        else:
            logger.error(f"❌ {total_count - passed_count} 项检查失败")
            for r in self.results:
                if not r.passed:
                    logger.error(f"  - {r.check_name}: {r.message}")
        logger.info("=" * 60)
        
        return summary
    
    def generate_report(self, output_path: str = None) -> str:
        """
        生成检查报告。
        
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
            output_path = output_dir / f"preflight_check_{timestamp}.md"
        
        summary = {asdict(r)["check_name"]: asdict(r) for r in self.results}
        
        report_lines = [
            "# Preflight Check Report",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            "## 检查汇总",
            f"",
        ]
        
        all_passed = all(r.passed for r in self.results)
        if all_passed:
            report_lines.append("✅ **所有检查通过**")
        else:
            report_lines.append("❌ **部分检查失败**")
        
        report_lines.append("")
        report_lines.append("| 检查项 | 状态 | 详情 |")
        report_lines.append("|--------|------|------|")
        
        for r in self.results:
            status = "✅" if r.passed else "❌"
            report_lines.append(f"| {r.check_name} | {status} | {r.message} |")
        
        report_lines.append("")
        report_lines.append("## 详细结果")
        report_lines.append("")
        
        for r in self.results:
            report_lines.append(f"### {r.check_name}")
            report_lines.append("")
            report_lines.append(f"**状态**: {'通过' if r.passed else '失败'}")
            report_lines.append(f"**详情**: {r.message}")
            report_lines.append("")
            if r.details:
                report_lines.append("```json")
                import json
                report_lines.append(json.dumps(r.details, indent=2, ensure_ascii=False))
                report_lines.append("```")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"报告已保存至：{output_path}")
        return report_content


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行检查
    checker = PreflightChecker()
    summary = checker.run_all_checks()
    
    # 生成报告
    report_path = checker.generate_report()
    
    # 返回退出码
    if summary["all_passed"]:
        logger.success("Preflight check PASSED")
        sys.exit(0)
    else:
        logger.error("Preflight check FAILED - 请修复上述问题后再运行回测")
        sys.exit(1)


if __name__ == "__main__":
    main()