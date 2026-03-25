"""
V71 DB Validator Module - 数据库验证器

【V71 验证器核心功能】
1. 统计 stock_fund_flow 表的数据分布（按月计数）
2. 统计 stock_industry 表的数据分布
3. 验证 sync_log 表的同步状态
4. 生成数据入库证明报告
5. 检测数据断点和异常

作者：量化系统
版本：V71.0
日期：2026-03-25
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

TABLE_FUND_FLOW = "stock_fund_flow"
TABLE_INDUSTRY = "stock_industry"
TABLE_SYNC_LOG = "sync_log"

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================
# V71 数据库验证器
# ===========================================

class V71DBValidator:
    """
    V71 数据库验证器
    
    【核心功能】
    1. 统计两张表的数据分布（按月计数）
    2. 验证数据完整性和连续性
    3. 检测数据断点
    4. 生成验证报告
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        """
        初始化数据库验证器
        
        Parameters
        ----------
        db : Optional[DatabaseManager]
            数据库管理器实例
        """
        self.db = db if db else DatabaseManager()
        if not hasattr(self.db, '_initialized') or not self.db._initialized:
            self.db.connect()
    
    def check_table_exists(self, table_name: str) -> bool:
        """
        检查表是否存在
        
        Parameters
        ----------
        table_name : str
            表名
        
        Returns
        -------
        bool
            表是否存在
        """
        try:
            return self.db.table_exists(table_name)
        except Exception as e:
            logger.error(f"检查表 {table_name} 存在性失败：{e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        获取表的总行数
        
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
            return int(result["cnt"][0]) if len(result) > 0 else 0
        except Exception as e:
            logger.error(f"获取表 {table_name} 行数失败：{e}")
            return 0
    
    def get_fund_flow_monthly_stats(self) -> pl.DataFrame:
        """
        获取 stock_fund_flow 表的月度统计
        
        Returns
        -------
        pl.DataFrame
            月度统计数据
        """
        try:
            query = f"""
                SELECT 
                    DATE_FORMAT(trade_date, '%Y-%m') as month,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT symbol) as unique_stocks,
                    MIN(trade_date) as min_date,
                    MAX(trade_date) as max_date
                FROM {TABLE_FUND_FLOW}
                WHERE trade_date IS NOT NULL AND trade_date != ''
                GROUP BY DATE_FORMAT(trade_date, '%Y-%m')
                ORDER BY month ASC
            """
            result = self.db.read_sql(query)
            return result
        except Exception as e:
            logger.error(f"获取资金流向月度统计失败：{e}")
            return pl.DataFrame()
    
    def get_fund_flow_daily_stats(self, start_date: str = None, 
                                   end_date: str = None) -> pl.DataFrame:
        """
        获取 stock_fund_flow 表的每日统计
        
        Parameters
        ----------
        start_date : Optional[str]
            开始日期，默认全部
        end_date : Optional[str]
            结束日期，默认全部
        
        Returns
        -------
        pl.DataFrame
            每日统计数据
        """
        try:
            where_clause = "WHERE trade_date IS NOT NULL AND trade_date != ''"
            if start_date:
                where_clause += f" AND trade_date >= '{start_date}'"
            if end_date:
                where_clause += f" AND trade_date <= '{end_date}'"
            
            query = f"""
                SELECT 
                    trade_date,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT symbol) as unique_stocks
                FROM {TABLE_FUND_FLOW}
                {where_clause}
                GROUP BY trade_date
                ORDER BY trade_date ASC
            """
            result = self.db.read_sql(query)
            return result
        except Exception as e:
            logger.error(f"获取资金流向每日统计失败：{e}")
            return pl.DataFrame()
    
    def get_industry_stats(self) -> pl.DataFrame:
        """
        获取 stock_industry 表的统计
        
        Returns
        -------
        pl.DataFrame
            行业统计数据
        """
        try:
            query = f"""
                SELECT 
                    industry_name,
                    industry_code,
                    COUNT(*) as stock_count,
                    GROUP_CONCAT(DISTINCT symbol) as symbols
                FROM {TABLE_INDUSTRY}
                WHERE industry_name IS NOT NULL AND industry_name != ''
                GROUP BY industry_name, industry_code
                ORDER BY stock_count DESC
            """
            result = self.db.read_sql(query)
            return result
        except Exception as e:
            logger.error(f"获取行业统计失败：{e}")
            return pl.DataFrame()
    
    def get_sync_log_stats(self) -> pl.DataFrame:
        """
        获取 sync_log 表的统计
        
        Returns
        -------
        pl.DataFrame
            同步日志统计
        """
        try:
            if not self.check_table_exists(TABLE_SYNC_LOG):
                logger.warning(f"表 {TABLE_SYNC_LOG} 不存在")
                return pl.DataFrame(schema={
                    'trade_date': pl.Utf8,
                    'status': pl.Utf8,
                    'retry_count': pl.Int64,
                    'count': pl.Int64
                })
            
            query = f"""
                SELECT 
                    trade_date,
                    status,
                    SUM(retry_count) as total_retries,
                    COUNT(*) as attempt_count
                FROM {TABLE_SYNC_LOG}
                GROUP BY trade_date, status
                ORDER BY trade_date ASC
            """
            result = self.db.read_sql(query)
            return result
        except Exception as e:
            logger.error(f"获取同步日志统计失败：{e}")
            return pl.DataFrame()
    
    def detect_data_gaps(self, expected_min_records: int = 100) -> List[Dict]:
        """
        检测数据断点
        
        Parameters
        ----------
        expected_min_records : int
            期望的最小记录数
        
        Returns
        -------
        List[Dict]
            断点列表
        """
        gaps = []
        
        try:
            # 获取每日统计
            daily_stats = self.get_fund_flow_daily_stats()
            
            if daily_stats.is_empty():
                return gaps
            
            # 转换为 pandas 便于处理日期
            pdf = daily_stats.to_pandas()
            pdf['trade_date'] = pl.from_pandas(pdf)['trade_date'].to_pandas()
            
            # 检测记录数过少的日期
            low_record_dates = pdf[pdf['record_count'] < expected_min_records]['trade_date'].tolist()
            
            for date in low_record_dates:
                gaps.append({
                    'date': str(date),
                    'type': 'low_record_count',
                    'record_count': int(pdf[pdf['trade_date'] == date]['record_count'].values[0]),
                    'expected': expected_min_records,
                })
            
            # 检测连续日期缺失（简化版）
            if len(pdf) > 1:
                dates = sorted(pdf['trade_date'].unique())
                for i in range(1, len(dates)):
                    prev_date = dates[i-1]
                    curr_date = dates[i]
                    
                    # 计算日期差（跳过周末）
                    try:
                        prev_dt = datetime.strptime(str(prev_date), '%Y-%m-%d')
                        curr_dt = datetime.strptime(str(curr_date), '%Y-%m-%d')
                        delta = (curr_dt - prev_dt).days
                        
                        # 如果超过 3 天（包含周末），可能是断点
                        if delta > 3:
                            gaps.append({
                                'date_range': f"{prev_date} ~ {curr_date}",
                                'type': 'missing_dates',
                                'gap_days': delta,
                            })
                    except Exception:
                        pass
            
        except Exception as e:
            logger.error(f"检测数据断点失败：{e}")
        
        return gaps
    
    def get_stock_coverage(self) -> Dict[str, Any]:
        """
        获取股票覆盖率统计
        
        Returns
        -------
        Dict[str, Any]
            覆盖率统计
        """
        try:
            # 获取所有唯一股票
            query_stocks = f"""
                SELECT DISTINCT symbol 
                FROM {TABLE_FUND_FLOW}
                WHERE symbol IS NOT NULL AND symbol != ''
            """
            stocks_df = self.db.read_sql(query_stocks)
            total_stocks = len(stocks_df)
            
            # 获取每只股票的数据天数
            query_days = f"""
                SELECT 
                    symbol,
                    COUNT(DISTINCT trade_date) as data_days,
                    MIN(trade_date) as first_date,
                    MAX(trade_date) as last_date
                FROM {TABLE_FUND_FLOW}
                WHERE symbol IS NOT NULL AND symbol != ''
                GROUP BY symbol
                ORDER BY data_days DESC
            """
            days_df = self.db.read_sql(query_days)
            
            # 计算平均数据天数
            avg_days = float(days_df['data_days'].mean()) if len(days_df) > 0 else 0
            
            return {
                'total_stocks': total_stocks,
                'avg_data_days_per_stock': round(avg_days, 1),
                'top_stocks': days_df.head(10).to_dict() if len(days_df) > 0 else {},
            }
            
        except Exception as e:
            logger.error(f"获取股票覆盖率失败：{e}")
            return {}
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        生成完整的验证报告
        
        Returns
        -------
        Dict[str, Any]
            验证报告
        """
        logger.info("=" * 80)
        logger.info("V71 数据库验证报告")
        logger.info("=" * 80)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'tables': {},
            'fund_flow_monthly': [],
            'fund_flow_daily': [],
            'industry_stats': [],
            'sync_log_stats': [],
            'data_gaps': [],
            'stock_coverage': {},
            'summary': {},
        }
        
        # 检查表存在性
        for table in [TABLE_FUND_FLOW, TABLE_INDUSTRY, TABLE_SYNC_LOG]:
            exists = self.check_table_exists(table)
            row_count = self.get_table_row_count(table) if exists else 0
            report['tables'][table] = {
                'exists': exists,
                'row_count': row_count,
            }
            logger.info(f"表 {table}: {'存在' if exists else '不存在'} ({row_count} 行)")
        
        # 资金流向月度统计
        if report['tables'][TABLE_FUND_FLOW]['exists']:
            monthly_stats = self.get_fund_flow_monthly_stats()
            if not monthly_stats.is_empty():
                report['fund_flow_monthly'] = monthly_stats.to_dicts()
                logger.info(f"\n资金流向月度统计:")
                for row in report['fund_flow_monthly']:
                    logger.info(f"  {row.get('month', 'N/A')}: "
                              f"{row.get('record_count', 0)} 条记录，"
                              f"{row.get('unique_stocks', 0)} 只股票")
        
        # 资金流向每日统计（最近 30 天）
        if report['tables'][TABLE_FUND_FLOW]['exists']:
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            daily_stats = self.get_fund_flow_daily_stats(start_date=thirty_days_ago)
            if not daily_stats.is_empty():
                report['fund_flow_daily'] = daily_stats.to_dicts()
                logger.info(f"\n最近 30 天每日统计:")
                for row in report['fund_flow_daily'][-10:]:  # 只显示最近 10 天
                    logger.info(f"  {row.get('trade_date', 'N/A')}: "
                              f"{row.get('record_count', 0)} 条记录")
        
        # 行业统计
        if report['tables'][TABLE_INDUSTRY]['exists']:
            industry_stats = self.get_industry_stats()
            if not industry_stats.is_empty():
                report['industry_stats'] = industry_stats.to_dicts()
                logger.info(f"\n行业统计:")
                for row in report['industry_stats'][:10]:  # 只显示前 10 个行业
                    logger.info(f"  {row.get('industry_name', 'N/A')}: "
                              f"{row.get('stock_count', 0)} 只股票")
        
        # 同步日志统计
        if report['tables'][TABLE_SYNC_LOG]['exists']:
            sync_log_stats = self.get_sync_log_stats()
            if not sync_log_stats.is_empty():
                report['sync_log_stats'] = sync_log_stats.to_dicts()
                logger.info(f"\n同步日志统计:")
                success_count = sum(1 for r in report['sync_log_stats'] if r.get('status') == 'success')
                failed_count = sum(1 for r in report['sync_log_stats'] if r.get('status') == 'failed')
                logger.info(f"  成功：{success_count} 次，失败：{failed_count} 次")
        
        # 检测数据断点
        data_gaps = self.detect_data_gaps()
        report['data_gaps'] = data_gaps
        if data_gaps:
            logger.warning(f"\n发现 {len(data_gaps)} 个数据断点/异常:")
            for gap in data_gaps[:10]:  # 只显示前 10 个
                if gap.get('type') == 'low_record_count':
                    logger.warning(f"  {gap.get('date')}: 仅 {gap.get('record_count')} 条记录")
                elif gap.get('type') == 'missing_dates':
                    logger.warning(f"  {gap.get('date_range')}: 缺失 {gap.get('gap_days')} 天")
        else:
            logger.info("\n未发现明显数据断点")
        
        # 股票覆盖率
        stock_coverage = self.get_stock_coverage()
        report['stock_coverage'] = stock_coverage
        if stock_coverage:
            logger.info(f"\n股票覆盖率:")
            logger.info(f"  总股票数：{stock_coverage.get('total_stocks', 0)}")
            logger.info(f"  平均每只股票数据天数：{stock_coverage.get('avg_data_days_per_stock', 0)}")
        
        # 总结
        report['summary'] = {
            'total_fund_flow_records': report['tables'][TABLE_FUND_FLOW]['row_count'],
            'total_industry_records': report['tables'][TABLE_INDUSTRY]['row_count'],
            'months_with_data': len(report['fund_flow_monthly']),
            'data_gaps_found': len(data_gaps),
            'total_stocks_covered': stock_coverage.get('total_stocks', 0),
            'health_status': 'healthy' if len(data_gaps) == 0 else 'warning' if len(data_gaps) < 5 else 'critical',
        }
        
        logger.info("\n" + "=" * 80)
        logger.info("验证总结:")
        logger.info(f"  资金流向总记录数：{report['summary']['total_fund_flow_records']}")
        logger.info(f"  行业记录数：{report['summary']['total_industry_records']}")
        logger.info(f"  有数据的月份数：{report['summary']['months_with_data']}")
        logger.info(f"  数据断点数：{report['summary']['data_gaps_found']}")
        logger.info(f"  覆盖股票数：{report['summary']['total_stocks_covered']}")
        logger.info(f"  健康状态：{report['summary']['health_status']}")
        logger.info("=" * 80)
        
        return report
    
    def save_report(self, report: Dict[str, Any], 
                    output_path: Path = None) -> Path:
        """
        保存验证报告到文件
        
        Parameters
        ----------
        report : Dict[str, Any]
            验证报告
        output_path : Optional[Path]
            输出路径，默认自动生成
        
        Returns
        -------
        Path
            保存的文件路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = REPORT_DIR / f"v71_db_validation_report_{timestamp}.md"
        
        lines = [
            "# V71 数据库验证报告",
            "",
            f"**生成时间**: {report.get('timestamp', 'N/A')}",
            "",
            "## 表概览",
            "",
            "| 表名 | 存在 | 行数 |",
            "|------|------|------|",
        ]
        
        for table_name, info in report.get('tables', {}).items():
            exists_str = "✓" if info.get('exists') else "✗"
            lines.append(f"| {table_name} | {exists_str} | {info.get('row_count', 0):,} |")
        
        lines.extend([
            "",
            "## 资金流向月度统计",
            "",
        ])
        
        for row in report.get('fund_flow_monthly', []):
            lines.append(f"- **{row.get('month', 'N/A')}**: "
                        f"{row.get('record_count', 0):,} 条记录，"
                        f"{row.get('unique_stocks', 0):,} 只股票 "
                        f"({row.get('min_date', 'N/A')} ~ {row.get('max_date', 'N/A')})")
        
        lines.extend([
            "",
            "## 股票覆盖率",
            "",
            f"- 总股票数：{report.get('stock_coverage', {}).get('total_stocks', 0):,}",
            f"- 平均每只股票数据天数：{report.get('stock_coverage', {}).get('avg_data_days_per_stock', 0):.1f}",
            "",
            "## 数据断点",
            "",
        ])
        
        gaps = report.get('data_gaps', [])
        if gaps:
            for gap in gaps[:20]:
                if gap.get('type') == 'low_record_count':
                    lines.append(f"- ⚠️ {gap.get('date')}: 仅 {gap.get('record_count')} 条记录")
                elif gap.get('type') == 'missing_dates':
                    lines.append(f"- ⚠️ {gap.get('date_range')}: 缺失 {gap.get('gap_days')} 天")
            if len(gaps) > 20:
                lines.append(f"- ... 还有 {len(gaps) - 20} 个断点")
        else:
            lines.append("未发现明显数据断点 ✓")
        
        lines.extend([
            "",
            "## 健康状态",
            "",
            f"**{report.get('summary', {}).get('health_status', 'unknown').upper()}**",
            "",
            f"- 资金流向总记录数：{report.get('summary', {}).get('total_fund_flow_records', 0):,}",
            f"- 行业记录数：{report.get('summary', {}).get('total_industry_records', 0):,}",
            f"- 有数据的月份数：{report.get('summary', {}).get('months_with_data', 0)}",
            f"- 数据断点数：{report.get('summary', {}).get('data_gaps_found', 0)}",
            f"- 覆盖股票数：{report.get('summary', {}).get('total_stocks_covered', 0):,}",
            "",
            "---",
            "*报告由 V71 DB Validator 自动生成*",
        ])
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"验证报告已保存：{output_path}")
        return output_path
    
    def close(self) -> None:
        """关闭数据库连接"""
        if self.db:
            self.db.close()
            logger.info("V71: 数据库连接已关闭")


# ===========================================
# 便捷函数
# ===========================================

def validate_database(db: Optional[DatabaseManager] = None,
                      save_report: bool = True) -> Dict[str, Any]:
    """
    便捷函数：运行数据库验证
    
    Parameters
    ----------
    db : Optional[DatabaseManager]
        数据库管理器实例
    save_report : bool
        是否保存报告
    
    Returns
    -------
    Dict[str, Any]
        验证报告
    """
    validator = V71DBValidator(db=db)
    
    try:
        report = validator.generate_validation_report()
        
        if save_report:
            validator.save_report(report)
        
        return report
    
    finally:
        validator.close()


# ===========================================
# CLI 入口
# ===========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V71 DB Validator - 数据库验证器")
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
    parser.add_argument("--no-save", action="store_true",
                       help="不保存报告到文件")
    
    args = parser.parse_args()
    
    # 设置环境变量
    import os
    if args.db_host:
        os.environ["MYSQL_HOST"] = args.db_host
    if args.db_port:
        os.environ["MYSQL_PORT"] = args.db_port
    if args.db_user:
        os.environ["MYSQL_USER"] = args.db_user
    if args.db_password:
        os.environ["MYSQL_PASSWORD"] = args.db_password
    if args.db_name:
        os.environ["MYSQL_DATABASE"] = args.db_name
    
    # 运行验证
    report = validate_database(save_report=not args.no_save)
    
    # 退出码
    health_status = report.get('summary', {}).get('health_status', 'unknown')
    sys.exit(0 if health_status == 'healthy' else 1)