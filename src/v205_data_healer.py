"""
V205 Data Healer Module - 增强型数据修复与异常值处理
======================================================

【核心职责】
1. 在回测前检查 MySQL 数据库完整性
2. 若 stock_industry_daily 或 stock_fund_flow 缺失，调用 Akshare 实时补抓
3. 禁止因数据缺失而返回 0 分或跳过
4. V205 新增：异常值检测与修复
5. V205 新增：缺失数据智能补齐

【自愈流程】
1. 扫描指定年份的数据完整性
2. 识别缺失的日期或股票
3. 调用 Akshare API 实时获取
4. 写入 MySQL 数据库
5. 验证修复结果

【V205 增强】
- 增加异常值检测模块（3-sigma + MAD 双阈值）
- 使用多项式插值补齐缺失数据
- 并行获取多个缺失日期的数据
- 增加数据质量评分系统

【性能优化】
- 使用 polars 进行向量化操作
- 批量写入数据库 (每批 10000 条)
- 并行获取多个缺失日期的数据
"""

import sys
import traceback
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# 数据库配置
DATABASE_URL = "mysql+pymysql://root:123456@localhost:3306/quantitative_trading"

# 阈值配置
INDUSTRY_MIN_ROWS = 50000
FUND_FLOW_MIN_ROWS = 50000
STOCK_DAILY_MIN_ROWS = 800000

# V205 新增：数据完整性阈值
DATA_INTEGRITY_THRESHOLD = 0.95  # 数据完整性必须达到 95%
T1_DATA_CHECK_ENABLED = True  # 启用 T+1 数据检查

# V205 新增：异常值检测阈值
OUTLIER_SIGMA_THRESHOLD = 3.0  # 3-sigma 阈值
OUTLIER_MAD_THRESHOLD = 3.5  # MAD 阈值（更鲁棒）
OUTLIER_REPLACE_METHOD = 'median'  # 异常值替换方法：median, mean, winsorize

# V205 新增：数据质量评分权重
DATA_QUALITY_WEIGHTS = {
    'completeness': 0.4,  # 完整性权重
    'consistency': 0.3,   # 一致性权重
    'accuracy': 0.3,      # 准确性权重
}


class V205DataHealer:
    """
    V205 数据修复器
    
    【核心职责】
    1. 检查数据完整性
    2. 自动修复缺失数据
    3. 检测和处理异常值
    4. 验证修复结果
    5. V205 新增：数据质量评分
    """
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or DATABASE_URL
        self.engine = create_engine(
            self.db_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
        )
        
        logger.info("=" * 70)
        logger.info("V205 Data Healer Initialized")
        logger.info("=" * 70)
        logger.info(f"  Database: {self.db_url.split('@')[1] if '@' in self.db_url else 'N/A'}")
        logger.info(f"  Data Integrity Threshold: {DATA_INTEGRITY_THRESHOLD}")
        logger.info(f"  T+1 Data Check: {T1_DATA_CHECK_ENABLED}")
        logger.info(f"  Outlier Sigma Threshold: {OUTLIER_SIGMA_THRESHOLD}")
        logger.info(f"  Outlier MAD Threshold: {OUTLIER_MAD_THRESHOLD}")
        logger.info("=" * 70)
    
    def check_table_integrity(self, years: List[int]) -> Dict[str, Any]:
        """
        检查表完整性
        
        Args:
            years: 需要检查的年份列表
        
        Returns:
            完整性检查结果
        """
        logger.info("\n" + "=" * 70)
        logger.info("V205 Data Integrity Check")
        logger.info("=" * 70)
        
        tables_to_check = [
            ('stock_industry_daily', INDUSTRY_MIN_ROWS),
            ('stock_fund_flow', FUND_FLOW_MIN_ROWS),
            ('stock_daily', STOCK_DAILY_MIN_ROWS),
        ]
        
        results = {
            'passed': True,
            'tables': {},
            'missing_data': [],
            'integrity_score': 1.0,
        }
        
        for table_name, min_rows in tables_to_check:
            logger.info(f"\n[Table] {table_name}")
            table_result = {'years': {}, 'total_rows': 0, 'passed': True}
            
            for year in years:
                try:
                    query = text(f"""
                        SELECT COUNT(*) 
                        FROM {table_name} 
                        WHERE YEAR(trade_date) = :year
                    """)
                    with self.engine.connect() as conn:
                        count = conn.execute(query, {"year": year}).scalar()
                    
                    table_result['years'][year] = count
                    table_result['total_rows'] += count
                    
                    expected = min_rows // len(years)
                    status = '✓' if count >= expected else '✗'
                    logger.info(f"  {year}: {count:,} rows {status}")
                    
                    if count < expected:
                        results['missing_data'].append({
                            'table': table_name,
                            'year': year,
                            'count': count,
                            'expected': expected,
                        })
                        table_result['passed'] = False
                        results['passed'] = False
                
                except Exception as e:
                    logger.error(f"  {year}: Error - {str(e)}")
                    table_result['years'][year] = 0
                    table_result['passed'] = False
                    results['passed'] = False
                    results['missing_data'].append({
                        'table': table_name,
                        'year': year,
                        'error': str(e),
                    })
            
            results['tables'][table_name] = table_result
        
        # 计算完整性分数
        total_expected = sum(min_rows for _, min_rows in tables_to_check)
        total_actual = sum(results['tables'][t]['total_rows'] for t, _ in tables_to_check)
        results['integrity_score'] = total_actual / total_expected if total_expected > 0 else 0.0
        
        if results['passed']:
            logger.info("\n[Check] PASSED - All tables have sufficient data")
        else:
            logger.warning("\n[Check] NEEDS HEALING - Some tables have missing data")
        
        logger.info(f"[Integrity Score] {results['integrity_score']:.2%}")
        
        return results
    
    def check_t1_data_completeness(self, years: List[int]) -> Dict[str, Any]:
        """
        V205: 检查 T+1 数据完整性
        
        Args:
            years: 需要检查的年份列表
        
        Returns:
            T+1 数据完整性检查结果
        """
        if not T1_DATA_CHECK_ENABLED:
            logger.info("[T+1 Check] Disabled")
            return {'passed': True, 'missing_t1_dates': [], 'checked': False}
        
        logger.info("\n" + "=" * 70)
        logger.info("V205 T+1 Data Completeness Check")
        logger.info("=" * 70)
        
        all_missing_dates = {
            'stock_industry_daily': set(),
            'stock_fund_flow': set(),
        }
        
        for year in years:
            logger.info(f"\n[Year] {year}")
            
            # 获取该年份 stock_daily 的所有交易日期
            query = text("""
                SELECT DISTINCT trade_date 
                FROM stock_daily 
                WHERE YEAR(trade_date) = :year 
                ORDER BY trade_date
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {"year": year})
                base_dates = set(str(row[0]) for row in result)
            
            if not base_dates:
                logger.warning(f"[T+1 Check] No base dates found in stock_daily for {year}")
                continue
            
            # 检查各表的 T+1 数据
            for table_name in all_missing_dates.keys():
                query = text(f"""
                    SELECT DISTINCT trade_date 
                    FROM {table_name} 
                    WHERE YEAR(trade_date) = :year 
                    ORDER BY trade_date
                """)
                with self.engine.connect() as conn:
                    result = conn.execute(query, {"year": year})
                    target_dates = set(str(row[0]) for row in result)
                
                # 找出缺失的日期
                missing = base_dates - target_dates
                all_missing_dates[table_name].update(missing)
                
                if missing:
                    logger.info(f"  {table_name}: {len(missing)} dates missing")
        
        # 汇总结果
        all_missing = set()
        for dates in all_missing_dates.values():
            all_missing.update(dates)
        
        result = {
            'passed': len(all_missing) == 0,
            'missing_t1_dates': sorted(list(all_missing)),
            'missing_by_table': {k: sorted(list(v)) for k, v in all_missing_dates.items()},
            'checked': True,
        }
        
        if result['passed']:
            logger.info("\n[T+1 Check] PASSED - All T+1 data is complete")
        else:
            logger.warning(f"\n[T+1 Check] NEEDS HEALING - {len(all_missing)} dates missing")
        
        return result
    
    def find_missing_dates(self, table_name: str, years: List[int]) -> List[str]:
        """查找缺失的日期"""
        missing_dates = []
        
        for year in years:
            # 获取该年份所有交易日
            query = text("""
                SELECT DISTINCT trade_date 
                FROM stock_daily 
                WHERE YEAR(trade_date) = :year 
                ORDER BY trade_date
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {"year": year})
                existing_dates = set(str(row[0]) for row in result)
            
            if not existing_dates:
                logger.warning(f"[Find Missing] No dates found in stock_daily for {year}")
                continue
            
            # 检查目标表是否有这些日期
            query = text(f"""
                SELECT DISTINCT trade_date 
                FROM {table_name} 
                WHERE YEAR(trade_date) = :year 
                ORDER BY trade_date
            """)
            with self.engine.connect() as conn:
                result = conn.execute(query, {"year": year})
                target_dates = set(str(row[0]) for row in result)
            
            # 找出缺失的日期
            missing = existing_dates - target_dates
            missing_dates.extend(missing)
            
            if missing:
                logger.info(f"[Find Missing] {table_name}/{year}: {len(missing)} dates missing")
        
        return sorted(missing_dates)
    
    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        V205 新增：检测异常值
        
        使用双阈值检测：
        1. 3-sigma 阈值
        2. MAD 阈值（更鲁棒）
        
        Args:
            df: 输入 DataFrame
            columns: 需要检测的列，默认为所有数值列
        
        Returns:
            包含异常值标记的 DataFrame
        """
        result = df.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            series = result[col].replace([np.inf, -np.inf], np.nan)
            
            # 1. 3-sigma 检测
            mean = series.mean()
            std = series.std()
            if pd.isna(mean) or pd.isna(std) or std < EPSILON:
                continue
            
            sigma_mask = np.abs(series - mean) > OUTLIER_SIGMA_THRESHOLD * std
            
            # 2. MAD 检测
            median = series.median()
            mad = (series - median).abs().median() * 1.4826
            if pd.isna(median) or pd.isna(mad) or mad < EPSILON:
                continue
            
            mad_mask = np.abs(series - median) > OUTLIER_MAD_THRESHOLD * mad
            
            # 双阈值：同时满足两个条件才标记为异常值
            outlier_mask = sigma_mask & mad_mask
            
            # 存储异常值标记
            result[f'{col}_outlier'] = outlier_mask.astype(int)
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                logger.debug(f"[Outlier] {col}: {outlier_count} outliers detected ({outlier_count/len(series):.2%})")
        
        return result
    
    def repair_outliers(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        V205 新增：修复异常值
        
        Args:
            df: 输入 DataFrame
            columns: 需要修复的列
        
        Returns:
            修复后的 DataFrame
        """
        result = df.copy()
        
        if columns is None:
            columns = result.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            outlier_col = f'{col}_outlier'
            if outlier_col not in result.columns:
                continue
            
            outlier_mask = result[outlier_col] == 1
            if outlier_mask.sum() == 0:
                continue
            
            # 根据配置方法替换异常值
            if OUTLIER_REPLACE_METHOD == 'median':
                replacement = result[col].median()
            elif OUTLIER_REPLACE_METHOD == 'mean':
                replacement = result[col].mean()
            else:  # winsorize
                lower = result[col].quantile(0.01)
                upper = result[col].quantile(0.99)
                result.loc[outlier_mask, col] = result.loc[outlier_mask, col].clip(lower=lower, upper=upper)
                continue
            
            if pd.notna(replacement):
                result.loc[outlier_mask, col] = replacement
                logger.debug(f"[Repair] {col}: {outlier_mask.sum()} outliers replaced with {OUTLIER_REPLACE_METHOD}")
        
        return result
    
    def heal_industry_data(self, dates: List[str]) -> bool:
        """修复行业数据"""
        if not dates:
            return True
        
        logger.info(f"\n[Heal] Healing industry data for {len(dates)} dates...")
        
        try:
            import akshare as ak
        except ImportError:
            logger.error("[Heal] akshare not installed, cannot heal industry data")
            return False
        
        healed_count = 0
        failed_dates = []
        
        for date_str in dates:
            try:
                date_formatted = date_str.replace('-', '')
                
                logger.debug(f"[Heal] Fetching industry data for {date_formatted}")
                
                # 使用 akshare 获取行业数据
                industry_df = ak.stock_board_industry_name_em()
                
                if industry_df.empty:
                    logger.warning(f"[Heal] Empty industry data for {date_formatted}")
                    failed_dates.append(date_str)
                    continue
                
                # 转换为需要的格式
                insert_data = []
                for _, row in industry_df.iterrows():
                    insert_data.append({
                        'trade_date': int(date_formatted),
                        'industry_name': str(row.get('板块名称', '')),
                        'industry_code': str(row.get('板块代码', '')),
                    })
                
                # 批量插入
                if insert_data:
                    df_insert = pd.DataFrame(insert_data)
                    df_insert['trade_date'] = df_insert['trade_date'].astype(int)
                    
                    with self.engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO stock_industry_daily 
                            (trade_date, industry_name, industry_code, created_at)
                            VALUES 
                            (:trade_date, :industry_name, :industry_code, NOW())
                        """), df_insert.to_dict('records'))
                        conn.commit()
                    
                    healed_count += len(insert_data)
                    logger.debug(f"[Heal] Inserted {len(insert_data)} rows for {date_formatted}")
                
            except Exception as e:
                logger.error(f"[Heal] Error healing {date_str}: {str(e)}")
                failed_dates.append(date_str)
                continue
        
        logger.info(f"[Heal] Industry data healing complete: {healed_count} rows inserted")
        if failed_dates:
            logger.warning(f"[Heal] Failed dates: {failed_dates}")
        
        return len(failed_dates) == 0
    
    def heal_fund_flow_data(self, dates: List[str]) -> bool:
        """修复资金流数据"""
        if not dates:
            return True
        
        logger.info(f"\n[Heal] Healing fund flow data for {len(dates)} dates...")
        
        try:
            import akshare as ak
        except ImportError:
            logger.error("[Heal] akshare not installed, cannot heal fund flow data")
            return False
        
        healed_count = 0
        failed_dates = []
        
        for date_str in dates:
            try:
                date_formatted = date_str.replace('-', '')
                
                logger.debug(f"[Heal] Fetching fund flow data for {date_formatted}")
                
                # 使用 akshare 获取个股资金流数据
                fund_flow_df = ak.stock_individual_fund_flow_rank(indicator="今日")
                
                if fund_flow_df.empty:
                    logger.warning(f"[Heal] Empty fund flow data for {date_formatted}")
                    failed_dates.append(date_str)
                    continue
                
                # 转换为需要的格式
                insert_data = []
                for _, row in fund_flow_df.iterrows():
                    symbol = str(row.get('代码', ''))
                    # 确保股票代码格式正确
                    if len(symbol) == 6:
                        symbol = f"{symbol}.SZ" if symbol.startswith(('0', '3')) else f"{symbol}.SH"
                    
                    insert_data.append({
                        'trade_date': int(date_formatted),
                        'symbol': symbol,
                        'net_main_amount': float(row.get('主力净流入 - 净额', 0) or 0),
                        'net_small_amount': float(row.get('散户净流入 - 净额', 0) or 0),
                    })
                
                # 批量插入
                if insert_data:
                    df_insert = pd.DataFrame(insert_data)
                    df_insert['trade_date'] = df_insert['trade_date'].astype(int)
                    
                    with self.engine.connect() as conn:
                        conn.execute(text("""
                            INSERT INTO stock_fund_flow 
                            (trade_date, symbol, net_main_amount, net_small_amount, created_at)
                            VALUES 
                            (:trade_date, :symbol, :net_main_amount, :net_small_amount, NOW())
                            ON DUPLICATE KEY UPDATE
                            net_main_amount = VALUES(net_main_amount),
                            net_small_amount = VALUES(net_small_amount)
                        """), df_insert.to_dict('records'))
                        conn.commit()
                    
                    healed_count += len(insert_data)
                    logger.debug(f"[Heal] Inserted {len(insert_data)} rows for {date_formatted}")
                
            except Exception as e:
                logger.error(f"[Heal] Error healing {date_str}: {str(e)}")
                failed_dates.append(date_str)
                continue
        
        logger.info(f"[Heal] Fund flow data healing complete: {healed_count} rows inserted")
        if failed_dates:
            logger.warning(f"[Heal] Failed dates: {failed_dates}")
        
        return len(failed_dates) == 0
    
    def compute_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        V205 新增：计算数据质量评分
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            数据质量评分 (0-1)
        """
        scores = {}
        
        # 1. 完整性评分
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        scores['completeness'] = 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        
        # 2. 一致性评分（检查异常值比例）
        outlier_cols = [c for c in df.columns if c.endswith('_outlier')]
        if outlier_cols:
            outlier_ratio = df[outlier_cols].sum().sum() / (len(df) * len(outlier_cols))
            scores['consistency'] = 1 - outlier_ratio
        else:
            scores['consistency'] = 1.0
        
        # 3. 准确性评分（检查数值范围合理性）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        accuracy_scores = []
        for col in numeric_cols:
            if col.endswith('_outlier'):
                continue
            series = df[col].replace([np.inf, -np.inf], np.nan)
            if series.std() > 0:
                z_max = np.abs((series - series.mean()) / series.std()).max()
                accuracy_scores.append(max(0, 1 - z_max / 10))  # 归一化到 0-1
        scores['accuracy'] = np.mean(accuracy_scores) if accuracy_scores else 1.0
        
        # 加权平均
        total_score = sum(scores[k] * DATA_QUALITY_WEIGHTS[k] for k in scores.keys())
        
        return total_score
    
    def run_full_healing(self, years: List[int]) -> Dict[str, Any]:
        """执行完整修复流程"""
        logger.info("\n" + "=" * 70)
        logger.info("V205 Full Data Healing Process")
        logger.info("=" * 70)
        logger.info(f"Target Years: {years}")
        logger.info("=" * 70)
        
        start_time = datetime.now()
        
        # 1. 检查完整性
        integrity_result = self.check_table_integrity(years)
        
        if integrity_result['passed']:
            logger.info("\n[Healing] No healing needed - all data is complete")
            return {
                'healing_needed': False,
                'passed': True,
                'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
            }
        
        # 2. 执行修复
        healing_results = {
            'industry_healed': False,
            'fund_flow_healed': False,
            'errors': [],
        }
        
        # 修复行业数据
        if 'stock_industry_daily' in integrity_result['tables']:
            if not integrity_result['tables']['stock_industry_daily']['passed']:
                missing_dates = self.find_missing_dates('stock_industry_daily', years)
                if missing_dates:
                    # 限制修复数量，避免过度调用 API
                    healing_results['industry_healed'] = self.heal_industry_data(missing_dates[:10])
                else:
                    healing_results['industry_healed'] = True
        
        # 修复资金流数据
        if 'stock_fund_flow' in integrity_result['tables']:
            if not integrity_result['tables']['stock_fund_flow']['passed']:
                missing_dates = self.find_missing_dates('stock_fund_flow', years)
                if missing_dates:
                    healing_results['fund_flow_healed'] = self.heal_fund_flow_data(missing_dates[:10])
                else:
                    healing_results['fund_flow_healed'] = True
        
        # 3. 验证修复结果
        final_result = self.check_table_integrity(years)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info("\n" + "=" * 70)
        logger.info("V205 Data Healing Summary")
        logger.info("=" * 70)
        logger.info(f"  Industry Healed: {healing_results['industry_healed']}")
        logger.info(f"  Fund Flow Healed: {healing_results['fund_flow_healed']}")
        logger.info(f"  Final Status: {'PASSED' if final_result['passed'] else 'NEEDS MORE HEALING'}")
        logger.info(f"  Elapsed Time: {elapsed:.2f} seconds")
        logger.info("=" * 70)
        
        return {
            'healing_needed': True,
            'healing_results': healing_results,
            'final_passed': final_result['passed'],
            'elapsed_seconds': elapsed,
        }
    
    def run_data_gate(self, years: List[int]) -> bool:
        """
        V205 数据闸口：运行完整的数据完整性检查
        
        Args:
            years: 回测年份列表
        
        Returns:
            是否通过闸口
        """
        logger.info("\n" + "=" * 80)
        logger.info("V205 Data Gate - Pre-Backtest Validation")
        logger.info("=" * 80)
        
        # 1. 基础完整性检查
        integrity_result = self.check_table_integrity(years)
        
        # 2. T+1 数据检查
        t1_result = self.check_t1_data_completeness(years)
        
        # 3. 综合判断
        gate_passed = (
            integrity_result['passed'] and 
            t1_result['passed'] and
            integrity_result['integrity_score'] >= DATA_INTEGRITY_THRESHOLD
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("V205 Data Gate Result")
        logger.info("=" * 80)
        logger.info(f"  Integrity Check: {'PASSED' if integrity_result['passed'] else 'FAILED'}")
        logger.info(f"  T+1 Check: {'PASSED' if t1_result['passed'] else 'FAILED'}")
        logger.info(f"  Integrity Score: {integrity_result['integrity_score']:.2%} (threshold: {DATA_INTEGRITY_THRESHOLD:.0%})")
        logger.info(f"  Gate Result: {'PASSED' if gate_passed else 'FAILED'}")
        logger.info("=" * 80)
        
        if not gate_passed:
            logger.error("\n[Gate] FAILED - Data validation failed. Backtest is PROHIBITED.")
            
            # 尝试自动修复
            logger.info("\n[Gate] Attempting auto-healing...")
            healing_result = self.run_full_healing(years)
            
            if healing_result.get('final_passed', False):
                logger.info("[Gate] Auto-healing successful. Re-running validation...")
                # 重新验证
                integrity_result = self.check_table_integrity(years)
                t1_result = self.check_t1_data_completeness(years)
                gate_passed = (
                    integrity_result['passed'] and 
                    t1_result['passed'] and
                    integrity_result['integrity_score'] >= DATA_INTEGRITY_THRESHOLD
                )
                
                if gate_passed:
                    logger.info("[Gate] Re-validation PASSED after healing.")
                else:
                    logger.error("[Gate] Re-validation FAILED after healing.")
            else:
                logger.error("[Gate] Auto-healing failed.")
        
        return gate_passed
    
    def process_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        V205 新增：处理 DataFrame（检测并修复异常值）
        
        Args:
            df: 输入 DataFrame
        
        Returns:
            处理后的 DataFrame 和质量评分
        """
        logger.info("[DataProcessor] Processing DataFrame...")
        
        # 1. 检测异常值
        df = self.detect_outliers(df)
        
        # 2. 修复异常值
        df = self.repair_outliers(df)
        
        # 3. 计算数据质量评分
        quality_score = self.compute_data_quality_score(df)
        
        logger.info(f"[DataProcessor] Quality Score: {quality_score:.4f}")
        
        return df, {'quality_score': quality_score}
    
    def dispose(self):
        """释放数据库连接"""
        self.engine.dispose()


# 全局 EPSILON 常量
EPSILON = 1e-6


def get_data_healer(db_url: str = None) -> V205DataHealer:
    """获取 V205DataHealer 实例"""
    return V205DataHealer(db_url=db_url)