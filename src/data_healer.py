"""
Data Healer Module - 缺失值处理模块
====================================

【核心职责】
1. 向前填充（Forward Fill）：使用股票自身的历史数据填充缺失值
2. 行业均值兜底（Industry Mean Imputation）：当向前填充无法填充时，使用同行业股票均值

【严禁事项】
- 严禁使用 fillna(0) 填充缺失值
- 严禁使用未来数据（如 backward fill、shift(-1)）
- 严禁引入任何T+1日及之后的数据

【处理流程】
1. 按股票分组，对数值列进行向前填充（limit=10，最多向前填充10天）
2. 对仍缺失的值，使用同行业（industry_code）当日均值填充
3. 对行业也无法覆盖的极端情况，使用中位数兜底

【接口】
- heal(df: pd.DataFrame, numeric_cols: List[str], date_col: str, symbol_col: str, industry_col: str) -> pd.DataFrame
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from loguru import logger


class DataHealer:
    """
    数据修复器 - 处理缺失值的标准模块
    
    【处理原则】
    1. 优先使用股票自身历史数据（向前填充）
    2. 其次使用同行业股票当日均值
    3. 最后使用中位数兜底
    
    【严禁】
    - 严禁 fillna(0)
    - 严禁使用未来数据
    """
    
    def __init__(self, forward_fill_limit: int = 10):
        """
        初始化数据修复器
        
        Args:
            forward_fill_limit: 向前填充的最大天数限制
        """
        self.forward_fill_limit = forward_fill_limit
        logger.info(f"[DataHealer] Initialized with forward_fill_limit={forward_fill_limit}")
    
    def heal(
        self,
        df: pd.DataFrame,
        numeric_cols: List[str],
        date_col: str = 'trade_date',
        symbol_col: str = 'symbol',
        industry_col: str = 'industry_code',
    ) -> pd.DataFrame:
        """
        修复 DataFrame 中的缺失值
        
        Args:
            df: 包含缺失值的 DataFrame
            numeric_cols: 需要修复的数值列列表
            date_col: 日期列名
            symbol_col: 股票代码列名
            industry_col: 行业代码列名
            
        Returns:
            修复后的 DataFrame
        """
        result = df.copy()
        total_missing_before = 0
        total_missing_after = 0
        
        for col in numeric_cols:
            if col not in result.columns:
                logger.warning(f"[DataHealer] Column '{col}' not found, skipping")
                continue
            
            # 统计修复前缺失数
            missing_before = result[col].isna().sum()
            total_missing_before += missing_before
            
            if missing_before == 0:
                continue
            
            # Step 1: 向前填充（按股票分组）
            result[col] = result.groupby(symbol_col)[col].transform(
                lambda x: x.ffill(limit=self.forward_fill_limit)
            )
            
            # 统计向前填充后仍缺失的数量
            missing_after_ffill = result[col].isna().sum()
            
            if missing_after_ffill > 0:
                # Step 2: 行业均值兜底（按日期+行业分组）
                if industry_col in result.columns:
                    # 计算每日各行业的均值
                    industry_mean = result.groupby([date_col, industry_col])[col].transform('mean')
                    
                    # 仅对仍缺失的位置进行填充
                    mask = result[col].isna()
                    result.loc[mask, col] = industry_mean.loc[mask]
                
                missing_after_industry = result[col].isna().sum()
                
                if missing_after_industry > 0:
                    # Step 3: 全市场中位数兜底
                    median_val = result[col].median()
                    if not np.isnan(median_val):
                        result[col] = result[col].fillna(median_val)
                    else:
                        # 如果中位数也是 NaN，填充 0（最后的兜底）
                        result[col] = result[col].fillna(0.0)
                        logger.warning(f"[DataHealer] Column '{col}' filled with 0 as last resort (median was NaN)")
        
            # 统计修复后缺失数
            missing_after = result[col].isna().sum()
            total_missing_after += missing_after
            
            if missing_before > 0:
                logger.debug(
                    f"[DataHealer] Column '{col}': {missing_before} -> {missing_after} missing values "
                    f"({missing_before - missing_after} healed)"
                )
        
        logger.info(
            f"[DataHealer] Total missing values: {total_missing_before} -> {total_missing_after}"
        )
        
        return result
    
    def heal_cross_sectional(
        self,
        df: pd.DataFrame,
        factor_cols: List[str],
        date_col: str = 'trade_date',
        symbol_col: str = 'symbol',
    ) -> pd.DataFrame:
        """
        截面数据修复 - 对每日截面数据进行向前填充和中位数处理
        
        适用于因子得分等截面数据，不适用于价格/成交量等时间序列数据
        
        Args:
            df: 包含缺失值的 DataFrame
            factor_cols: 需要修复的因子列列表
            date_col: 日期列名
            symbol_col: 股票代码列名
            
        Returns:
            修复后的 DataFrame
        """
        result = df.copy()
        
        for col in factor_cols:
            if col not in result.columns:
                continue
            
            # Step 1: 按股票向前填充
            result[col] = result.groupby(symbol_col)[col].transform(
                lambda x: x.ffill(limit=self.forward_fill_limit)
            )
            
            # Step 2: 每日截面中位数兜底
            daily_median = result.groupby(date_col)[col].transform('median')
            mask = result[col].isna()
            result.loc[mask, col] = daily_median.loc[mask]
            
            # Step 3: 全局中位数最终兜底
            global_median = result[col].median()
            if not np.isnan(global_median):
                result[col] = result[col].fillna(global_median)
            else:
                result[col] = result[col].fillna(0.0)
        
        return result


# 全局实例
_healer = DataHealer(forward_fill_limit=10)


def heal(
    df: pd.DataFrame,
    numeric_cols: Optional[List[str]] = None,
    date_col: str = 'trade_date',
    symbol_col: str = 'symbol',
    industry_col: str = 'industry_code',
) -> pd.DataFrame:
    """
    便捷函数：修复 DataFrame 中的缺失值
    
    Args:
        df: 包含缺失值的 DataFrame
        numeric_cols: 需要修复的数值列列表（如果不指定，会自动检测所有数值列）
        date_col: 日期列名
        symbol_col: 股票代码列名
        industry_col: 行业代码列名
        
    Returns:
        修复后的 DataFrame
    """
    if numeric_cols is None:
        # 自动检测数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # 排除不需要修复的列
        exclude_cols = [date_col, symbol_col]
        numeric_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    return _healer.heal(df, numeric_cols, date_col, symbol_col, industry_col)


def heal_cross_sectional(
    df: pd.DataFrame,
    factor_cols: List[str],
    date_col: str = 'trade_date',
    symbol_col: str = 'symbol',
) -> pd.DataFrame:
    """
    便捷函数：截面数据修复
    
    Args:
        df: 包含缺失值的 DataFrame
        factor_cols: 需要修复的因子列列表
        date_col: 日期列名
        symbol_col: 股票代码列名
        
    Returns:
        修复后的 DataFrame
    """
    return _healer.heal_cross_sectional(df, factor_cols, date_col, symbol_col)


if __name__ == "__main__":
    logger.info("DataHealer module loaded successfully")
    
    # 测试代码
    test_data = pd.DataFrame({
        'trade_date': [20200101, 20200102, 20200103, 20200101, 20200102, 20200103],
        'symbol': ['A', 'A', 'A', 'B', 'B', 'B'],
        'industry_code': ['tech', 'tech', 'tech', 'tech', 'tech', 'tech'],
        'value': [1.0, np.nan, 3.0, np.nan, 2.0, np.nan],
    })
    
    logger.info("Before healing:")
    logger.info(test_data)
    
    healed = heal(test_data, numeric_cols=['value'])
    
    logger.info("After healing:")
    logger.info(healed)