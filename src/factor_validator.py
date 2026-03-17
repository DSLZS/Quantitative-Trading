"""
Factor Validator Module - 因子验证与相关性分析。

核心功能:
    - 计算所有基础因子的 Rank IC
    - 输出因子相关性矩阵
    - 因子自相关性分析
    - 因子 IC 衰减分析
    - 因子筛选（剔除冗余因子）

使用示例:
    >>> from factor_validator import FactorValidator
    >>> validator = FactorValidator()
    >>> validator.validate_factors(data_df)
"""

import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .factor_engine import FactorEngine
except ImportError:
    from db_manager import DatabaseManager
    from factor_engine import FactorEngine


class FactorValidator:
    """
    因子验证器 - 评估因子有效性和冗余度。
    
    核心功能:
        1. Rank IC 计算 - 评估因子预测能力
        2. 相关性矩阵 - 检测因子共线性
        3. 自相关性分析 - 检测因子时间序列相关性
        4. IC 衰减分析 - 评估因子预测周期
        5. 因子筛选 - 剔除冗余因子
    """
    
    # 默认配置
    DEFAULT_CONFIG = {
        "ic_threshold": 0.03,  # IC 阈值，低于此值的因子视为无效
        "correlation_threshold": 0.8,  # 相关性阈值，高于此值的因子视为冗余
        "autocorr_threshold": 0.9,  # 自相关阈值，高于此值的因子视为高度自相关
        "min_periods": 60,  # 最小计算周期
    }
    
    def __init__(
        self,
        config_path: str = "config/factors.yaml",
        db: Optional[DatabaseManager] = None,
        ic_threshold: float = 0.03,
        correlation_threshold: float = 0.8,
    ):
        """
        初始化因子验证器。
        
        Args:
            config_path: 因子配置文件路径
            db: 数据库管理器
            ic_threshold: IC 阈值
            correlation_threshold: 相关性阈值
        """
        self.config_path = config_path
        self.db = db or DatabaseManager()
        self.ic_threshold = ic_threshold
        self.correlation_threshold = correlation_threshold
        self.factor_engine = None
        
        # 初始化因子引擎
        try:
            self.factor_engine = FactorEngine(config_path, validate=False)
        except FileNotFoundError:
            logger.warning(f"Factor config not found: {config_path}, using default factors")
            self.factor_engine = FactorEngine.__new__(FactorEngine)
            self.factor_engine.factors = []
        
        logger.info(f"FactorValidator initialized: ic_threshold={ic_threshold}, corr_threshold={correlation_threshold}")
    
    def compute_rank_ic(
        self,
        factor_values: pl.Series,
        forward_returns: pl.Series,
    ) -> float:
        """
        计算 Rank IC（秩相关系数）。
        
        【金融逻辑】
        - Rank IC = Spearman 相关系数
        - 衡量因子值排名与未来收益排名的相关性
        - IC > 0.03 通常认为因子有效
        
        Args:
            factor_values: 因子值序列
            forward_returns: 未来收益序列
            
        Returns:
            Rank IC 值
        """
        if len(factor_values) < 10:
            return 0.0
        
        # 处理空值
        valid_mask = factor_values.is_not_null() & forward_returns.is_not_null()
        if valid_mask.sum() < 10:
            return 0.0
        
        factor_valid = factor_values.filter(valid_mask)
        return_valid = forward_returns.filter(valid_mask)
        
        # 计算秩
        factor_rank = factor_valid.rank("average")
        return_rank = return_valid.rank("average")
        
        # 计算 Pearson 相关系数（对秩）
        if factor_rank.std() < 1e-10 or return_rank.std() < 1e-10:
            return 0.0
        
        ic = (factor_rank * return_rank).mean() - factor_rank.mean() * return_rank.mean()
        ic = ic / (factor_rank.std() * return_rank.std())
        
        return float(ic) if np.isfinite(ic) else 0.0
    
    def compute_ic_by_date(
        self,
        data: pl.DataFrame,
        factor_name: str,
        return_column: str = "future_return_5d",
    ) -> pl.DataFrame:
        """
        计算因子每天的 IC 值。
        
        Args:
            data: 包含因子值和收益的 DataFrame
            factor_name: 因子名称
            return_column: 收益列名
            
        Returns:
            包含每天 IC 的 DataFrame
        """
        if "trade_date" not in data.columns:
            logger.error("Missing trade_date column")
            return pl.DataFrame()
        
        # 按日期分组计算 IC
        ic_by_date = data.group_by("trade_date", maintain_order=True).agg([
            pl.struct([factor_name, return_column]).apply(
                lambda x: self.compute_rank_ic(
                    pl.Series([v[factor_name] for v in x]),
                    pl.Series([v[return_column] for v in x])
                ) if len(x) >= 10 else 0.0
            ).alias("rank_ic")
        ])
        
        return ic_by_date
    
    def compute_factor_ic_statistics(
        self,
        data: pl.DataFrame,
        factor_names: Optional[List[str]] = None,
        return_column: str = "future_return_5d",
    ) -> Dict[str, Any]:
        """
        计算因子 IC 统计指标。
        
        Args:
            data: DataFrame
            factor_names: 因子名称列表
            return_column: 收益列名
            
        Returns:
            IC 统计字典
        """
        if factor_names is None:
            factor_names = self.factor_engine.get_factor_names() if self.factor_engine else []
        
        ic_stats = {}
        
        for factor_name in factor_names:
            if factor_name not in data.columns:
                continue
            
            if return_column not in data.columns:
                logger.warning(f"Return column '{return_column}' not found")
                continue
            
            # 计算每天 IC
            ic_by_date = self.compute_ic_by_date(data, factor_name, return_column)
            
            if ic_by_date.is_empty():
                continue
            
            ic_values = ic_by_date["rank_ic"].drop_nulls()
            
            if len(ic_values) == 0:
                continue
            
            # 计算 IC 统计指标
            ic_mean = float(ic_values.mean())
            ic_std = float(ic_values.std())
            ic_ir = ic_mean / (ic_std + 1e-10) if ic_std > 0 else 0  # IC IR
            ic_tstat = ic_mean / (ic_std / np.sqrt(len(ic_values)) + 1e-10) if ic_std > 0 else 0
            
            # 计算 IC > 0 的比例
            ic_positive_rate = (ic_values > 0).sum() / len(ic_values)
            
            ic_stats[factor_name] = {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_tstat": ic_tstat,
                "ic_positive_rate": ic_positive_rate,
                "n_periods": len(ic_values),
            }
        
        return ic_stats
    
    def compute_correlation_matrix(
        self,
        data: pl.DataFrame,
        factor_names: Optional[List[str]] = None,
    ) -> pl.DataFrame:
        """
        计算因子相关性矩阵。
        
        【金融逻辑】
        - 高度相关的因子（> 0.8）提供相似信息
        - 应该剔除冗余因子，减少模型复杂度
        
        Args:
            data: DataFrame
            factor_names: 因子名称列表
            
        Returns:
            相关性矩阵 DataFrame
        """
        if factor_names is None:
            factor_names = self.factor_engine.get_factor_names() if self.factor_engine else []
        
        # 过滤出因子列
        available_factors = [f for f in factor_names if f in data.columns]
        
        if len(available_factors) < 2:
            logger.warning("Not enough factors to compute correlation matrix")
            return pl.DataFrame()
        
        # 计算相关系数矩阵
        corr_data = data.select(available_factors)
        corr_matrix = corr_data.corr()
        
        return corr_matrix
    
    def compute_autocorrelation(
        self,
        data: pl.DataFrame,
        factor_name: str,
        max_lag: int = 20,
    ) -> Dict[int, float]:
        """
        计算因子自相关性。
        
        【金融逻辑】
        - 高度自相关的因子（> 0.9）变化缓慢
        - 增量因子（因子变化量）可能更有预测力
        
        Args:
            data: DataFrame
            factor_name: 因子名称
            max_lag: 最大滞后阶数
            
        Returns:
            自相关系数字典 {lag: autocorr}
        """
        if factor_name not in data.columns:
            return {}
        
        factor_values = data[factor_name].drop_nulls()
        
        if len(factor_values) < max_lag + 10:
            return {}
        
        # 计算均值和方差
        factor_mean = factor_values.mean()
        factor_var = factor_values.var()
        
        if factor_var < 1e-10:
            return {}
        
        autocorrs = {}
        
        for lag in range(1, max_lag + 1):
            # 计算滞后自相关
            factor_lagged = factor_values.shift(lag)
            
            # 共同部分
            valid_mask = factor_values.is_not_null() & factor_lagged.is_not_null()
            if valid_mask.sum() < 10:
                continue
            
            f1 = factor_values.filter(valid_mask)
            f2 = factor_lagged.filter(valid_mask)
            
            # 自相关系数
            autocorr = ((f1 - factor_mean) * (f2 - factor_mean)).mean() / factor_var
            
            autocorrs[lag] = float(autocorr) if np.isfinite(autocorr) else 0.0
        
        return autocorrs
    
    def identify_redundant_factors(
        self,
        corr_matrix: pl.DataFrame,
        ic_stats: Dict[str, Any],
    ) -> List[str]:
        """
        识别冗余因子。
        
        【筛选逻辑】
        1. 对于高度相关的因子对（> threshold）
        2. 保留 IC 更高的因子
        3. 剔除 IC 较低的因子
        
        Args:
            corr_matrix: 相关性矩阵
            ic_stats: IC 统计
            
        Returns:
            冗余因子列表
        """
        redundant = []
        
        if corr_matrix.is_empty():
            return redundant
        
        # 获取因子名称
        factor_names = [c for c in corr_matrix.columns if c != corr_matrix.columns[0]]
        
        # 遍历上三角矩阵
        for i, f1 in enumerate(factor_names):
            for j, f2 in enumerate(factor_names):
                if i >= j:
                    continue
                
                # 获取相关系数
                try:
                    corr_value = corr_matrix.filter(pl.col(corr_matrix.columns[0]) == f1)[f2][0]
                except (KeyError, IndexError):
                    continue
                
                if abs(corr_value) > self.correlation_threshold:
                    # 高度相关，保留 IC 更高的
                    ic1 = ic_stats.get(f1, {}).get("ic_mean", 0)
                    ic2 = ic_stats.get(f2, {}).get("ic_mean", 0)
                    
                    if ic1 >= ic2:
                        redundant.append(f2)
                    else:
                        redundant.append(f1)
        
        return list(set(redundant))
    
    def compute_ic_decay(
        self,
        data: pl.DataFrame,
        factor_name: str,
        max_horizon: int = 10,
        return_prefix: str = "future_return_",
    ) -> Dict[int, float]:
        """
        计算因子 IC 衰减 - 评估因子预测周期。
        
        【金融逻辑】
        - IC 衰减越慢，因子预测周期越长
        - 短期因子（5 日）可能在长期失效
        
        Args:
            data: DataFrame
            factor_name: 因子名称
            max_horizon: 最大预测周期
            return_prefix: 收益列前缀
            
        Returns:
            IC 衰减字典 {horizon: ic}
        """
        ic_decay = {}
        
        for horizon in range(1, max_horizon + 1):
            return_col = f"{return_prefix}{horizon}d"
            
            if return_col not in data.columns:
                continue
            
            ic_by_date = self.compute_ic_by_date(data, factor_name, return_col)
            
            if ic_by_date.is_empty():
                continue
            
            ic_values = ic_by_date["rank_ic"].drop_nulls()
            
            if len(ic_values) > 0:
                ic_decay[horizon] = float(ic_values.mean())
        
        return ic_decay
    
    def validate_factors(
        self,
        data: pl.DataFrame,
        return_column: str = "future_return_5d",
    ) -> Dict[str, Any]:
        """
        执行完整的因子验证流程。
        
        Args:
            data: 包含因子值和收益的 DataFrame
            return_column: 收益列名
            
        Returns:
            验证结果字典
        """
        logger.info("=" * 60)
        logger.info("FACTOR VALIDATION - 因子验证")
        logger.info("=" * 60)
        
        # 1. 计算 IC 统计
        logger.info("Step 1: Computing IC statistics...")
        ic_stats = self.compute_factor_ic_statistics(data, return_column=return_column)
        
        # 输出 IC 统计
        logger.info("-" * 40)
        logger.info("Factor IC Statistics:")
        valid_factors = []
        for factor_name, stats in sorted(ic_stats.items(), key=lambda x: abs(x[1]["ic_mean"]), reverse=True):
            status = "✅" if abs(stats["ic_mean"]) > self.ic_threshold else "⚠️"
            logger.info(f"  {status} {factor_name}: IC={stats['ic_mean']:.4f}, IR={stats['ic_ir']:.3f}")
            if abs(stats["ic_mean"]) > self.ic_threshold:
                valid_factors.append(factor_name)
        
        # 2. 计算相关性矩阵
        logger.info("Step 2: Computing correlation matrix...")
        corr_matrix = self.compute_correlation_matrix(data, factor_names=list(ic_stats.keys()))
        
        if not corr_matrix.is_empty():
            logger.info("-" * 40)
            logger.info("High Correlation Pairs (>{:.1f}%):".format(self.correlation_threshold * 100))
            # 找出高度相关的因子对
            factor_names = list(ic_stats.keys())
            high_corr_pairs = []
            for i, f1 in enumerate(factor_names):
                for j, f2 in enumerate(factor_names):
                    if i >= j:
                        continue
                    try:
                        corr_value = corr_matrix.filter(pl.col(corr_matrix.columns[0]) == f1)[f2][0]
                        if abs(corr_value) > self.correlation_threshold:
                            high_corr_pairs.append((f1, f2, corr_value))
                            logger.info(f"  ⚠️ {f1} <-> {f2}: {corr_value:.3f}")
                    except (KeyError, IndexError):
                        continue
            if not high_corr_pairs:
                logger.info("  None found")
        
        # 3. 识别冗余因子
        logger.info("Step 3: Identifying redundant factors...")
        redundant = self.identify_redundant_factors(corr_matrix, ic_stats)
        
        if redundant:
            logger.warning(f"  Redundant factors: {redundant}")
        else:
            logger.info("  No redundant factors found")
        
        # 4. 自相关性分析（针对 Top 5 因子）
        logger.info("Step 4: Autocorrelation analysis...")
        top_factors = sorted(ic_stats.items(), key=lambda x: abs(x[1]["ic_mean"]), reverse=True)[:5]
        
        for factor_name, stats in top_factors:
            autocorrs = self.compute_autocorrelation(data, factor_name, max_lag=10)
            if autocorrs:
                lag1_autocorr = autocorrs.get(1, 0)
                status = "⚠️" if abs(lag1_autocorr) > 0.9 else "✅"
                logger.info(f"  {status} {factor_name}: Lag-1 Autocorr={lag1_autocorr:.3f}")
        
        # 5. IC 衰减分析（针对 Top 3 因子）
        logger.info("Step 5: IC decay analysis...")
        top_3_factors = [f[0] for f in top_factors[:3]]
        
        for factor_name in top_3_factors:
            ic_decay = self.compute_ic_decay(data, factor_name, max_horizon=10)
            if ic_decay:
                decay_str = ", ".join([f"T+{h}:{ic:.3f}" for h, ic in ic_decay.items()])
                logger.info(f"  {factor_name}: {decay_str}")
        
        # 汇总结果
        result = {
            "ic_statistics": ic_stats,
            "correlation_matrix": corr_matrix,
            "redundant_factors": redundant,
            "valid_factors": valid_factors,
            "summary": {
                "total_factors": len(ic_stats),
                "valid_factors_count": len(valid_factors),
                "redundant_factors_count": len(redundant),
            }
        }
        
        logger.info("=" * 60)
        logger.info(f"VALIDATION COMPLETE: {len(valid_factors)}/{len(ic_stats)} factors valid")
        logger.info("=" * 60)
        
        return result
    
    def generate_report(
        self,
        validation_result: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> str:
        """
        生成因子验证报告。
        
        Args:
            validation_result: 验证结果
            output_path: 输出路径
            
        Returns:
            报告内容
        """
        if output_path is None:
            output_dir = Path("reports")
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"factor_validation_{timestamp}.md"
        
        report_lines = [
            "# Factor Validation Report",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 汇总",
            "",
            f"| 指标 | 值 |",
            f"|------|-----|",
            f"| 总因子数 | {validation_result['summary']['total_factors']} |",
            f"| 有效因子数 | {validation_result['summary']['valid_factors_count']} |",
            f"| 冗余因子数 | {validation_result['summary']['redundant_factors_count']} |",
            "",
            "## IC 统计",
            "",
            "| 因子 | IC Mean | IC Std | IC IR | Positive Rate |",
            "|------|---------|--------|-------|---------------|",
        ]
        
        for factor_name, stats in sorted(
            validation_result["ic_statistics"].items(),
            key=lambda x: abs(x[1]["ic_mean"]),
            reverse=True
        ):
            status = "✅" if abs(stats["ic_mean"]) > self.ic_threshold else "⚠️"
            report_lines.append(
                f"| {status} {factor_name} | {stats['ic_mean']:.4f} | "
                f"{stats['ic_std']:.4f} | {stats['ic_ir']:.3f} | "
                f"{stats['ic_positive_rate']:.1%} |"
            )
        
        report_lines.extend([
            "",
            "## 冗余因子",
            "",
        ])
        
        if validation_result["redundant_factors"]:
            report_lines.append(f"以下因子与其他因子高度相关（>{self.correlation_threshold:.0%}），建议剔除:")
            report_lines.append("")
            for factor in validation_result["redundant_factors"]:
                report_lines.append(f"- {factor}")
        else:
            report_lines.append("未发现冗余因子")
        
        report_lines.extend([
            "",
            "## 相关性矩阵",
            "",
            "```",
        ])
        
        if not validation_result["correlation_matrix"].is_empty():
            report_lines.append(str(validation_result["correlation_matrix"]))
        
        report_lines.extend([
            "```",
            "",
        ])
        
        report_content = "\n".join(report_lines)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {output_path}")
        
        return report_content


def load_factor_data_from_db(
    db: Optional[DatabaseManager] = None,
    start_date: str = "2025-01-01",
    end_date: str = "2026-03-17",
) -> pl.DataFrame:
    """
    从数据库加载因子数据。
    
    Args:
        db: 数据库管理器
        start_date: 开始日期
        end_date: 结束日期
        
    Returns:
        因子数据 DataFrame
    """
    if db is None:
        db = DatabaseManager()
    
    # 查询数据
    query = f"""
        SELECT 
            symbol,
            trade_date,
            open, high, low, close, pre_close,
            volume, amount, pct_chg,
            industry_code, total_mv, is_st
        FROM stock_daily
        WHERE trade_date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY trade_date, symbol
    """
    
    data = db.read_sql(query)
    
    return data


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 加载数据
    logger.info("Loading factor data from database...")
    data = load_factor_data_from_db()
    
    if data.is_empty():
        logger.error("No data loaded from database")
        return
    
    logger.info(f"Loaded {len(data)} rows, {data['symbol'].n_unique()} stocks")
    
    # 初始化因子引擎并计算因子
    logger.info("Computing factors...")
    factor_engine = FactorEngine("config/factors.yaml")
    
    # 计算因子
    data_with_factors = factor_engine.compute_factors(data)
    
    # 验证因子
    logger.info("Starting factor validation...")
    validator = FactorValidator("config/factors.yaml")
    result = validator.validate_factors(data_with_factors)
    
    # 生成报告
    validator.generate_report(result)


if __name__ == "__main__":
    main()