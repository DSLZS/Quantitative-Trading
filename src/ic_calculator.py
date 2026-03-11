#!/usr/bin/env python3
"""
IC (Information Coefficient) Calculator - Factor IC Analysis

This script calculates IC values for factors to measure their
predictive power for future returns.

IC 值计算脚本：
    - 计算每个因子的 Rank IC 值
    - 计算 IC 序列的统计信息
    - 支持新因子 IC 值分析

使用示例:
    python src/ic_calculator.py
    python src/ic_calculator.py --parquet data/parquet/features_latest.parquet
"""

import sys
import argparse
from pathlib import Path
from typing import Any
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from loguru import logger
import polars as pl
import numpy as np

try:
    from factor_engine import FactorEngine
except ImportError:
    from src.factor_engine import FactorEngine

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class ICCalculator:
    """
    IC 值计算器，用于评估因子预测能力。
    
    功能特性:
        - 计算 Rank IC 值
        - 计算 IC 序列统计信息
        - 支持因子 IC 对比分析
    
    IC 值说明:
        - IC > 0.05: 强预测能力
        - IC > 0.03: 中等预测能力
        - IC > 0.01: 弱预测能力
        - IC < 0: 反向预测能力
    """
    
    def __init__(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        config_path: str = "config/factors.yaml",
        output_dir: str = "data/plots",
    ) -> None:
        """
        初始化 IC 计算器。
        
        Args:
            parquet_path (str): Parquet 特征文件路径
            config_path (str): 因子配置文件路径
            output_dir (str): 输出目录
        """
        self.parquet_path = parquet_path
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载因子引擎
        self.factor_engine = FactorEngine(config_path)
        
        logger.info(f"ICCalculator initialized")
        logger.info(f"  Parquet path: {parquet_path}")
        logger.info(f"  Config path: {config_path}")
    
    def load_data(self) -> pl.DataFrame:
        """加载 Parquet 数据。"""
        logger.info(f"Loading data from {self.parquet_path}")
        df = pl.read_parquet(self.parquet_path)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def calculate_rank_ic(
        self,
        factor_values: pl.Series,
        label_values: pl.Series,
    ) -> float:
        """
        计算 Rank IC 值（Spearman 相关系数）。
        
        Args:
            factor_values: 因子值序列
            label_values: 标签值序列（未来收益率）
            
        Returns:
            float: Rank IC 值
        """
        # 去除空值
        mask = factor_values.is_not_null() & label_values.is_not_null()
        factor_clean = factor_values.filter(mask)
        label_clean = label_values.filter(mask)
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算秩
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        # 计算 Pearson 相关系数（在秩上）
        factor_np = factor_ranks.to_numpy()
        label_np = label_ranks.to_numpy()
        
        if np.std(factor_np) < 1e-10 or np.std(label_np) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_np, label_np)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_factor_ic(
        self,
        df: pl.DataFrame,
        factor_name: str,
        label_column: str = "future_return_5",
    ) -> dict[str, Any]:
        """
        计算单个因子的 IC 值（按日期分组计算）。
        
        Args:
            df: 包含因子值和标签的 DataFrame
            factor_name: 因子名称
            label_column: 标签列名
            
        Returns:
            dict: IC 统计信息
        """
        if factor_name not in df.columns:
            logger.warning(f"Factor {factor_name} not found in data")
            return {
                "factor_name": factor_name,
                "mean_ic": 0.0,
                "ic_std": 0.0,
                "ic_ir": 0.0,
                "positive_ratio": 0.0,
                "num_valid_days": 0,
                "min_ic": 0.0,
                "max_ic": 0.0,
                "t_stat": 0.0,
            }
        
        # 按日期分组计算 IC
        unique_dates = df["trade_date"].unique()
        ic_series = []
        
        for date in unique_dates:
            day_data = df.filter(pl.col("trade_date") == date)
            
            if len(day_data) < 10:
                continue
            
            factor_values = day_data[factor_name]
            label_values = day_data[label_column]
            
            ic = self.calculate_rank_ic(factor_values, label_values)
            
            if ic != 0 or not np.isnan(ic):
                ic_series.append({
                    "trade_date": date,
                    "factor_name": factor_name,
                    "ic": ic,
                })
        
        if not ic_series:
            return {
                "factor_name": factor_name,
                "mean_ic": 0.0,
                "ic_std": 0.0,
                "ic_ir": 0.0,
                "positive_ratio": 0.0,
                "num_valid_days": 0,
            }
        
        ic_df = pl.DataFrame(ic_series)
        ic_values = ic_df["ic"].to_numpy()
        
        # 计算 IC 统计信息
        mean_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0  # IC 比率（IR）
        positive_ratio = float(np.sum(ic_values > 0) / len(ic_values))
        
        t_stat = mean_ic / (ic_std / np.sqrt(len(ic_values))) if ic_std > 0 and len(ic_values) > 0 else 0.0
        
        result = {
            "factor_name": factor_name,
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "positive_ratio": positive_ratio,
            "num_valid_days": len(ic_values),
            "min_ic": float(np.min(ic_values)),
            "max_ic": float(np.max(ic_values)),
            "t_stat": t_stat,
        }
        
        logger.debug(f"Factor {factor_name}: mean_ic={mean_ic:.4f}, ic_ir={ic_ir:.2f}")
        
        return result
    
    def calculate_all_factors_ic(
        self,
        df: pl.DataFrame,
        label_column: str = "future_return_5",
    ) -> list[dict[str, Any]]:
        """
        计算所有因子的 IC 值。
        
        Args:
            df: 包含因子值和标签的 DataFrame
            label_column: 标签列名
            
        Returns:
            list[dict]: 所有因子的 IC 统计信息列表
        """
        factor_names = self.factor_engine.get_factor_names()
        
        logger.info(f"Calculating IC for {len(factor_names)} factors")
        
        results = []
        for factor_name in factor_names:
            logger.info(f"  Calculating IC for: {factor_name}")
            result = self.calculate_factor_ic(df, factor_name, label_column)
            results.append(result)
        
        # 按 mean_ic 绝对值排序
        results.sort(key=lambda x: abs(x["mean_ic"]), reverse=True)
        
        return results
    
    def print_ic_summary(self, results: list[dict[str, Any]]) -> None:
        """打印 IC 值汇总表。"""
        if not results:
            logger.warning("No IC results to display")
            return
        
        logger.info("\n" + "=" * 100)
        logger.info("FACTOR IC ANALYSIS SUMMARY")
        logger.info("=" * 100)
        
        # 打印表头
        header = (
            f"{'Rank':<5} {'Factor Name':<30} {'Mean IC':<12} {'IC Std':<10} "
            f"{'IC IR':<10} {'Positive%':<12} {'T-Stat':<10} {'Valid Days':<12}"
        )
        logger.info(header)
        logger.info("-" * 100)
        
        # 打印所有结果
        for i, r in enumerate(results, 1):
            # 根据 IC 绝对值添加标记
            ic_marker = ""
            if abs(r["mean_ic"]) >= 0.05:
                ic_marker = " ***"
            elif abs(r["mean_ic"]) >= 0.03:
                ic_marker = " **"
            elif abs(r["mean_ic"]) >= 0.01:
                ic_marker = " *"
            
            row = (
                f"{i:<5} {r['factor_name']:<30} {r['mean_ic']:>10.4f}   "
                f"{r['ic_std']:>8.4f}   {r['ic_ir']:>8.2f}   "
                f"{r['positive_ratio']:>10.1%}   {r['t_stat']:>8.2f}   "
                f"{r['num_valid_days']:>10d}{ic_marker}"
            )
            logger.info(row)
        
        logger.info("-" * 100)
        logger.info("IC Legend: *** >= 0.05, ** >= 0.03, * >= 0.01")
        logger.info("=" * 100)
    
    def save_ic_results(
        self,
        results: list[dict[str, Any]],
        filename: str = "factor_ic_results.json",
    ) -> str:
        """保存 IC 结果到 JSON 文件。"""
        output_path = self.output_dir / filename
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"IC results saved to: {output_path}")
        return str(output_path)
    
    def plot_ic_heatmap(
        self,
        results: list[dict[str, Any]],
        save_path: str | None = None,
    ) -> None:
        """
        绘制 IC 值热力图。
        
        Args:
            results: IC 结果列表
            save_path: 保存路径
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            if not results:
                logger.warning("No IC data to plot")
                return
            
            # 准备数据
            factor_names = [r["factor_name"] for r in results]
            mean_ics = [r["mean_ic"] for r in results]
            ic_stds = [r["ic_std"] for r in results]
            
            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle("Factor IC Analysis", fontsize=14)
            
            # 左图：Mean IC 条形图
            ax1 = axes[0]
            colors = ['green' if ic > 0 else 'red' for ic in mean_ics]
            bars = ax1.barh(range(len(factor_names)), mean_ics, color=colors, alpha=0.7)
            ax1.set_yticks(range(len(factor_names)))
            ax1.set_yticklabels(factor_names)
            ax1.set_xlabel("Mean IC")
            ax1.set_title("Mean IC by Factor")
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax1.axvline(x=0.01, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.01')
            ax1.axvline(x=-0.01, color='gray', linestyle='--', alpha=0.5)
            ax1.legend()
            
            # 添加数值标注
            for i, (bar, ic) in enumerate(zip(bars, mean_ics)):
                ax1.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f"{ic:.4f}", va='center', fontsize=8)
            
            # 右图：IC IR 对比
            ax2 = axes[1]
            ic_irs = [r["ic_ir"] for r in results]
            colors_ir = ['green' if ir > 0 else 'red' for ir in ic_irs]
            bars_ir = ax2.barh(range(len(factor_names)), ic_irs, color=colors_ir, alpha=0.7)
            ax2.set_yticks(range(len(factor_names)))
            ax2.set_yticklabels([])
            ax2.set_xlabel("IC IR (Information Ratio)")
            ax2.set_title("IC IR by Factor")
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
            ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold 0.5')
            ax2.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
            ax2.legend()
            
            plt.tight_layout()
            
            if save_path:
                save_file = Path(save_path)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                logger.info(f"IC heatmap saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate IC heatmap: {e}")
    
    def run_full_analysis(
        self,
        plot: bool = True,
    ) -> dict[str, Any]:
        """
        运行完整 IC 分析流程。
        
        Args:
            plot (bool): 是否绘制图表
            
        Returns:
            dict: 分析结果
        """
        # 加载数据
        df = self.load_data()
        
        # 计算所有因子 IC
        results = self.calculate_all_factors_ic(df)
        
        # 打印汇总表
        self.print_ic_summary(results)
        
        # 保存结果
        self.save_ic_results(results)
        
        # 绘制热力图
        if plot:
            heatmap_path = str(self.output_dir / "factor_ic_heatmap.png")
            self.plot_ic_heatmap(results, save_path=heatmap_path)
        
        # 找出新因子的 IC
        new_factors = ["volatility_contraction_10", "volume_shrink_ratio", 
                       "volume_price_stable", "accumulation_distribution_20"]
        new_factor_results = [r for r in results if r["factor_name"] in new_factors]
        
        if new_factor_results:
            logger.info("\n" + "=" * 60)
            logger.info("NEW FACTORS IC ANALYSIS")
            logger.info("=" * 60)
            for r in new_factor_results:
                logger.info(f"  {r['factor_name']}:")
                logger.info(f"    Mean IC: {r['mean_ic']:.4f}")
                logger.info(f"    IC IR: {r['ic_ir']:.2f}")
                logger.info(f"    Positive Ratio: {r['positive_ratio']:.1%}")
        
        return {
            "all_results": results,
            "new_factors": new_factor_results,
            "best_factor": results[0] if results else None,
        }


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="Factor IC Calculator")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/parquet/features_latest.parquet",
        help="Path to features Parquet file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/factors.yaml",
        help="Path to factor config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/plots",
        help="Output directory",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Factor IC Analysis")
    logger.info("=" * 60)
    
    # 初始化计算器
    calculator = ICCalculator(
        parquet_path=args.parquet,
        config_path=args.config,
        output_dir=args.output,
    )
    
    # 运行完整分析
    results = calculator.run_full_analysis(plot=not args.no_plot)
    
    logger.info("\n" + "=" * 60)
    logger.info("IC Analysis Complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()