#!/usr/bin/env python3
"""
Parameter Sensitivity Scanner - Grid Search for Optimal Parameters

This script performs grid search over parameter combinations to find
the optimal configuration that maximizes return/drawdown ratio.

参数敏感性扫描脚本：
    - 对 min_hold_days, threshold, max_positions 进行网格搜索
    - 支持动态仓位管理参数扫描
    - 生成 Heatmap 风格的参数对比表
    - 输出最优参数组合

使用示例:
    python src/parameter_scan.py
    python src/parameter_scan.py --parquet data/parquet/features_latest.parquet
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any
from itertools import product
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from loguru import logger
import polars as pl

try:
    from backtester import Backtester
    from visualizer import Visualizer
except ImportError:
    from src.backtester import Backtester
    from src.visualizer import Visualizer

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)


class ParameterScanner:
    """
    参数敏感性扫描器，用于网格搜索最优参数组合。
    
    功能特性:
        - 支持多参数网格搜索
        - 计算收益/回撤比等核心指标
        - 生成 Heatmap 数据
        - 支持动态仓位管理对比
    
    使用示例:
        >>> scanner = ParameterScanner(parquet_path="data/parquet/features_latest.parquet")
        >>> results = scanner.run_grid_search()
        >>> scanner.print_summary_table(results)
    """
    
    # 默认参数范围
    DEFAULT_MIN_HOLD_DAYS = [3, 5, 8, 10]
    DEFAULT_THRESHOLD = [0.01, 0.015, 0.02, 0.025]
    DEFAULT_MAX_POSITIONS = [3, 5, 10]
    
    def __init__(
        self,
        parquet_path: str = "data/parquet/features_latest.parquet",
        model_path: str = "data/models/stock_model.txt",
        initial_capital: float = 1_000_000.0,
        output_dir: str = "data/plots",
    ) -> None:
        """
        初始化参数扫描器。
        
        Args:
            parquet_path (str): Parquet 特征文件路径
            model_path (str): 模型文件路径
            initial_capital (float): 初始资金
            output_dir (str): 输出目录
        """
        self.parquet_path = parquet_path
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 参数范围
        self.min_hold_days_range = self.DEFAULT_MIN_HOLD_DAYS
        self.threshold_range = self.DEFAULT_THRESHOLD
        self.max_positions_range = self.DEFAULT_MAX_POSITIONS
        
        logger.info(f"ParameterScanner initialized")
        logger.info(f"  Parquet path: {parquet_path}")
        logger.info(f"  Output dir: {output_dir}")
    
    def set_parameter_range(
        self,
        min_hold_days: list[int] | None = None,
        threshold: list[float] | None = None,
        max_positions: list[int] | None = None,
    ) -> None:
        """设置参数扫描范围。"""
        if min_hold_days:
            self.min_hold_days_range = min_hold_days
        if threshold:
            self.threshold_range = threshold
        if max_positions:
            self.max_positions_range = max_positions
        
        logger.info(f"Parameter ranges updated:")
        logger.info(f"  min_hold_days: {self.min_hold_days_range}")
        logger.info(f"  threshold: {[f'{t:.3f}' for t in self.threshold_range]}")
        logger.info(f"  max_positions: {self.max_positions_range}")
    
    def run_single_backtest(
        self,
        min_hold_days: int,
        threshold: float,
        max_positions: int,
        use_dynamic_position: bool = False,
        max_single_position: float = 0.3,
    ) -> dict[str, Any]:
        """
        运行单次回测。
        
        Returns:
            dict: 包含参数和绩效指标的字典
        """
        backtester = Backtester(
            initial_capital=self.initial_capital,
            prediction_threshold=threshold,
            max_positions=max_positions,
            min_hold_days=min_hold_days,
            use_dynamic_position=use_dynamic_position,
            max_single_position=max_single_position,
        )
        
        try:
            results = backtester.run(
                parquet_path=self.parquet_path,
                model_path=self.model_path,
            )
            
            metrics = results["metrics"]
            
            # 计算收益/回撤比
            return_draw_ratio = 0.0
            if metrics["max_drawdown"] > 0:
                return_draw_ratio = metrics["annualized_return"] / metrics["max_drawdown"]
            
            return {
                "min_hold_days": min_hold_days,
                "threshold": threshold,
                "max_positions": max_positions,
                "use_dynamic_position": use_dynamic_position,
                "total_return": metrics["total_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "return_draw_ratio": return_draw_ratio,
                "volatility": metrics.get("volatility", 0),
                "win_rate": metrics.get("win_rate", 0),
                "num_trades": metrics.get("num_trades", 0),
                "cost_ratio": metrics.get("cost_ratio", 0),
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"Backtest failed for params: min_hold_days={min_hold_days}, "
                        f"threshold={threshold}, max_positions={max_positions}: {e}")
            return {
                "min_hold_days": min_hold_days,
                "threshold": threshold,
                "max_positions": max_positions,
                "use_dynamic_position": use_dynamic_position,
                "total_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 1.0,
                "sharpe_ratio": 0.0,
                "return_draw_ratio": 0.0,
                "success": False,
                "error": str(e),
            }
    
    def run_grid_search(
        self,
        use_dynamic_position: bool = False,
        skip_failed: bool = True,
    ) -> list[dict[str, Any]]:
        """
        运行网格搜索。
        
        Args:
            use_dynamic_position (bool): 是否使用动态仓位管理
            skip_failed (bool): 是否跳过失败的回测
            
        Returns:
            list[dict]: 所有参数组合的结果列表
        """
        logger.info("=" * 60)
        logger.info("Starting Grid Search")
        logger.info("=" * 60)
        
        total_combinations = (
            len(self.min_hold_days_range) * 
            len(self.threshold_range) * 
            len(self.max_positions_range)
        )
        
        logger.info(f"Total parameter combinations: {total_combinations}")
        logger.info(f"  min_hold_days: {self.min_hold_days_range}")
        logger.info(f"  threshold: {[f'{t:.3f}' for t in self.threshold_range]}")
        logger.info(f"  max_positions: {self.max_positions_range}")
        logger.info(f"  use_dynamic_position: {use_dynamic_position}")
        logger.info("=" * 60)
        
        results = []
        completed = 0
        
        for min_hold_days, threshold, max_positions in product(
            self.min_hold_days_range,
            self.threshold_range,
            self.max_positions_range,
        ):
            completed += 1
            logger.info(f"[{completed}/{total_combinations}] Running: "
                       f"min_hold_days={min_hold_days}, threshold={threshold:.3f}, "
                       f"max_positions={max_positions}")
            
            result = self.run_single_backtest(
                min_hold_days=min_hold_days,
                threshold=threshold,
                max_positions=max_positions,
                use_dynamic_position=use_dynamic_position,
            )
            
            if skip_failed and not result["success"]:
                continue
            
            results.append(result)
            
            logger.info(f"  Result: annualized_return={result['annualized_return']:.2%}, "
                       f"max_drawdown={result['max_drawdown']:.2%}, "
                       f"sharpe={result['sharpe_ratio']:.2f}, "
                       f"return/drawdown={result['return_draw_ratio']:.2f}")
        
        logger.info("=" * 60)
        logger.info(f"Grid Search Complete: {len(results)} combinations tested")
        logger.info("=" * 60)
        
        return results
    
    def find_best_params(
        self,
        results: list[dict[str, Any]],
        metric: str = "return_draw_ratio",
    ) -> dict[str, Any]:
        """
        查找最优参数组合。
        
        Args:
            results: 回测结果列表
            metric: 排序指标，可选：
                - return_draw_ratio (收益/回撤比)
                - sharpe_ratio (夏普比率)
                - annualized_return (年化收益)
                - total_return (总收益)
                
        Returns:
            dict: 最优参数组合
        """
        if not results:
            return {}
        
        # 按指定指标排序
        sorted_results = sorted(
            results,
            key=lambda x: x.get(metric, 0),
            reverse=True,
        )
        
        best = sorted_results[0]
        
        logger.info(f"Best parameters by {metric}:")
        logger.info(f"  min_hold_days: {best['min_hold_days']}")
        logger.info(f"  threshold: {best['threshold']:.4f}")
        logger.info(f"  max_positions: {best['max_positions']}")
        logger.info(f"  annualized_return: {best['annualized_return']:.2%}")
        logger.info(f"  max_drawdown: {best['max_drawdown']:.2%}")
        logger.info(f"  sharpe_ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"  return_draw_ratio: {best['return_draw_ratio']:.2f}")
        
        return best
    
    def print_summary_table(self, results: list[dict[str, Any]]) -> None:
        """打印参数对比汇总表。"""
        if not results:
            logger.warning("No results to display")
            return
        
        logger.info("\n" + "=" * 100)
        logger.info("PARAMETER SENSITIVITY ANALYSIS SUMMARY")
        logger.info("=" * 100)
        
        # 按收益/回撤比排序
        sorted_results = sorted(
            results,
            key=lambda x: x.get("return_draw_ratio", 0),
            reverse=True,
        )
        
        # 打印表头
        header = (
            f"{'Rank':<5} {'min_hold':<10} {'threshold':<12} {'max_pos':<8} "
            f"{'Ann.Return':<14} {'MaxDD':<10} {'Sharpe':<8} {'Ret/DD':<8} {'WinRate':<10}"
        )
        logger.info(header)
        logger.info("-" * 100)
        
        # 打印前 20 条结果
        for i, r in enumerate(sorted_results[:20], 1):
            row = (
                f"{i:<5} {r['min_hold_days']:<10} {r['threshold']:<12.4f} {r['max_positions']:<8} "
                f"{r['annualized_return']:>10.2%}   {r['max_drawdown']:>8.2%}   "
                f"{r['sharpe_ratio']:>6.2f}   {r['return_draw_ratio']:>6.2f}   "
                f"{r['win_rate']:>8.2%}"
            )
            logger.info(row)
        
        logger.info("=" * 100)
    
    def generate_heatmap_data(
        self,
        results: list[dict[str, Any]],
        metric: str = "return_draw_ratio",
    ) -> dict[str, Any]:
        """
        生成 Heatmap 数据。
        
        Args:
            results: 回测结果列表
            metric: 用于 Heatmap 的指标
            
        Returns:
            dict: Heatmap 数据，包含：
                - x_labels: X 轴标签 (threshold)
                - y_labels: Y 轴标签 (min_hold_days)
                - data: 二维数据数组
                - max_positions: 对应的 max_positions 值
        """
        if not results:
            return {}
        
        # 按 max_positions 分组
        heatmap_data = {}
        
        for max_pos in self.max_positions_range:
            subset = [r for r in results if r["max_positions"] == max_pos]
            
            if not subset:
                continue
            
            # 创建二维矩阵
            data = []
            for min_hold in self.min_hold_days_range:
                row = []
                for threshold in self.threshold_range:
                    match = next(
                        (r for r in subset 
                         if r["min_hold_days"] == min_hold and abs(r["threshold"] - threshold) < 1e-6),
                        None
                    )
                    if match:
                        row.append(match.get(metric, 0))
                    else:
                        row.append(None)
                data.append(row)
            
            heatmap_data[max_pos] = {
                "x_labels": [f"{t:.3f}" for t in self.threshold_range],
                "y_labels": [str(m) for m in self.min_hold_days_range],
                "data": data,
                "metric": metric,
            }
        
        return heatmap_data
    
    def save_results(
        self,
        results: list[dict[str, Any]],
        filename: str = "parameter_scan_results.json",
    ) -> str:
        """保存结果到 JSON 文件。"""
        output_path = self.output_dir / filename
        
        # 转换结果为可序列化格式
        serializable_results = []
        for r in results:
            sr = {}
            for k, v in r.items():
                if isinstance(v, float):
                    sr[k] = round(v, 6)
                else:
                    sr[k] = v
            serializable_results.append(sr)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        return str(output_path)
    
    def plot_heatmap(
        self,
        heatmap_data: dict[str, Any],
        save_path: str | None = None,
    ) -> None:
        """
        绘制 Heatmap 图。
        
        Args:
            heatmap_data: Heatmap 数据
            save_path: 保存路径
        """
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            
            num_configs = len(heatmap_data)
            if num_configs == 0:
                logger.warning("No heatmap data to plot")
                return
            
            fig, axes = plt.subplots(1, num_configs, figsize=(5 * num_configs, 4))
            if num_configs == 1:
                axes = [axes]
            
            for idx, (max_pos, data) in enumerate(sorted(heatmap_data.items())):
                ax = axes[idx]
                
                matrix = np.array(data["data"])
                
                # 处理 None 值
                mask = np.isnan(matrix) if matrix.dtype == float else np.array([[v is None for v in row] for row in matrix])
                
                im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto')
                
                # 设置标签
                ax.set_xticks(range(len(data["x_labels"])))
                ax.set_yticks(range(len(data["y_labels"])))
                ax.set_xticklabels(data["x_labels"])
                ax.set_yticklabels(data["y_labels"])
                
                ax.set_xlabel("Threshold")
                ax.set_ylabel("Min Hold Days")
                ax.set_title(f"Max Positions = {max_pos}")
                
                # 添加数值标注
                for i in range(len(data["y_labels"])):
                    for j in range(len(data["x_labels"])):
                        val = matrix[i, j]
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            ax.text(j, i, f"{val:.2f}", ha='center', va='center', fontsize=8)
                
                plt.colorbar(im, ax=ax, label=data["metric"])
            
            plt.suptitle("Parameter Sensitivity Heatmap", fontsize=14)
            plt.tight_layout()
            
            if save_path:
                save_file = Path(save_path)
                save_file.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_file, dpi=150, bbox_inches='tight')
                logger.info(f"Heatmap saved to: {save_path}")
            
            plt.close()
            
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
    
    def run_full_scan(
        self,
        use_dynamic_position: bool = False,
        plot_heatmap: bool = True,
    ) -> dict[str, Any]:
        """
        运行完整扫描流程。
        
        Args:
            use_dynamic_position (bool): 是否使用动态仓位管理
            plot_heatmap (bool): 是否绘制 Heatmap
            
        Returns:
            dict: 扫描结果摘要
        """
        # 运行网格搜索
        results = self.run_grid_search(use_dynamic_position=use_dynamic_position)
        
        # 保存结果
        self.save_results(results)
        
        # 打印汇总表
        self.print_summary_table(results)
        
        # 查找最优参数
        best_by_return_draw = self.find_best_params(results, "return_draw_ratio")
        best_by_sharpe = self.find_best_params(results, "sharpe_ratio")
        
        # 生成 Heatmap 数据
        heatmap_data = self.generate_heatmap_data(results)
        
        # 绘制 Heatmap
        if plot_heatmap and heatmap_data:
            heatmap_path = str(self.output_dir / "parameter_heatmap.png")
            self.plot_heatmap(heatmap_data, save_path=heatmap_path)
        
        return {
            "all_results": results,
            "best_by_return_draw": best_by_return_draw,
            "best_by_sharpe": best_by_sharpe,
            "heatmap_data": heatmap_data,
        }


def main() -> None:
    """主入口函数。"""
    parser = argparse.ArgumentParser(description="Parameter Sensitivity Scanner")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/parquet/features_latest.parquet",
        help="Path to features Parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/stock_model.txt",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/plots",
        help="Output directory",
    )
    parser.add_argument(
        "--min-hold-days",
        type=str,
        default="3,5,8,10",
        help="Min hold days values (comma-separated)",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        default="0.01,0.015,0.02,0.025",
        help="Threshold values (comma-separated)",
    )
    parser.add_argument(
        "--max-positions",
        type=str,
        default="3,5,10",
        help="Max positions values (comma-separated)",
    )
    parser.add_argument(
        "--dynamic-position",
        action="store_true",
        help="Use dynamic position sizing",
    )
    parser.add_argument(
        "--no-heatmap",
        action="store_true",
        help="Skip heatmap generation",
    )
    
    args = parser.parse_args()
    
    # 解析参数列表
    min_hold_days = [int(x) for x in args.min_hold_days.split(",")]
    threshold = [float(x) for x in args.threshold.split(",")]
    max_positions = [int(x) for x in args.max_positions.split(",")]
    
    logger.info("=" * 60)
    logger.info("Parameter Sensitivity Scanner")
    logger.info("=" * 60)
    
    # 初始化扫描器
    scanner = ParameterScanner(
        parquet_path=args.parquet,
        model_path=args.model,
        initial_capital=args.capital,
        output_dir=args.output,
    )
    
    # 设置参数范围
    scanner.set_parameter_range(
        min_hold_days=min_hold_days,
        threshold=threshold,
        max_positions=max_positions,
    )
    
    # 运行完整扫描
    results = scanner.run_full_scan(
        use_dynamic_position=args.dynamic_position,
        plot_heatmap=not args.no_heatmap,
    )
    
    # 输出最优参数
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDED PARAMETERS")
    logger.info("=" * 60)
    
    best = results["best_by_return_draw"]
    if best:
        logger.info(f"For maximum Return/Drawdown ratio:")
        logger.info(f"  min_hold_days: {best['min_hold_days']}")
        logger.info(f"  threshold: {best['threshold']:.4f}")
        logger.info(f"  max_positions: {best['max_positions']}")
        logger.info(f"  Expected annualized return: {best['annualized_return']:.2%}")
        logger.info(f"  Expected max drawdown: {best['max_drawdown']:.2%}")
        logger.info(f"  Expected Sharpe ratio: {best['sharpe_ratio']:.2f}")


if __name__ == "__main__":
    main()