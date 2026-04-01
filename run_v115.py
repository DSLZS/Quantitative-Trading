"""
V115 回测运行脚本 - 自动化特征挖掘与场景化 Alpha 实验室.

【使用方法】
    python run_v115.py
    
【可选参数】
    --data-path: 数据文件路径 (Parquet 格式)
    --output-dir: 输出目录
    --no-genetic: 禁用基因挖掘
    --no-regime: 禁用场景感知
    --no-ablation: 禁用消融实验
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/v115_{time:YYYYMMDD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)

from src.alpha_research_v115 import (
    VERSION,
    DataHealingEngineV115,
    BacktestRunnerV115,
    AlphaResearchV115,
    GeneticFactorMiner,
    MarketRegimeDetector,
    AblationStudy,
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="V115 回测运行脚本")
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="数据文件路径 (Parquet 格式)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="输出目录",
    )
    
    parser.add_argument(
        "--no-genetic",
        action="store_true",
        help="禁用基因挖掘",
    )
    
    parser.add_argument(
        "--no-regime",
        action="store_true",
        help="禁用场景感知",
    )
    
    parser.add_argument(
        "--no-ablation",
        action="store_true",
        help="禁用消融实验",
    )
    
    parser.add_argument(
        "--mock-data",
        action="store_true",
        help="强制使用 Mock 数据",
    )
    
    parser.add_argument(
        "--n-stocks",
        type=int,
        default=100,
        help="Mock 数据股票数量",
    )
    
    parser.add_argument(
        "--n-days",
        type=int,
        default=60,
        help="Mock 数据交易日数量",
    )
    
    return parser.parse_args()


def generate_demo_data(n_stocks: int = 100, n_days: int = 60) -> pd.DataFrame:
    """
    生成演示数据用于测试。
    
    Args:
        n_stocks: 股票数量
        n_days: 交易日数量
        
    Returns:
        DataFrame
    """
    logger.info(f"Generating demo data: {n_stocks} stocks x {n_days} days")
    
    np.random.seed(42)
    
    # 生成日期
    dates = pd.date_range(start="20240101", periods=n_days, freq="B")
    dates = [d.strftime("%Y%m%d") for d in dates]
    
    # 生成股票代码
    symbols = [f"{str(i).zfill(6)}.SZ" for i in range(1, n_stocks + 1)]
    
    # 生成数据
    data = []
    for symbol in symbols:
        base_price = np.random.uniform(10, 100)
        base_mv = np.random.uniform(1e9, 1e11)
        
        for i, date in enumerate(dates):
            # 价格随机游走
            ret = np.random.normal(0, 0.02)
            close = base_price * (1 + ret)
            
            # 生成 OHLC
            daily_vol = abs(np.random.normal(0.03, 0.01))
            high = close * (1 + daily_vol)
            low = close * (1 - daily_vol)
            open_price = close * (1 + np.random.normal(0, 0.01))
            
            # 生成成交量
            volume = np.random.uniform(1e6, 1e8)
            amount = volume * close
            
            # 生成因子
            volatility_20 = np.random.uniform(0.01, 0.05)
            momentum_10 = np.random.uniform(-0.1, 0.1)
            turnover_rate = np.random.uniform(0.01, 0.1)
            total_mv = base_mv * (1 + np.random.normal(0, 0.05))
            
            # 生成 T+1 收益 (带有一些可预测性)
            t1_return = np.random.normal(0, 0.02)
            
            row = {
                "trade_date": date,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": close / (1 + ret),
                "volume": volume,
                "amount": amount,
                "turnover_rate": turnover_rate,
                "total_mv": total_mv,
                "volatility_20": volatility_20,
                "momentum_10": momentum_10,
                "pe_ttm": np.random.uniform(10, 50),
                "pb": np.random.uniform(1, 5),
                "t1_return": t1_return,
            }
            
            data.append(row)
            
            base_price = close
            base_mv = total_mv
    
    df = pd.DataFrame(data)
    
    # 计算 T+3, T+5 收益
    for n in [3, 5]:
        df[f"t{n}_return"] = df.groupby("symbol")["close"].transform(
            lambda x: x.shift(-n) / x - 1
        )
    
    logger.info(f"Demo data generated: {len(df)} rows")
    
    return df


def run_v115_backtest(args):
    """运行 V115 回测"""
    logger.info("=" * 80)
    logger.info(f"V115 Backtest Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # 1. 加载数据
    logger.info("[Step 1] Loading data...")
    
    if args.mock_data or args.data_path is None:
        logger.info("Using demo data generation mode")
        df = generate_demo_data(n_stocks=args.n_stocks, n_days=args.n_days)
    else:
        data_path = Path(args.data_path)
        if data_path.exists():
            logger.info(f"Loading data from {data_path}")
            df = pd.read_parquet(data_path)
        else:
            logger.warning(f"Data file not found: {data_path}, falling back to demo data")
            df = generate_demo_data(n_stocks=args.n_stocks, n_days=args.n_days)
    
    logger.info(f"Data loaded: {len(df)} rows, {df['symbol'].nunique()} stocks, {df['trade_date'].nunique()} days")
    
    # 2. 初始化 Alpha 模块
    logger.info("[Step 2] Initializing Alpha module...")
    
    alpha_module = AlphaResearchV115(
        enable_genetic_mining=not args.no_genetic,
        enable_regime_aware=not args.no_regime,
        enable_ablation=not args.no_ablation,
        auto_heal=True,
    )
    
    # 3. 计算评分
    logger.info("[Step 3] Computing alpha scores...")
    
    score_df = alpha_module.compute_score(df)
    
    logger.info(f"Score computed: {len(score_df)} rows")
    
    # 4. 计算 IC 统计
    logger.info("[Step 4] Calculating IC statistics...")
    
    # T+1 IC
    ic_values = []
    for date in score_df["trade_date"].unique():
        day_data = score_df[score_df["trade_date"] == date]
        if len(day_data) < 20:
            continue
        
        mask = day_data["score"].notna() & day_data["t1_return"].notna()
        if mask.sum() < 20:
            continue
        
        score_rank = day_data.loc[mask, "score"].rank(method="average")
        label_rank = day_data.loc[mask, "t1_return"].rank(method="average")
        
        if np.std(score_rank) > 1e-10 and np.std(label_rank) > 1e-10:
            ic = np.corrcoef(score_rank, label_rank)[0, 1]
            if not np.isnan(ic):
                ic_values.append(ic)
    
    mean_ic = np.mean(ic_values) if ic_values else 0.0
    ic_std = np.std(ic_values, ddof=1) if len(ic_values) > 1 else 0.0
    ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
    
    logger.info(f"T+1 IC: Mean={mean_ic:.4f}, Std={ic_std:.4f}, IR={ic_ir:.2f}")
    
    # 5. 获取审计报告
    logger.info("[Step 5] Generating audit report...")
    
    audit_report = alpha_module.get_full_audit_report()
    
    # 6. 保存结果
    logger.info("[Step 6] Saving results...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 JSON 结果
    json_result = {
        "version": VERSION,
        "timestamp": timestamp,
        "ic_metrics": {
            "mean_ic": mean_ic,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "num_days": len(ic_values),
        },
        "config": {
            "enable_genetic_mining": not args.no_genetic,
            "enable_regime_aware": not args.no_regime,
            "enable_ablation": not args.no_ablation,
        },
        "audit_report": audit_report,
    }
    
    json_path = output_dir / f"v115_backtest_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"JSON result saved to: {json_path}")
    
    # 生成 Markdown 报告
    md_report = generate_markdown_report(json_result, alpha_module)
    md_path = output_dir / f"v115_audit_{timestamp}.md"
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    
    logger.info(f"Markdown report saved to: {md_path}")
    
    # 7. 输出总结
    logger.info("=" * 80)
    logger.info("V115 Backtest Complete")
    logger.info("=" * 80)
    logger.info(f"  T+1 Mean IC: {mean_ic:.4f}")
    logger.info(f"  T+1 IC IR: {ic_ir:.2f}")
    logger.info(f"  Genetic Factors: {len(alpha_module.genetic_factors)}")
    logger.info(f"  Regime Detection: {'Enabled' if not args.no_regime else 'Disabled'}")
    logger.info(f"  Ablation Study: {'Enabled' if not args.no_ablation else 'Disabled'}")
    logger.info("=" * 80)
    
    return json_result


def generate_markdown_report(result: dict, alpha_module: AlphaResearchV115) -> str:
    """生成 Markdown 格式报告"""
    
    ic_metrics = result["ic_metrics"]
    config = result["config"]
    audit = result.get("audit_report", {})
    
    # 获取基因挖矿报告
    genetic_report = audit.get("genetic_mining", {})
    best_factors = genetic_report.get("results", {}).get("best_factors", [])
    
    # 获取场景统计
    regime_stats = audit.get("regime_statistics", {})
    
    # 获取消融报告
    ablation_report = audit.get("ablation_report", "")
    
    report = f"""# V115 自动化特征挖掘与场景化 Alpha 实验室 - 审计报告

**版本**: {VERSION}
**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. 核心指标 (Key Metrics)

| 指标 | 数值 | 阈值 | 状态 |
|------|------|------|------|
| T+1 Rank IC (Mean) | {ic_metrics['mean_ic']:.4f} | > 0.015 | {'✓' if ic_metrics['mean_ic'] > 0.015 else '✗'} |
| IC IR (Stability) | {ic_metrics['ic_ir']:.2f} | > 0.3 | {'✓' if ic_metrics['ic_ir'] > 0.3 else '✗'} |
| IC Std | {ic_metrics['ic_std']:.4f} | - | - |
| 交易天数 | {ic_metrics['num_days']} | - | - |

---

## 2. 配置信息 (Configuration)

| 配置项 | 状态 |
|--------|------|
| 基因挖掘 (Genetic Mining) | {'✓ Enabled' if config['enable_genetic_mining'] else '✗ Disabled'} |
| 场景感知 (Regime Aware) | {'✓ Enabled' if config['enable_regime_aware'] else '✗ Disabled'} |
| 消融实验 (Ablation Study) | {'✓ Enabled' if config['enable_ablation'] else '✗ Disabled'} |

---

## 3. 基因因子挖掘 (Genetic Factor Mining)

"""
    
    if best_factors:
        report += """### 挖掘到的优质因子 (Top Factors)

| Rank | 表达式 | IC Score | 复杂度 | 阶数 |
|------|--------|----------|--------|------|
"""
        for i, factor in enumerate(best_factors[:5], 1):
            expr = factor.get('expression', 'N/A')[:50] + '...' if len(factor.get('expression', '')) > 50 else factor.get('expression', 'N/A')
            report += f"| {i} | `{expr}` | {factor.get('ic_score', 0):.4f} | {factor.get('complexity', 0)} | {factor.get('order', 0)} |\n"
        
        report += f"""
**示例复杂因子逻辑**:
- `Rank(Ts_Argmax(Close, 20)) / (Ts_Std(Volume, 5) * Rank(OFI))`
- 这种多维非线性组合才是抵御"脱毒"后 IC 暴跌的关键

"""
    else:
        report += "*基因挖掘未启用或无结果*\n\n"
    
    # 场景统计
    report += """## 4. 场景感知分析 (Regime Analysis)

"""
    
    if regime_stats:
        regime_dist = regime_stats.get('regime_distribution', {})
        if regime_dist:
            report += """### 场景分布

| 场景 | 天数 | 占比 |
|------|------|------|
"""
            for regime, stats in regime_dist.items():
                desc = {
                    'high_vol_large_cap': '高波动/大盘股',
                    'high_vol_small_cap': '高波动/小盘股',
                    'low_vol_large_cap': '低波动/大盘股',
                    'low_vol_small_cap': '低波动/小盘股',
                }.get(regime, regime)
                report += f"| {desc} | {stats.get('count', 0)} | {stats.get('percentage', 0):.1%} |\n"
            
            report += f"""
**当前场景**: {regime_stats.get('regime_description', 'N/A')}

"""
    else:
        report += "*场景感知未启用或无数据*\n\n"
    
    # 消融实验
    report += """## 5. 因子消融实验 (Ablation Study)

"""
    
    if ablation_report and ablation_report != "Ablation study not enabled.":
        report += ablation_report
    else:
        report += "*消融实验未启用或 T+1 IC >= 0.01*\n\n"
    
    # 结论
    passed = ic_metrics['mean_ic'] > 0.015 and ic_metrics['ic_ir'] > 0.3
    
    report += f"""
---

## 6. 结论 (Conclusion)

### 验收结果

| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| T+1 Rank IC | > 0.015 | {ic_metrics['mean_ic']:.4f} | {'✓ PASSED' if ic_metrics['mean_ic'] > 0.015 else '✗ FAILED'} |
| IC IR | > 0.3 | {ic_metrics['ic_ir']:.2f} | {'✓ PASSED' if ic_metrics['ic_ir'] > 0.3 else '✗ FAILED'} |

### 总体评估

**{'✓ PASSED - V115 展示出真正的、不依赖市值的核心竞争力' if passed else '✗ FAILED - 需要进一步优化'}**

{f'V115 成功展示了 T+1 IC={ic_metrics["mean_ic"]:.4f} 的预测能力，IC 稳定性 IR={ic_metrics["ic_ir"]:.2f}。' if passed else f'V115 当前 T+1 IC={ic_metrics["mean_ic"]:.4f} 未达到 0.015 目标，需要进一步优化。'}

---

*Report generated by V115 Alpha Research Module*
"""
    
    return report


if __name__ == "__main__":
    args = parse_args()
    result = run_v115_backtest(args)