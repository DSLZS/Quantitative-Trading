"""
IC 诊断脚本
分析为什么 T+1 IC 始终 < 0.05
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from loguru import logger

from src.backtest_engine import BacktestEngine
from src.alpha_model_v203 import AlphaModel


def diagnose_ic():
    """诊断 IC 计算"""
    
    # 1. 加载数据
    logger.info("=" * 70)
    logger.info("Step 1: Loading data...")
    logger.info("=" * 70)
    
    engine = BacktestEngine()
    
    # 只加载 2024 年数据（最快）
    df = engine.load_data(years=[2024], warmup_year=2023)
    logger.info(f"Loaded {len(df)} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # 2. 计算 T+1 收益率
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Computing T+1 returns...")
    logger.info("=" * 70)
    
    df = df.sort_values(['symbol', 'trade_date'])
    df['t1_return'] = df.groupby('symbol')['pct_chg'].shift(-1)
    
    # 检查 t1_return 分布
    valid_t1 = df['t1_return'].dropna()
    logger.info(f"T+1 return stats:")
    logger.info(f"  Valid: {len(valid_t1)}")
    logger.info(f"  Mean: {valid_t1.mean():.4f}")
    logger.info(f"  Std: {valid_t1.std():.4f}")
    logger.info(f"  Median: {valid_t1.median():.4f}")
    logger.info(f"  Min: {valid_t1.min():.4f}")
    logger.info(f"  Max: {valid_t1.max():.4f}")
    
    # 3. 计算 Alpha 评分
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Computing alpha scores...")
    logger.info("=" * 70)
    
    model = AlphaModel()
    scored_df = model.get_score(df.copy())
    
    # 4. 分析 score 和 t1_return 的关系
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Analyzing score vs t1_return...")
    logger.info("=" * 70)
    
    analysis_df = scored_df.dropna(subset=['t1_return', 'score']).copy()
    logger.info(f"Valid pairs: {len(analysis_df)}")
    
    # 全局相关性
    global_score = analysis_df['score']
    global_t1 = analysis_df['t1_return']
    
    spearman_corr, spearman_p = spearmanr(global_score, global_t1)
    pearson_corr, pearson_p = pearsonr(global_score, global_t1)
    
    logger.info(f"\nGlobal correlation (all data pooled):")
    logger.info(f"  Spearman: {spearman_corr:.6f} (p={spearman_p:.2e})")
    logger.info(f"  Pearson: {pearson_corr:.6f} (p={pearson_p:.2e})")
    
    # ========== 关键修复：对 T+1 收益率缩尾 ==========
    logger.info(f"\n" + "=" * 70)
    logger.info("Step 4b: Winsorizing T+1 returns (critical!)")
    logger.info("=" * 70)
    
    analysis_df['t1_return_winsorized'] = analysis_df['t1_return'].copy()
    t1_mean = analysis_df['t1_return_winsorized'].mean()
    t1_std = analysis_df['t1_return_winsorized'].std()
    lower = t1_mean - 3 * t1_std
    upper = t1_mean + 3 * t1_std
    analysis_df['t1_return_winsorized'] = analysis_df['t1_return_winsorized'].clip(lower, upper)
    
    winsorized_valid = analysis_df['t1_return_winsorized'].dropna()
    logger.info(f"After winsorization (3σ):")
    logger.info(f"  Valid: {len(winsorized_valid)}")
    logger.info(f"  Mean: {winsorized_valid.mean():.4f}")
    logger.info(f"  Std: {winsorized_valid.std():.4f}")
    logger.info(f"  Min: {winsorized_valid.min():.4f}")
    logger.info(f"  Max: {winsorized_valid.max():.4f}")
    
    # 截面 IC (逐日计算) - 使用原始 T+1
    logger.info(f"\nCross-sectional IC (daily, raw t1_return):")
    unique_dates = sorted(analysis_df['trade_date'].unique())
    logger.info(f"  Unique dates: {len(unique_dates)}")
    
    ic_values = []
    for date in unique_dates[:10]:  # 先检查前10天
        day_data = analysis_df[analysis_df['trade_date'] == date]
        if len(day_data) < 100:
            continue
        
        score_vals = day_data['score'].values
        t1_vals = day_data['t1_return'].values
        
        # Spearman rank IC
        sp_corr, sp_p = spearmanr(score_vals, t1_vals)
        
        # Pearson IC
        pr_corr, pr_p = pearsonr(score_vals, t1_vals)
        
        ic_values.append({
            'date': date,
            'n_stocks': len(day_data),
            'spearman_ic': sp_corr if not np.isnan(sp_corr) else 0,
            'pearson_ic': pr_corr if not np.isnan(pr_corr) else 0,
            'score_mean': score_vals.mean(),
            'score_std': score_vals.std(),
            't1_mean': t1_vals.mean(),
            't1_std': t1_vals.std(),
        })
        
        if len(ic_values) <= 5:
            logger.info(f"  Date {date}: n={len(day_data)}, "
                       f"Spearman IC={sp_corr:.6f}, Pearson IC={pr_corr:.6f}")
    
    ic_df = pd.DataFrame(ic_values)
    
    if len(ic_df) > 0:
        logger.info(f"\nIC Summary (first {len(ic_df)} days, raw t1_return):")
        logger.info(f"  Mean Spearman IC: {ic_df['spearman_ic'].mean():.6f}")
        logger.info(f"  Mean Pearson IC: {ic_df['pearson_ic'].mean():.6f}")
        logger.info(f"  IC Std: {ic_df['spearman_ic'].std():.6f}")
        logger.info(f"  IC IR: {ic_df['spearman_ic'].mean() / ic_df['spearman_ic'].std():.4f}")
    
    # 截面 IC (逐日计算) - 使用缩尾后的 T+1
    logger.info(f"\nCross-sectional IC (daily, winsorized t1_return):")
    
    ic_values_winsorized = []
    for date in unique_dates[:10]:
        day_data = analysis_df[analysis_df['trade_date'] == date]
        if len(day_data) < 100:
            continue
        
        score_vals = day_data['score'].values
        t1_vals = day_data['t1_return_winsorized'].values
        
        sp_corr, _ = spearmanr(score_vals, t1_vals)
        pr_corr, _ = pearsonr(score_vals, t1_vals)
        
        ic_values_winsorized.append({
            'date': date,
            'spearman_ic': sp_corr if not np.isnan(sp_corr) else 0,
            'pearson_ic': pr_corr if not np.isnan(pr_corr) else 0,
        })
        
        if len(ic_values_winsorized) <= 5:
            logger.info(f"  Date {date}: Spearman IC={sp_corr:.6f}, Pearson IC={pr_corr:.6f}")
    
    ic_df_w = pd.DataFrame(ic_values_winsorized)
    if len(ic_df_w) > 0:
        logger.info(f"\nIC Summary (winsorized t1_return):")
        logger.info(f"  Mean Spearman IC: {ic_df_w['spearman_ic'].mean():.6f}")
        logger.info(f"  Mean Pearson IC: {ic_df_w['pearson_ic'].mean():.6f}")
        logger.info(f"  IC Std: {ic_df_w['spearman_ic'].std():.6f}")
        logger.info(f"  IC IR: {ic_df_w['spearman_ic'].mean() / ic_df_w['spearman_ic'].std():.4f}")
    
    # 5. 检查因子原始值与 t1_return 的关系
    logger.info(f"\n" + "=" * 70)
    logger.info("Step 5: Factor raw values vs t1_return")
    logger.info("=" * 70)
    
    factor_cols = ['fund_flow_strength', 'intraday_buy_pressure', 'momentum_1d', 
                   'momentum_2d', 'volume_price_divergence']
    
    for col in factor_cols:
        if col in analysis_df.columns:
            valid = analysis_df[[col, 't1_return']].dropna()
            if len(valid) > 100:
                sp_corr, _ = spearmanr(valid[col], valid['t1_return'])
                pr_corr, _ = pearsonr(valid[col], valid['t1_return'])
                logger.info(f"  {col:30s}: Spearman={sp_corr:.6f}, Pearson={pr_corr:.6f}")
    
    # 6. 截面排名后的效果
    logger.info(f"\n" + "=" * 70)
    logger.info("Step 6: After cross-sectional ranking")
    logger.info("=" * 70)
    
    rank_cols = ['fund_flow_rank', 'intraday_rank', 'mom1d_rank', 
                 'mom2d_rank', 'volprice_rank', 'raw_score', 'score']
    
    for col in rank_cols:
        if col in analysis_df.columns:
            valid = analysis_df[[col, 't1_return']].dropna()
            if len(valid) > 100:
                sp_corr, _ = spearmanr(valid[col], valid['t1_return'])
                pr_corr, _ = pearsonr(valid[col], valid['t1_return'])
                logger.info(f"  {col:30s}: Spearman={sp_corr:.6f}, Pearson={pr_corr:.6f}")
    
    # 7. 分位数分析
    logger.info(f"\n" + "=" * 70)
    logger.info("Step 7: Quantile analysis (top vs bottom)")
    logger.info("=" * 70)
    
    for date in unique_dates[:5]:
        day_data = analysis_df[analysis_df['trade_date'] == date]
        if len(day_data) < 100:
            continue
        
        # 按 score 分5组
        day_data = day_data.copy()
        day_data['quantile'] = pd.qcut(day_data['score'], 5, labels=False, duplicates='drop')
        
        quantile_stats = day_data.groupby('quantile')['t1_return'].agg(['mean', 'std', 'count'])
        logger.info(f"\n  Date {date}:")
        logger.info(f"    Quantile means: {quantile_stats['mean'].to_dict()}")
        
        # Top - Bottom
        if 4 in quantile_stats.index and 0 in quantile_stats.index:
            top_bottom = quantile_stats.loc[4, 'mean'] - quantile_stats.loc[0, 'mean']
            logger.info(f"    Top - Bottom: {top_bottom:.6f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Diagnosis complete!")
    logger.info("=" * 70)


if __name__ == '__main__':
    diagnose_ic()