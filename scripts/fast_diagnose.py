"""
Fast Diagnostic Script - Test Hypotheses Without Full Backtest
=============================================================
This script computes IC directly from scores without running the full backtest,
to verify hypotheses about industry neutralization, rank labels, and market cap.
"""

import sys
import os
import time
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from loguru import logger
from sqlalchemy import create_engine, text
import numpy as np

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Database config - match .env variable names
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3306)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "quantitative_trading"),
}


def get_db_url():
    return f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"


def load_data_with_industry(years, db_engine):
    """Load stock data WITH industry classification"""
    year_filter = ", ".join(str(y) for y in years)
    
    query = f"""
    SELECT 
        t.symbol AS ts_code, t.trade_date, t.close, t.open, t.high, t.low, 
        t.volume, t.amount,
        t.industry_code AS industry,
        t.total_mv
    FROM stock_daily t
    WHERE YEAR(t.trade_date) IN ({year_filter})
    ORDER BY t.trade_date, t.symbol
    """
    
    logger.info(f"[Data] Loading stock data with industry for years: {years}")
    start = time.time()
    
    df = pl.read_database(query, db_engine)
    
    # Cast decimal columns to float64 for Polars compatibility
    for col in ['close', 'open', 'high', 'low', 'volume', 'amount', 'total_mv']:
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Float64))
    
    logger.info(f"[Data] Loaded {len(df)} rows, {df['ts_code'].n_unique()} symbols in {time.time()-start:.1f}s")
    
    return df


def compute_features(df):
    """Compute features: ret_5d, vol_20d, vol_adj_momentum using LazyFrame"""
    logger.info(f"[Features] Computing features for {len(df)} rows...")
    start = time.time()
    
    # Use LazyFrame for proper window function support
    lf = df.lazy().sort(["ts_code", "trade_date"])
    
    # Compute features using LazyFrame
    lf = lf.with_columns([
        pl.col("close").shift(5).over("ts_code").alias("close_5d"),
        pl.col("close").shift(20).over("ts_code").alias("close_20d"),
        (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1).alias("daily_ret"),
    ])
    
    lf = lf.with_columns([
        (pl.col("close") / pl.col("close_5d") - 1).alias("ret_5d"),
        pl.col("daily_ret").rolling_std(window_size=20, min_periods=10).over("ts_code").alias("vol_20d"),
        ((pl.col("close") / pl.col("close_20d") - 1) / (pl.col("daily_ret").rolling_std(window_size=20, min_periods=10).over("ts_code") + 0.001)).alias("vol_adj_momentum"),
        (pl.col("close").shift(-1).over("ts_code") / pl.col("close") - 1).alias("t1_return"),
    ])
    
    # Collect and drop intermediates
    df = lf.drop(["close_5d", "close_20d", "daily_ret"]).collect()
    df = df.drop_nulls(subset=["ret_5d", "vol_20d", "vol_adj_momentum", "t1_return"])
    
    logger.info(f"[Features] Computed {len(df)} valid rows in {time.time()-start:.1f}s")
    return df


def cross_sectional_zscore(df, feature_cols):
    """Apply cross-sectional z-score normalization"""
    logger.info(f"[ZScore] Applying cross-sectional z-score to {feature_cols}")
    
    for c in feature_cols:
        mean_col = f"{c}_mean"
        std_col = f"{c}_std"
        df = df.with_columns([
            pl.col(c).mean().over("trade_date").alias(mean_col),
            pl.col(c).std().over("trade_date").alias(std_col),
        ])
        df = df.with_columns([
            ((pl.col(c) - pl.col(mean_col)) / (pl.col(std_col) + 1e-8)).alias(f"{c}_z")
        ])
        df = df.drop([mean_col, std_col])
    
    return df


def industry_neutralize(df, feature_cols):
    """Subtract industry mean from each feature (Hypothesis A)"""
    logger.info(f"[Industry] Applying industry neutralization to {feature_cols}")
    
    # Fill null industries with "Unknown"
    df = df.with_columns([
        pl.col("industry").fill_null("Unknown").cast(pl.Utf8),
    ])
    
    # For each feature, subtract industry mean using window functions instead of join
    for col in feature_cols:
        z_col = f"{col}_z"
        df = df.with_columns([
            (pl.col(z_col) - pl.col(z_col).mean().over("trade_date", "industry")).alias(f"{col}_in")
        ])
    
    return df


def compute_ic(df, score_col, label_col="t1_return"):
    """Compute daily IC (correlation between score and next-day return)"""
    df_valid = df.drop_nulls(subset=[score_col, label_col])
    
    if len(df_valid) == 0:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "pos_ratio": 0.0, "num_days": 0}
    
    daily_ic = df_valid.group_by("trade_date").agg([
        pl.corr(pl.col(score_col), pl.col(label_col)).alias("ic"),
        pl.len().alias("stock_count"),
    ])
    daily_ic = daily_ic.drop_nulls(subset=["ic"])
    
    if len(daily_ic) == 0:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "pos_ratio": 0.0, "num_days": 0}
    
    mean_ic = daily_ic["ic"].mean()
    ic_std = daily_ic["ic"].std()
    if ic_std is None:
        ic_std = 1e-8
    icir = mean_ic / (ic_std + 1e-8)
    pos_ratio = (daily_ic["ic"] > 0).sum() / len(daily_ic)
    
    return {
        "mean_ic": float(mean_ic),
        "ic_std": float(ic_std),
        "ic_ir": float(icir),
        "pos_ratio": float(pos_ratio),
        "num_days": len(daily_ic),
    }


def main():
    print("=" * 80)
    print("Fast Diagnostic Script - Hypothesis Testing")
    print("=" * 80)
    
    # Load data for single year test (2022 - the middle year)
    years = [2022]
    
    db_url = get_db_url()
    engine = create_engine(db_url, pool_size=5, max_overflow=10)
    
    # Step 1: Load data
    print("\n[Step 1] Loading data with industry...")
    df = load_data_with_industry(years, engine)
    print(f"  -> Loaded {len(df)} rows, {df['ts_code'].n_unique()} symbols")
    
    # Step 2: Compute features
    print("\n[Step 2] Computing features...")
    df = compute_features(df)
    print(f"  -> Features computed: {len(df)} valid rows")
    
    # Step 3: Cross-sectional z-score
    print("\n[Step 3] Applying cross-sectional z-score...")
    feature_cols = ["ret_5d", "vol_20d", "vol_adj_momentum"]
    df = cross_sectional_zscore(df, feature_cols)
    
    # Step 4: Compute composite score with V222 weights
    print("\n[Step 4] Computing composite scores...")
    # V222 weights (NORMAL regime as default)
    weights = {"ret_5d": -0.4, "vol_20d": -0.1, "vol_adj_momentum": 0.5}
    
    # Compute weighted columns
    for c in feature_cols:
        w = weights[c]
        df = df.with_columns([
            (pl.col(f"{c}_z") * w).alias(f"weighted_{c}")
        ])
    
    # Build score column by column
    score_expr = pl.col(f"weighted_{feature_cols[0]}")
    for c in feature_cols[1:]:
        score_expr = score_expr + pl.col(f"weighted_{c}")
    df = df.with_columns([
        score_expr.alias("score_base")
    ])
    
    # Test Hypothesis A: Industry Neutralization
    print("\n[Step 5] Testing Hypothesis A: Industry Neutralization...")
    df_ind = industry_neutralize(df.clone(), feature_cols)
    
    # Compute base score IC
    ic_base = compute_ic(df, "score_base")
    print(f"\n  BASELINE IC (V222-style):")
    print(f"    Mean IC: {ic_base['mean_ic']:.6f}")
    print(f"    IC Std:  {ic_base['ic_std']:.6f}")
    print(f"    ICIR:    {ic_base['ic_ir']:.6f}")
    print(f"    Pos%:    {ic_base['pos_ratio']:.4f}")
    
    # Compute industry-neutralized IC
    ic_ind = compute_ic(df_ind, "ret_5d_in")  # Use first neutralized feature
    print(f"\n  HYPOTHESIS A (Industry Neutral):")
    print(f"    Mean IC: {ic_ind['mean_ic']:.6f}")
    print(f"    IC Std:  {ic_ind['ic_std']:.6f}")
    print(f"    ICIR:    {ic_ind['ic_ir']:.6f}")
    print(f"    Pos%:    {ic_ind['pos_ratio']:.4f}")
    
    ic_diff = ic_ind['mean_ic'] - ic_base['mean_ic']
    print(f"    IC Improvement: {ic_diff:+.6f}")
    
    if abs(ic_diff) > 0.01:
        print("    -> HYPOTHESIS A PASSED (IC improvement > 0.01)")
    else:
        print("    -> HYPOTHESIS A FAILED (IC improvement < 0.01)")
    
    # Step 6: Test Hypothesis B: Rank-based scoring
    print("\n[Step 6] Testing Hypothesis B: Rank-based Scoring...")
    # Compute rank-based score
    df_rank = df.clone()
    for c in feature_cols:
        df_rank = df_rank.with_columns([
            pl.col(f"{c}_z").rank().over("trade_date").alias(f"{c}_rank")
        ])
    
    # Compute composite rank score
    first = True
    for c in feature_cols:
        w = weights[c]
        if first:
            df_rank = df_rank.with_columns([
                (pl.col(f"{c}_rank") * w).alias("score_rank")
            ])
            first = False
        else:
            df_rank = df_rank.with_columns([
                (pl.col("score_rank") + pl.col(f"{c}_rank") * w).alias("score_rank")
            ])
    
    ic_rank = compute_ic(df_rank, "score_rank")
    print(f"\n  HYPOTHESIS B (Rank-based):")
    print(f"    Mean IC: {ic_rank['mean_ic']:.6f}")
    print(f"    IC Std:  {ic_rank['ic_std']:.6f}")
    print(f"    ICIR:    {ic_rank['ic_ir']:.6f}")
    print(f"    Pos%:    {ic_rank['pos_ratio']:.4f}")
    
    ic_diff_b = ic_rank['mean_ic'] - ic_base['mean_ic']
    print(f"    IC Improvement: {ic_diff_b:+.6f}")
    
    if abs(ic_diff_b) > 0.01:
        print("    -> HYPOTHESIS B PASSED (IC improvement > 0.01)")
    else:
        print("    -> HYPOTHESIS B FAILED (IC improvement < 0.01)")
    
    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Mean IC':>10} {'ICIR':>10} {'Status':<20}")
    print("-" * 80)
    print(f"{'Baseline (V222)':<30} {ic_base['mean_ic']:>10.6f} {ic_base['ic_ir']:>10.6f} {'Reference':<20}")
    print(f"{'Hypo A (Industry Neutral)':<30} {ic_ind['mean_ic']:>10.6f} {ic_ind['ic_ir']:>10.6f} {'PASSED' if abs(ic_diff)>0.01 else 'FAILED':<20}")
    print(f"{'Hypo B (Rank-based)':<30} {ic_rank['mean_ic']:>10.6f} {ic_rank['ic_ir']:>10.6f} {'PASSED' if abs(ic_diff_b)>0.01 else 'FAILED':<20}")
    
    # Step 7: Test Hypothesis C: Market Cap factor
    print("\n[Step 7] Testing Hypothesis C: Market Cap Factor...")
    if 'total_mv' in df.columns:
        df_mc = df.clone()
        # Log transform market cap and z-score normalize cross-sectionally (by trade_date)
        df_mc = df_mc.with_columns([
            pl.col("total_mv").log10().alias("log_mv"),
        ])
        # Cross-sectional z-score (group by trade_date)
        df_mc = df_mc.with_columns([
            pl.col("log_mv").mean().over("trade_date").alias("log_mv_mean"),
            pl.col("log_mv").std().over("trade_date").alias("log_mv_std"),
        ])
        df_mc = df_mc.with_columns([
            ((pl.col("log_mv") - pl.col("log_mv_mean")) / (pl.col("log_mv_std") + 1e-8)).alias("log_mv_z"),
        ])
        df_mc = df_mc.drop(["log_mv_mean", "log_mv_std", "log_mv"])
        # Compute IC of market cap alone
        ic_mv = compute_ic(df_mc, "log_mv_z")
        print(f"\n  HYPOTHESIS C (Market Cap Factor Alone):")
        print(f"    Mean IC: {ic_mv['mean_ic']:.6f}")
        print(f"    IC Std:  {ic_mv['ic_std']:.6f}")
        print(f"    ICIR:    {ic_mv['ic_ir']:.6f}")
        print(f"    Pos%:    {ic_mv['pos_ratio']:.4f}")
        
        # Test adding market cap to composite score
        df_mc_combined = df_mc.with_columns([
            (pl.col("score_base") + pl.col("log_mv_z") * 0.2).alias("score_mc_combined")
        ])
        ic_mc_combined = compute_ic(df_mc_combined, "score_mc_combined")
        print(f"\n  HYPOTHESIS C (Market Cap + Baseline Combined):")
        print(f"    Mean IC: {ic_mc_combined['mean_ic']:.6f}")
        print(f"    IC Std:  {ic_mc_combined['ic_std']:.6f}")
        print(f"    ICIR:    {ic_mc_combined['ic_ir']:.6f}")
        print(f"    Pos%:    {ic_mc_combined['pos_ratio']:.4f}")
        
        ic_diff_c = ic_mc_combined['mean_ic'] - ic_base['mean_ic']
        print(f"    IC Improvement: {ic_diff_c:+.6f}")
        
        if abs(ic_diff_c) > 0.01:
            print("    -> HYPOTHESIS C PASSED (IC improvement > 0.01)")
        else:
            print("    -> HYPOTHESIS C FAILED (IC improvement < 0.01)")
        
        results = {
            "baseline": ic_base,
            "hypothesis_A_industry_neutral": {**ic_ind, "ic_diff": ic_diff},
            "hypothesis_B_rank": {**ic_rank, "ic_diff": ic_diff_b},
            "hypothesis_C_market_cap": {**ic_mv, "note": "market cap alone"},
            "hypothesis_C_combined": {**ic_mc_combined, "ic_diff": ic_diff_c},
            "year": 2022,
        }
    else:
        print("  -> HYPOTHESIS C: No total_mv column available, skipping")
        ic_mv = {"mean_ic": 0, "ic_std": 0, "ic_ir": 0, "pos_ratio": 0}
        ic_mc_combined = {"mean_ic": 0, "ic_std": 0, "ic_ir": 0, "pos_ratio": 0}
        ic_diff_c = 0
        results = {
            "baseline": ic_base,
            "hypothesis_A_industry_neutral": {**ic_ind, "ic_diff": ic_diff},
            "hypothesis_B_rank": {**ic_rank, "ic_diff": ic_diff_b},
            "year": 2022,
        }
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "fast_diagnose_result.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()