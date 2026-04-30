#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grid Search for Optimal Dynamic Weights in V222
Searches for weights that maximize IC across 2020, 2022, 2024
Uses the .venv Python which has all dependencies installed.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from loguru import logger
import json

load_dotenv()


def compute_features(df):
    """Compute the 3 features used in V222 using pandas"""
    df = df.sort_values(['symbol', 'trade_date'])
    
    results = []
    for symbol, group in df.groupby('symbol'):
        group = group.copy()
        # 5-day return (reversal signal)
        group['ret_5d'] = group['close'] / group['close'].shift(5) - 1
        # 20-day volatility
        group['vol_20d'] = group['close'].pct_change().rolling(20).std()
        # 20-day return
        group['ret_20d'] = group['close'] / group['close'].shift(20) - 1
        # Next day return (for IC computation)
        group['next_1d_ret'] = group['close'].shift(-1) / group['close'] - 1
        results.append(group)
    
    df = pd.concat(results, ignore_index=True)
    
    # vol_adj_momentum = ret_20d / vol_20d
    df['vol_adj_momentum'] = df['ret_20d'] / df['vol_20d']
    
    return df.dropna(subset=['ret_5d', 'vol_20d', 'vol_adj_momentum', 'next_1d_ret'])


def cross_sectional_zscore(df, col):
    """Apply cross-sectional z-score per date for a single column"""
    def zscore_group(g):
        mean = g.mean()
        std = g.std()
        if std > 1e-10:
            return (g - mean) / std
        else:
            return g - mean
    
    return df.groupby('trade_date')[col].transform(zscore_group)


def compute_volatility_state(df):
    """Classify each date into HIGH_VOL / LOW_VOL / NORMAL based on cross-sectional vol quantile"""
    # Compute average vol_20d per date
    daily_avg_vol = df.groupby('trade_date')['vol_20d'].mean().reset_index()
    daily_avg_vol.columns = ['trade_date', 'avg_vol']
    daily_avg_vol = daily_avg_vol.sort_values('trade_date').reset_index(drop=True)
    
    # Compute rolling 60-day vol quantile
    rolling_q70 = daily_avg_vol['avg_vol'].rolling(60, min_periods=20).quantile(0.7)
    rolling_q30 = daily_avg_vol['avg_vol'].rolling(60, min_periods=20).quantile(0.3)
    
    daily_avg_vol['vol_q70'] = rolling_q70
    daily_avg_vol['vol_q30'] = rolling_q30
    
    # Merge back
    df = df.merge(daily_avg_vol, on='trade_date', how='left')
    
    # Classify: compare daily avg_vol to its own rolling distribution
    df['market_state'] = 'NORMAL'
    df.loc[df['avg_vol'] > df['vol_q70'], 'market_state'] = 'HIGH_VOL'
    df.loc[df['avg_vol'] < df['vol_q30'], 'market_state'] = 'LOW_VOL'
    
    return df


def apply_weights_and_compute_ic(df, weights_high, weights_low, weights_normal):
    """Apply weights based on market state and compute IC"""
    df = df.copy()
    
    # Compute score based on market state
    score = np.zeros(len(df))
    
    high_mask = df['market_state'] == 'HIGH_VOL'
    low_mask = df['market_state'] == 'LOW_VOL'
    normal_mask = ~high_mask & ~low_mask
    
    if high_mask.any():
        w = weights_high
        score[high_mask] = (
            w['ret_5d'] * df.loc[high_mask, 'z_ret_5d'].values +
            w['vol_20d'] * df.loc[high_mask, 'z_vol_20d'].values +
            w['vol_adj_momentum'] * df.loc[high_mask, 'z_vol_adj_momentum'].values
        )
    
    if low_mask.any():
        w = weights_low
        score[low_mask] = (
            w['ret_5d'] * df.loc[low_mask, 'z_ret_5d'].values +
            w['vol_20d'] * df.loc[low_mask, 'z_vol_20d'].values +
            w['vol_adj_momentum'] * df.loc[low_mask, 'z_vol_adj_momentum'].values
        )
    
    if normal_mask.any():
        w = weights_normal
        score[normal_mask] = (
            w['ret_5d'] * df.loc[normal_mask, 'z_ret_5d'].values +
            w['vol_20d'] * df.loc[normal_mask, 'z_vol_20d'].values +
            w['vol_adj_momentum'] * df.loc[normal_mask, 'z_vol_adj_momentum'].values
        )
    
    df['score'] = score
    
    # Compute IC per date
    ics = []
    for date, group in df.groupby('trade_date'):
        if len(group) < 50:
            continue
        score_arr = group['score'].values
        ret_arr = group['next_1d_ret'].values
        
        # Filter out NaN
        valid = ~(np.isnan(score_arr) | np.isnan(ret_arr))
        if valid.sum() < 30:
            continue
        
        ic = np.corrcoef(score_arr[valid], ret_arr[valid])[0, 1]
        if not np.isnan(ic):
            ics.append(ic)
    
    if len(ics) == 0:
        return 0.0, 0.0
    
    mean_ic = np.mean(ics)
    ic_std = np.std(ics)
    ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
    
    return mean_ic, ic_ir


def main():
    logger.info("=" * 60)
    logger.info("V222 Grid Search for Optimal Weights")
    logger.info("=" * 60)
    
    # Load data
    db_url = os.getenv('DATABASE_URL')
    engine = create_engine(db_url)
    
    # Load only necessary columns for speed
    query = """
        SELECT symbol, trade_date, close
        FROM stock_daily 
        WHERE trade_date >= 20190101 AND trade_date <= 20241231
        ORDER BY symbol, trade_date
    """
    
    logger.info("Loading data from MySQL...")
    df = pd.read_sql(query, engine)
    engine.dispose()
    logger.info(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Compute features
    logger.info("Computing features...")
    df = compute_features(df)
    logger.info(f"After feature computation: {len(df)} rows")
    
    # Cross-sectional z-score
    logger.info("Computing cross-sectional z-scores...")
    df['z_ret_5d'] = cross_sectional_zscore(df, 'ret_5d')
    df['z_vol_20d'] = cross_sectional_zscore(df, 'vol_20d')
    df['z_vol_adj_momentum'] = cross_sectional_zscore(df, 'vol_adj_momentum')
    
    # Compute market state
    logger.info("Computing market states...")
    df = compute_volatility_state(df)
    logger.info(f"Market state distribution:")
    logger.info(f"  HIGH_VOL: {(df['market_state'] == 'HIGH_VOL').sum()}")
    logger.info(f"  LOW_VOL: {(df['market_state'] == 'LOW_VOL').sum()}")
    logger.info(f"  NORMAL: {(df['market_state'] == 'NORMAL').sum()}")
    
    # Separate by year for evaluation
    # Convert trade_date to int if it's a date object
    if isinstance(df['trade_date'].iloc[0], (pd.Timestamp,)):
        df['trade_date_int'] = df['trade_date'].dt.year * 10000 + df['trade_date'].dt.month * 100 + df['trade_date'].dt.day
    elif isinstance(df['trade_date'].iloc[0], int):
        df['trade_date_int'] = df['trade_date']
    else:
        # Assume it's a date object
        df['trade_date_int'] = df['trade_date'].apply(lambda x: x.year * 10000 + x.month * 100 + x.day)
    
    years_data = {}
    for year in [2020, 2022, 2024]:
        mask = (df['trade_date_int'] // 10000 == year)
        years_data[year] = df[mask].copy()
    
    logger.info(f"Data prepared: 2020={len(years_data[2020])}, 2022={len(years_data[2022])}, 2024={len(years_data[2024])}")
    
    # Grid search parameters
    # Weights must sum to 0 (for balance)
    # ret_5d: typically negative (reversal)
    # vol_20d: typically negative (low vol preferred)  
    # vol_adj_momentum: typically positive (momentum)
    
    ret_weights = np.arange(-0.7, -0.05, 0.1)  # -0.7 to -0.1
    vol_weights = np.arange(-0.5, -0.05, 0.1)  # -0.5 to -0.1
    
    # For each ret and vol, compute momentum weight to make sum = 0
    # momentum_weight = -(ret + vol)
    
    best_overall_score = -999
    best_configs = []
    
    logger.info("Starting grid search...")
    
    # Search with different patterns for HIGH_VOL, LOW_VOL, NORMAL
    configs_to_test = []
    
    for ret_w in ret_weights:
        for vol_w in vol_weights:
            mom_w = -(ret_w + vol_w)  # Sum to 0
            
            # Test different patterns for HIGH_VOL, LOW_VOL
            for scale in [0.5, 0.7, 1.0, 1.3, 1.5]:
                high_ret = ret_w * scale
                high_vol = vol_w * scale
                high_mom = mom_w * scale
                
                # For LOW_VOL, less reversal in calm markets
                low_scale = 1.0 / scale if scale > 0 else 1.0
                low_ret = ret_w * low_scale * 0.5
                low_vol = vol_w * low_scale * 0.5
                low_mom = -(low_ret + low_vol)
                
                # NORMAL: average
                norm_ret = (high_ret + low_ret) / 2
                norm_vol = (high_vol + low_vol) / 2
                norm_mom = -(norm_ret + norm_vol)
                
                configs_to_test.append({
                    'high': {'ret_5d': high_ret, 'vol_20d': high_vol, 'vol_adj_momentum': high_mom},
                    'low': {'ret_5d': low_ret, 'vol_20d': low_vol, 'vol_adj_momentum': low_mom},
                    'normal': {'ret_5d': norm_ret, 'vol_20d': norm_vol, 'vol_adj_momentum': norm_mom},
                })
    
    logger.info(f"Testing {len(configs_to_test)} configurations...")
    
    for i, config in enumerate(configs_to_test):
        if i % 200 == 0:
            logger.info(f"Progress: {i}/{len(configs_to_test)}")
        
        total_ic = 0
        total_ir = 0
        passed_years = 0
        
        for year, data in years_data.items():
            if len(data) == 0:
                continue
            mean_ic, ic_ir = apply_weights_and_compute_ic(
                data,
                config['high'], config['low'], config['normal']
            )
            total_ic += mean_ic
            total_ir += ic_ir
            if mean_ic > 0.05 and ic_ir > 0.6:
                passed_years += 1
        
        # Score: prioritize mean IC, with bonus for passing thresholds
        score = total_ic / 3 + 0.1 * passed_years
        
        if score > best_overall_score:
            best_overall_score = score
            best_configs.append({
                'config': config,
                'total_ic': total_ic / 3,
                'total_ir': total_ir / 3,
                'passed_years': passed_years,
                'score': score
            })
    
    # Sort by score
    best_configs.sort(key=lambda x: x['score'], reverse=True)
    
    logger.info("=" * 60)
    logger.info("Top 20 Weight Configurations")
    logger.info("=" * 60)
    
    for j, cfg in enumerate(best_configs[:20]):
        c = cfg['config']
        logger.info(f"\nRank #{j+1} (Score: {cfg['score']:.4f}, Avg IC: {cfg['total_ic']:.4f}, Avg IR: {cfg['total_ir']:.4f})")
        logger.info(f"  HIGH_VOL:   ret_5d={c['high']['ret_5d']:.2f}, vol_20d={c['high']['vol_20d']:.2f}, mom={c['high']['vol_adj_momentum']:.2f}")
        logger.info(f"  LOW_VOL:    ret_5d={c['low']['ret_5d']:.2f}, vol_20d={c['low']['vol_20d']:.2f}, mom={c['low']['vol_adj_momentum']:.2f}")
        logger.info(f"  NORMAL:     ret_5d={c['normal']['ret_5d']:.2f}, vol_20d={c['normal']['vol_20d']:.2f}, mom={c['normal']['vol_adj_momentum']:.2f}")
    
    # Save best config
    if best_configs:
        best = best_configs[0]['config']
        logger.info("\n" + "=" * 60)
        logger.info("BEST CONFIGURATION (to be used in lightgbm_model.py)")
        logger.info("=" * 60)
        logger.info(f"HIGH_VOL_WEIGHTS = {{'ret_5d': {best['high']['ret_5d']:.2f}, 'vol_20d': {best['high']['vol_20d']:.2f}, 'vol_adj_momentum': {best['high']['vol_adj_momentum']:.2f}}}")
        logger.info(f"LOW_VOL_WEIGHTS = {{'ret_5d': {best['low']['ret_5d']:.2f}, 'vol_20d': {best['low']['vol_20d']:.2f}, 'vol_adj_momentum': {best['low']['vol_adj_momentum']:.2f}}}")
        logger.info(f"NORMAL_WEIGHTS = {{'ret_5d': {best['normal']['ret_5d']:.2f}, 'vol_20d': {best['normal']['vol_20d']:.2f}, 'vol_adj_momentum': {best['normal']['vol_adj_momentum']:.2f}}}")
        
        # Save to JSON
        result = {
            'best_config': {
                'high_vol': best['high'],
                'low_vol': best['low'],
                'normal': best['normal'],
            },
            'top_20': []
        }
        for cfg in best_configs[:20]:
            c = cfg['config']
            result['top_20'].append({
                'score': cfg['score'],
                'avg_ic': cfg['total_ic'],
                'avg_ir': cfg['total_ir'],
                'passed_years': cfg['passed_years'],
                'high': c['high'],
                'low': c['low'],
                'normal': c['normal'],
            })
        
        os.makedirs('reports', exist_ok=True)
        with open('reports/grid_search_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nResults saved to reports/grid_search_results.json")


if __name__ == '__main__':
    main()