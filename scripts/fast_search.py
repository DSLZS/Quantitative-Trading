#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast Weight Search for V223 - Using trend_strength instead of vol_adj_momentum
===============================================================================
Searches for optimal weights using:
- Features: ret_5d (reversal), vol_20d (volatility), trend_strength
- Only tests ~20 key configurations (not 175)
- Focuses on the most promising weight ranges
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
    """Compute features using trend_strength (not vol_adj_momentum)"""
    df = df.sort_values(['symbol', 'trade_date'])
    
    results = []
    for symbol, group in df.groupby('symbol'):
        group = group.copy()
        group['ret_5d'] = group['close'] / group['close'].shift(5) - 1
        group['vol_20d'] = group['close'].pct_change().rolling(20).std()
        # trend_strength = close/ma20 - 1
        group['ma_20'] = group['close'].rolling(20, min_periods=10).mean()
        group['trend_strength'] = (group['close'] - group['ma_20']) / group['ma_20']
        group['next_1d_ret'] = group['close'].shift(-1) / group['close'] - 1
        results.append(group)
    
    df = pd.concat(results, ignore_index=True)
    return df.dropna(subset=['ret_5d', 'vol_20d', 'trend_strength', 'next_1d_ret'])


def cross_sectional_zscore(df, col):
    """Cross-sectional z-score per date"""
    def zscore_group(g):
        mean = g.mean()
        std = g.std()
        if std > 1e-10:
            return (g - mean) / std
        return g - mean
    return df.groupby('trade_date')[col].transform(zscore_group)


def compute_volatility_state(df):
    """Classify market state based on volatility percentile"""
    daily_avg_vol = df.groupby('trade_date')['vol_20d'].mean().reset_index()
    daily_avg_vol.columns = ['trade_date', 'avg_vol']
    daily_avg_vol = daily_avg_vol.sort_values('trade_date').reset_index(drop=True)
    
    rolling_q70 = daily_avg_vol['avg_vol'].rolling(60, min_periods=20).quantile(0.7)
    rolling_q30 = daily_avg_vol['avg_vol'].rolling(60, min_periods=20).quantile(0.3)
    
    daily_avg_vol['vol_q70'] = rolling_q70
    daily_avg_vol['vol_q30'] = rolling_q30
    
    df = df.merge(daily_avg_vol, on='trade_date', how='left')
    df['market_state'] = 'NORMAL'
    df.loc[df['avg_vol'] > df['vol_q70'], 'market_state'] = 'HIGH_VOL'
    df.loc[df['avg_vol'] < df['vol_q30'], 'market_state'] = 'LOW_VOL'
    
    return df


def apply_weights_and_compute_ic(df, weights_high, weights_low, weights_normal):
    """Apply weights and compute IC per date"""
    df = df.copy()
    score = np.zeros(len(df))
    
    high_mask = df['market_state'] == 'HIGH_VOL'
    low_mask = df['market_state'] == 'LOW_VOL'
    normal_mask = ~high_mask & ~low_mask
    
    for mask, w in [(high_mask, weights_high), (low_mask, weights_low), (normal_mask, weights_normal)]:
        if mask.any():
            score[mask] = (
                w['ret_5d'] * df.loc[mask, 'z_ret_5d'].values +
                w['vol_20d'] * df.loc[mask, 'z_vol_20d'].values +
                w['trend'] * df.loc[mask, 'z_trend'].values
            )
    
    df['score'] = score
    
    ics = []
    for date, group in df.groupby('trade_date'):
        if len(group) < 50:
            continue
        score_arr = group['score'].values
        ret_arr = group['next_1d_ret'].values
        
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
    logger.info("=" * 70)
    logger.info("V223 Fast Weight Search (using trend_strength)")
    logger.info("=" * 70)
    
    # Load data - only need OHLC
    db_url = os.getenv('DATABASE_URL')
    engine = create_engine(db_url)
    
    query = """
        SELECT symbol, trade_date, close, high, low
        FROM stock_daily 
        WHERE trade_date >= 20190101 AND trade_date <= 20241231
        ORDER BY symbol, trade_date
    """
    
    logger.info("Loading data from MySQL...")
    df = pd.read_sql(query, engine)
    engine.dispose()
    logger.info(f"Loaded {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # Compute features
    logger.info("Computing features (ret_5d, vol_20d, trend_strength)...")
    df = compute_features(df)
    logger.info(f"After feature computation: {len(df)} rows")
    
    # Cross-sectional z-score
    logger.info("Computing cross-sectional z-scores...")
    df['z_ret_5d'] = cross_sectional_zscore(df, 'ret_5d')
    df['z_vol_20d'] = cross_sectional_zscore(df, 'vol_20d')
    df['z_trend'] = cross_sectional_zscore(df, 'trend_strength')
    
    # Market state
    logger.info("Computing market states...")
    df = compute_volatility_state(df)
    logger.info(f"  HIGH_VOL: {(df['market_state'] == 'HIGH_VOL').sum()}")
    logger.info(f"  LOW_VOL: {(df['market_state'] == 'LOW_VOL').sum()}")
    logger.info(f"  NORMAL: {(df['market_state'] == 'NORMAL').sum()}")
    
    # Separate by year
    if isinstance(df['trade_date'].iloc[0], (pd.Timestamp,)):
        df['year'] = df['trade_date'].dt.year
    elif isinstance(df['trade_date'].iloc[0], int):
        df['year'] = df['trade_date'] // 10000
    else:
        df['year'] = df['trade_date'].apply(lambda x: x.year)
    
    years_data = {}
    for year in [2020, 2022, 2024]:
        mask = df['year'] == year
        years_data[year] = df[mask].copy()
    
    logger.info(f"Data: 2020={len(years_data[2020])}, 2022={len(years_data[2022])}, 2024={len(years_data[2024])}")
    
    # Key configurations to test (only ~20)
    # Strategy: vary ret_5d and trend weights, keep vol fixed at -0.1
    # Based on V222 analysis: trend_strength may be more effective than vol_adj_momentum
    
    configs = [
        # Base V222 weights but with trend_strength
        {'name': 'V222_base_trend', 
         'high': {'ret_5d': -0.5, 'vol_20d': -0.1, 'trend': 0.4},
         'low': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.6},
         'normal': {'ret_5d': -0.4, 'vol_20d': -0.1, 'trend': 0.5}},
        
        # Stronger reversal for all states
        {'name': 'strong_reversal',
         'high': {'ret_5d': -0.6, 'vol_20d': -0.1, 'trend': 0.3},
         'low': {'ret_5d': -0.5, 'vol_20d': -0.1, 'trend': 0.4},
         'normal': {'ret_5d': -0.55, 'vol_20d': -0.1, 'trend': 0.35}},
        
        # Stronger trend for all states
        {'name': 'strong_trend',
         'high': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.6},
         'low': {'ret_5d': -0.2, 'vol_20d': -0.1, 'trend': 0.7},
         'normal': {'ret_5d': -0.25, 'vol_20d': -0.1, 'trend': 0.65}},
        
        # Balanced
        {'name': 'balanced',
         'high': {'ret_5d': -0.4, 'vol_20d': -0.1, 'trend': 0.5},
         'low': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.5},
         'normal': {'ret_5d': -0.35, 'vol_20d': -0.1, 'trend': 0.5}},
        
        # High vol: strong reversal, Low vol: strong trend (extreme)
        {'name': 'extreme_gate',
         'high': {'ret_5d': -0.7, 'vol_20d': -0.1, 'trend': 0.2},
         'low': {'ret_5d': -0.1, 'vol_20d': -0.1, 'trend': 0.7},
         'normal': {'ret_5d': -0.4, 'vol_20d': -0.1, 'trend': 0.5}},
        
        # Moderate gate
        {'name': 'moderate_gate',
         'high': {'ret_5d': -0.55, 'vol_20d': -0.1, 'trend': 0.35},
         'low': {'ret_5d': -0.25, 'vol_20d': -0.1, 'trend': 0.65},
         'normal': {'ret_5d': -0.4, 'vol_20d': -0.1, 'trend': 0.5}},
        
        # Same weights for all states (no gating)
        {'name': 'no_gate_reversal',
         'high': {'ret_5d': -0.5, 'vol_20d': -0.1, 'trend': 0.4},
         'low': {'ret_5d': -0.5, 'vol_20d': -0.1, 'trend': 0.4},
         'normal': {'ret_5d': -0.5, 'vol_20d': -0.1, 'trend': 0.4}},
        
        {'name': 'no_gate_trend',
         'high': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.6},
         'low': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.6},
         'normal': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.6}},
        
        # Negative vol weight (penalize high vol stocks)
        {'name': 'neg_vol',
         'high': {'ret_5d': -0.5, 'vol_20d': -0.2, 'trend': 0.3},
         'low': {'ret_5d': -0.3, 'vol_20d': -0.2, 'trend': 0.5},
         'normal': {'ret_5d': -0.4, 'vol_20d': -0.2, 'trend': 0.4}},
        
        # Very strong trend in low vol
        {'name': 'trend_low_vol',
         'high': {'ret_5d': -0.4, 'vol_20d': -0.1, 'trend': 0.4},
         'low': {'ret_5d': -0.15, 'vol_20d': -0.1, 'trend': 0.75},
         'normal': {'ret_5d': -0.3, 'vol_20d': -0.1, 'trend': 0.55}},
    ]
    
    logger.info(f"Testing {len(configs)} configurations...")
    
    results = []
    for i, cfg in enumerate(configs):
        if i % 5 == 0:
            logger.info(f"Progress: {i}/{len(configs)}")
        
        total_ic = 0
        total_ir = 0
        year_ics = {}
        
        for year, data in years_data.items():
            if len(data) == 0:
                continue
            mean_ic, ic_ir = apply_weights_and_compute_ic(
                data, cfg['high'], cfg['low'], cfg['normal']
            )
            year_ics[year] = {'ic': mean_ic, 'ir': ic_ir}
            total_ic += mean_ic
            total_ir += ic_ir
        
        avg_ic = total_ic / 3
        avg_ir = total_ir / 3
        
        # Check pass criteria
        passed = all(year_ics[y]['ic'] >= 0.05 and year_ics[y]['ir'] >= 0.6 for y in [2020, 2022, 2024])
        
        results.append({
            'name': cfg['name'],
            'avg_ic': avg_ic,
            'avg_ir': avg_ir,
            'year_ics': year_ics,
            'passed': passed,
            'config': {'high': cfg['high'], 'low': cfg['low'], 'normal': cfg['normal']}
        })
        
        logger.info(f"  [{cfg['name']}] Avg IC={avg_ic:.4f}, IR={avg_ir:.2f}")
        for y in [2020, 2022, 2024]:
            yi = year_ics[y]
            logger.info(f"    {y}: IC={yi['ic']:.4f}, IR={yi['ir']:.2f}")
    
    # Sort by average IC
    results.sort(key=lambda x: x['avg_ic'], reverse=True)
    
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS (sorted by average IC)")
    logger.info("=" * 70)
    
    for j, r in enumerate(results):
        status = "PASS" if r['passed'] else "FAIL"
        logger.info(f"Rank #{j+1}: {r['name']} (Avg IC={r['avg_ic']:.4f}, IR={r['avg_ir']:.2f}) [{status}]")
        for y in [2020, 2022, 2024]:
            yi = r['year_ics'][y]
            logger.info(f"  {y}: IC={yi['ic']:.4f}, IR={yi['ir']:.2f}")
    
    # Save best config
    best = results[0]
    logger.info("\n" + "=" * 70)
    logger.info("BEST CONFIGURATION:")
    logger.info("=" * 70)
    logger.info(f"Name: {best['name']}")
    for state in ['high', 'low', 'normal']:
        w = best['config'][state]
        logger.info(f"  {state}: ret_5d={w['ret_5d']:.2f}, vol_20d={w['vol_20d']:.2f}, trend={w['trend']:.2f}")
    
    # Save to JSON
    os.makedirs('reports', exist_ok=True)
    with open('reports/fast_search_results.json', 'w') as f:
        json.dump({
            'best': best['name'],
            'best_config': best['config'],
            'all_results': [{'name': r['name'], 'avg_ic': r['avg_ic'], 'avg_ir': r['avg_ir'], 
                            'year_ics': r['year_ics'], 'passed': r['passed']} for r in results]
        }, f, indent=2)
    logger.info(f"\nResults saved to reports/fast_search_results.json")


if __name__ == '__main__':
    main()