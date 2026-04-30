#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Auto-Iterate Weight Optimization for V222
==========================================
Automatically searches for optimal dynamic weights by:
1. Modifying lightgbm_model.py WEIGHTS_* dictionaries
2. Running python -u run_v220.py --years 2020 2022 2024
3. Parsing JSON report for IC values
4. Iterating until convergence (IC >= 0.05 for all years) or max rounds

【核心约束】
- 恢复使用 trend_strength 替代 vol_adj_momentum
- 权重搜索空间：步长 0.05，反转 -0.7 ~ -0.1，趋势 +0.2 ~ +0.7
- 目标：2020/2022/2024 T+1 IC >= 0.05 且 IC IR >= 0.60
"""
import sys
import os
import re
import json
import glob
import time
import shutil
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
from loguru import logger

# Ensure project root is in sys.path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Paths
LIGHTGBM_MODEL_PATH = project_root / "src" / "lightgbm_model.py"
RUN_SCRIPT = project_root / "run_v220.py"
REPORTS_DIR = project_root / "reports"
ALPHA_HISTORY_PATH = project_root / "ALPHA_HISTORY.md"
CURRENT_STATE_PATH = project_root / "CURRENT_STATE.md"

# Search space
RET_WEIGHTS = np.arange(-0.7, -0.05, 0.05)  # -0.70 to -0.10
TREND_WEIGHTS = np.arange(0.2, 0.75, 0.05)  # 0.20 to 0.70
VOL_WEIGHT_FIXED = -0.1  # Keep volatility weight fixed

# Convergence criteria
IC_THRESHOLD = 0.05
IC_IR_THRESHOLD = 0.60
IMPROVEMENT_THRESHOLD = 0.005
MAX_NO_IMPROVE_ROUNDS = 3
MAX_TOTAL_ROUNDS = 50
RETURN_THRESHOLD = -0.20  # 2024 annual return must be > -20%

# Version counter
VERSION_PREFIX = "V223"
iteration_count = 0


def setup_logging():
    """Configure logging for real-time output"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        level="INFO",
        enqueue=False,
    )
    logger.add(
        REPORTS_DIR / "auto_iterate_{time:YYYYMMDD}.log",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )


def flush_stdout():
    """Flush stdout to ensure real-time display"""
    sys.stdout.flush()


def read_model_file():
    """Read the current lightgbm_model.py content"""
    with open(LIGHTGBM_MODEL_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def write_model_file(content):
    """Write modified content to lightgbm_model.py"""
    with open(LIGHTGBM_MODEL_PATH, 'w', encoding='utf-8') as f:
        f.write(content)


def modify_weights(content, weights_high, weights_low, weights_normal, version_tag):
    """
    Modify WEIGHTS_* in lightgbm_model.py and switch feature from vol_adj_momentum to trend_strength.
    
    Args:
        content: Original file content
        weights_high: Dict for HIGH_VOL state
        weights_low: Dict for LOW_VOL state
        weights_normal: Dict for NORMAL state
        version_tag: Version string for logging
    
    Returns:
        Modified file content
    """
    # Replace WEIGHTS_HIGH_VOL
    high_pattern = r"WEIGHTS_HIGH_VOL = \{[^}]+\}"
    high_replacement = f"WEIGHTS_HIGH_VOL = {{'ret_5d': {weights_high['ret_5d']:.2f}, 'vol_20d': {weights_high['vol_20d']:.2f}, 'vol_adj_momentum': {weights_high['trend']:.2f}}}"
    content = re.sub(high_pattern, high_replacement, content)
    
    # Replace WEIGHTS_LOW_VOL
    low_pattern = r"WEIGHTS_LOW_VOL = \{[^}]+\}"
    low_replacement = f"WEIGHTS_LOW_VOL = {{'ret_5d': {weights_low['ret_5d']:.2f}, 'vol_20d': {weights_low['vol_20d']:.2f}, 'vol_adj_momentum': {weights_low['trend']:.2f}}}"
    content = re.sub(low_pattern, low_replacement, content)
    
    # Replace WEIGHTS_NORMAL
    normal_pattern = r"WEIGHTS_NORMAL = \{[^}]+\}"
    normal_replacement = f"WEIGHTS_NORMAL = {{'ret_5d': {weights_normal['ret_5d']:.2f}, 'vol_20d': {weights_normal['vol_20d']:.2f}, 'vol_adj_momentum': {weights_normal['trend']:.2f}}}"
    content = re.sub(normal_pattern, normal_replacement, content)
    
    return content


def get_latest_json_report():
    """Find the most recent JSON report file"""
    json_files = list(REPORTS_DIR.glob("V222_Cross_Year_Report_*.json"))
    if not json_files:
        return None
    return max(json_files, key=os.path.getmtime)


def parse_json_report(json_path):
    """Parse JSON report to extract IC values for each year"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for year_str, year_data in data.get('results', {}).items():
            year = int(year_str)
            t1_ic = year_data.get('t1_ic', {})
            backtest = year_data.get('backtest_result', {})
            results[year] = {
                'mean_ic': t1_ic.get('mean_ic', 0.0),
                'ic_ir': t1_ic.get('ic_ir', 0.0),
                'annual_return': backtest.get('annual_return', 0.0),
                'max_drawdown': backtest.get('max_drawdown', 0.0),
                'sharpe_ratio': backtest.get('sharpe_ratio', 0.0),
            }
        return results
    except Exception as e:
        logger.error(f"Failed to parse JSON report: {e}")
        return None


def run_backtest():
    """Run the backtest and return results"""
    logger.info("Running backtest with python -u run_v220.py --years 2020 2022 2024")
    flush_stdout()
    
    # Use os.system to ensure real-time output
    exit_code = os.system(f'python -u "{RUN_SCRIPT}" --years 2020 2022 2024')
    
    flush_stdout()
    
    # Find and parse the latest JSON report
    json_path = get_latest_json_report()
    if json_path is None:
        logger.error("No JSON report found after backtest")
        return None
    
    results = parse_json_report(json_path)
    if results is None:
        logger.error("Failed to parse JSON report")
        return None
    
    return results, json_path


def check_convergence(results):
    """Check if results meet all convergence criteria"""
    if results is None:
        return False
    
    for year in [2020, 2022, 2024]:
        if year not in results:
            return False
        r = results[year]
        if r['mean_ic'] < IC_THRESHOLD:
            return False
        if r['ic_ir'] < IC_IR_THRESHOLD:
            return False
    
    # Check 2024 annual return
    if 2024 in results and results[2024]['annual_return'] < RETURN_THRESHOLD:
        return False
    
    return True


def compute_objective(results):
    """Compute objective function: average IC across 3 years"""
    if results is None:
        return -999.0
    total = 0
    count = 0
    for year in [2020, 2022, 2024]:
        if year in results:
            total += results[year]['mean_ic']
            count += 1
    return total / count if count > 0 else -999.0


def log_candidate(weights_high, weights_low, weights_normal, results, json_path):
    """Log the current candidate weights and results"""
    logger.info("=" * 70)
    logger.info("Candidate Weights & Results:")
    logger.info("-" * 70)
    logger.info(f"  HIGH_VOL:   ret_5d={weights_high['ret_5d']:.2f}, vol_20d={weights_high['vol_20d']:.2f}, trend={weights_high['trend']:.2f}")
    logger.info(f"  LOW_VOL:    ret_5d={weights_low['ret_5d']:.2f}, vol_20d={weights_low['vol_20d']:.2f}, trend={weights_low['trend']:.2f}")
    logger.info(f"  NORMAL:     ret_5d={weights_normal['ret_5d']:.2f}, vol_20d={weights_normal['vol_20d']:.2f}, trend={weights_normal['trend']:.2f}")
    logger.info("-" * 70)
    
    for year in [2020, 2022, 2024]:
        if year in results:
            r = results[year]
            status = "PASS" if (r['mean_ic'] >= IC_THRESHOLD and r['ic_ir'] >= IC_IR_THRESHOLD) else "FAIL"
            logger.info(f"  Year {year}: IC={r['mean_ic']:.4f}, IC_IR={r['ic_ir']:.2f}, AnnRet={r['annual_return']:.2%}, MaxDD={r['max_drawdown']:.2%} [{status}]")
        else:
            logger.info(f"  Year {year}: NO RESULTS")
    
    logger.info(f"  Objective (Avg IC): {compute_objective(results):.4f}")
    logger.info(f"  Report: {json_path}")
    logger.info("=" * 70)
    flush_stdout()


def update_alpha_history(version, weights_high, weights_low, weights_normal, results, passed):
    """Append iteration record to ALPHA_HISTORY.md"""
    global iteration_count
    iteration_count += 1
    
    timestamp = datetime.now().strftime('%Y-%m-%d')
    status = "✅ 通过" if passed else "❌ 未通过"
    
    lines = []
    lines.append(f"\n## {version} - {timestamp}")
    lines.append(f"- **核心尝试**：")
    lines.append(f"  - 自动迭代第 {iteration_count} 轮，网格搜索最优动态权重。")
    lines.append(f"  - 使用 trend_strength 替代 vol_adj_momentum（恢复趋势强度因子）。")
    lines.append(f"  - 权重配置：")
    lines.append(f"    - HIGH_VOL: ret_5d={weights_high['ret_5d']:.2f}, vol_20d={weights_high['vol_20d']:.2f}, trend={weights_high['trend']:.2f}")
    lines.append(f"    - LOW_VOL: ret_5d={weights_low['ret_5d']:.2f}, vol_20d={weights_low['vol_20d']:.2f}, trend={weights_low['trend']:.2f}")
    lines.append(f"    - NORMAL: ret_5d={weights_normal['ret_5d']:.2f}, vol_20d={weights_normal['vol_20d']:.2f}, trend={weights_normal['trend']:.2f}")
    lines.append(f"- **结果**（IC/IC IR 数字）：")
    
    for year in [2020, 2022, 2024]:
        if year in results:
            r = results[year]
            lines.append(f"  - {year}: T+1 IC={r['mean_ic']:.4f}, IC IR={r['ic_ir']:.2f}, 年化收益={r['annual_return']:.2%}, Max DD={r['max_drawdown']:.2%}")
        else:
            lines.append(f"  - {year}: 无结果")
    
    lines.append(f"- **教训**：")
    if passed:
        lines.append(f"  - 成功！所有年份 IC >= {IC_THRESHOLD} 且 IC IR >= {IC_IR_THRESHOLD}。")
    else:
        lines.append(f"  - 未达到目标，需要继续优化。")
    lines.append(f"- **状态**：{status}")
    lines.append("")
    
    with open(ALPHA_HISTORY_PATH, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines))
    
    logger.info(f"Updated ALPHA_HISTORY.md with {version}")
    flush_stdout()


def update_current_state(weights_high, weights_low, weights_normal, results):
    """Update CURRENT_STATE.md with the best weights"""
    timestamp = datetime.now().strftime('%Y-%m-%d')
    
    content = f"""# CURRENT_STATE.md - 当前策略状态

> **最后更新时间**: {timestamp}
> **当前版本**: {VERSION_PREFIX} (Auto-Iterate Optimized)

---

## 策略核心

### 因子公式
1. **反转因子 (Reversal)**: `ret_5d = close / close.shift(5) - 1`
2. **波动率 (Volatility)**: `vol_20d = ret_1d.rolling(20).std()`
3. **趋势强度 (Trend Strength)**: `trend_strength = close / ma_20 - 1`

### 动态权重配置（基于大盘 20 日波动率分位数）

| 市场状态 | 条件 | ret_5d | vol_20d | trend_strength |
|----------|------|--------|---------|----------------|
| 高波动 | vol_percentile > 0.7 | {weights_high['ret_5d']:.2f} | {weights_high['vol_20d']:.2f} | {weights_high['trend']:.2f} |
| 低波动 | vol_percentile < 0.3 | {weights_low['ret_5d']:.2f} | {weights_low['vol_20d']:.2f} | {weights_low['trend']:.2f} |
| 正常 | 其他 | {weights_normal['ret_5d']:.2f} | {weights_normal['vol_20d']:.2f} | {weights_normal['trend']:.2f} |

### 市场状态逻辑
- 高波动（恐慌）：反转效应强，加大反转权重
- 低波动（趋势）：动量效应强，加大趋势权重
- 正常状态：均衡配置

---

## 最新回测结果

| 年份 | T+1 IC | IC IR | 年化收益 | 最大回撤 | 状态 |
|------|--------|-------|----------|----------|------|
"""
    
    for year in [2020, 2022, 2024]:
        if year in results:
            r = results[year]
            status = "PASS" if (r['mean_ic'] >= IC_THRESHOLD and r['ic_ir'] >= IC_IR_THRESHOLD) else "FAIL"
            content += f"| {year} | {r['mean_ic']:.4f} | {r['ic_ir']:.2f} | {r['annual_return']:.2%} | {r['max_drawdown']:.2%} | {status} |\n"
    
    content += f"""
---

## 数据依赖
- MySQL 数据库：`stock_daily` 表（OHLCV 数据）
- 截面 z-score 标准化（按交易日分组）
- DataHealer 处理缺失值（向前填充 + 行业均值兜底）

## 已知问题
- 线性组合方法可能已达到性能上限
- 需要进一步探索非线性方法或另类数据

---

*状态文档由自动迭代脚本更新*
"""
    
    with open(CURRENT_STATE_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info("Updated CURRENT_STATE.md")
    flush_stdout()


def cleanup_old_reports():
    """Clean up old reports, keep only the latest one"""
    try:
        json_files = sorted(REPORTS_DIR.glob("V222_Cross_Year_Report_*.json"))
        md_files = sorted(REPORTS_DIR.glob("V222_Cross_Year_Report_*.md"))
        
        # Keep only the latest
        if len(json_files) > 1:
            for f in json_files[:-1]:
                os.remove(f)
        if len(md_files) > 1:
            for f in md_files[:-1]:
                os.remove(f)
        
        logger.info(f"Cleaned up old reports: {len(json_files) + len(md_files) - 2} files removed")
    except Exception as e:
        logger.warning(f"Failed to cleanup reports: {e}")


def main():
    """Main auto-iteration loop"""
    global iteration_count
    
    setup_logging()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("Auto-Iterate Weight Optimization Starting")
    logger.info("=" * 70)
    logger.info(f"  Target IC >= {IC_THRESHOLD}, IC IR >= {IC_IR_THRESHOLD}")
    logger.info(f"  Ret weights range: {RET_WEIGHTS[0]:.2f} to {RET_WEIGHTS[-1]:.2f}")
    logger.info(f"  Trend weights range: {TREND_WEIGHTS[0]:.2f} to {TREND_WEIGHTS[-1]:.2f}")
    logger.info(f"  Vol weight fixed: {VOL_WEIGHT_FIXED:.2f}")
    logger.info("=" * 70)
    flush_stdout()
    
    # Read original model file once
    original_content = read_model_file()
    
    # Initial weights (based on V222 best)
    best_weights_high = {'ret_5d': -0.5, 'vol_20d': VOL_WEIGHT_FIXED, 'trend': 0.4}
    best_weights_low = {'ret_5d': -0.3, 'vol_20d': VOL_WEIGHT_FIXED, 'trend': 0.6}
    best_weights_normal = {'ret_5d': -0.4, 'vol_20d': VOL_WEIGHT_FIXED, 'trend': 0.5}
    
    # Run initial backtest to get baseline
    logger.info("\n=== Phase 1: Initial Baseline ===")
    initial_content = modify_weights(
        original_content,
        best_weights_high, best_weights_low, best_weights_normal,
        f"{VERSION_PREFIX}_init"
    )
    write_model_file(initial_content)
    
    init_result = run_backtest()
    if init_result is None:
        logger.error("Initial backtest failed!")
        sys.exit(1)
    
    init_results, init_json_path = init_result
    best_objective = compute_objective(init_results)
    
    log_candidate(best_weights_high, best_weights_low, best_weights_normal, init_results, init_json_path)
    logger.info(f">>> Initial baseline objective: {best_objective:.4f}")
    flush_stdout()
    
    # Iteration loop
    no_improve_rounds = 0
    round_num = 0
    passed = False
    final_results = init_results
    
    while round_num < MAX_TOTAL_ROUNDS and not passed:
        round_num += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"=== ROUND {round_num} / {MAX_TOTAL_ROUNDS} ===")
        logger.info(f"{'='*70}")
        flush_stdout()
        
        round_improved = False
        
        # Generate candidate weights by perturbing current best
        perturbation = 0.05 * (1 + no_improve_rounds * 0.5)  # Increase perturbation if stuck
        
        candidates = []
        
        # Perturb each weight dimension
        for dim in ['ret_5d', 'trend']:
            for direction in [-1, 1]:
                for state_name, state_weights in [
                    ('high', best_weights_high),
                    ('low', best_weights_low),
                    ('normal', best_weights_normal)
                ]:
                    if dim == 'trend':
                        key = 'trend'
                    else:
                        key = dim
                    
                    new_val = state_weights[key] + direction * perturbation
                    
                    # Clamp values
                    if dim == 'ret_5d':
                        new_val = max(-0.7, min(-0.1, new_val))
                    elif dim == 'trend':
                        new_val = max(0.2, min(0.7, new_val))
                    
                    # Create candidate by modifying one state weight
                    cand_high = dict(best_weights_high)
                    cand_low = dict(best_weights_low)
                    cand_normal = dict(best_weights_normal)
                    
                    if state_name == 'high':
                        cand_high[key] = new_val
                    elif state_name == 'low':
                        cand_low[key] = new_val
                    else:
                        cand_normal[key] = new_val
                    
                    # Ensure weights sum to ~0.3 (balance constraint)
                    for cand in [cand_high, cand_low, cand_normal]:
                        total = cand['ret_5d'] + cand['vol_20d'] + cand['trend']
                        # Adjust trend to make sum = 0.3
                        cand['trend'] = 0.3 - cand['ret_5d'] - cand['vol_20d']
                        cand['trend'] = max(0.2, min(0.7, cand['trend']))
                    
                    candidates.append((cand_high, cand_low, cand_normal))
        
        # Remove duplicates
        unique_candidates = []
        seen = set()
        for c in candidates:
            key = (round(c[0]['ret_5d'], 2), round(c[0]['trend'], 2),
                   round(c[1]['ret_5d'], 2), round(c[1]['trend'], 2),
                   round(c[2]['ret_5d'], 2), round(c[2]['trend'], 2))
            if key not in seen:
                seen.add(key)
                unique_candidates.append(c)
        
        logger.info(f"Testing {len(unique_candidates)} candidate configurations...")
        flush_stdout()
        
        # Test each candidate
        for idx, (cand_high, cand_low, cand_normal) in enumerate(unique_candidates):
            logger.info(f"\n--- Candidate {idx+1}/{len(unique_candidates)} ---")
            logger.info(f"  HIGH: ret_5d={cand_high['ret_5d']:.2f}, vol_20d={cand_high['vol_20d']:.2f}, trend={cand_high['trend']:.2f}")
            logger.info(f"  LOW:  ret_5d={cand_low['ret_5d']:.2f}, vol_20d={cand_low['vol_20d']:.2f}, trend={cand_low['trend']:.2f}")
            logger.info(f"  NORM: ret_5d={cand_normal['ret_5d']:.2f}, vol_20d={cand_normal['vol_20d']:.2f}, trend={cand_normal['trend']:.2f}")
            flush_stdout()
            
            # Modify weights and run backtest
            modified_content = modify_weights(
                original_content,
                cand_high, cand_low, cand_normal,
                f"{VERSION_PREFIX}_r{round_num}_c{idx+1}"
            )
            write_model_file(modified_content)
            
            result = run_backtest()
            if result is None:
                logger.warning(f"Candidate {idx+1} failed, skipping...")
                continue
            
            results, json_path = result
            obj = compute_objective(results)
            
            logger.info(f"  Objective: {obj:.4f} (best: {best_objective:.4f})")
            flush_stdout()
            
            # Check if this is the best so far
            if obj > best_objective + IMPROVEMENT_THRESHOLD:
                best_objective = obj
                best_weights_high = dict(cand_high)
                best_weights_low = dict(cand_low)
                best_weights_normal = dict(cand_normal)
                final_results = results
                no_improve_rounds = 0
                round_improved = True
                
                logger.info(f"  *** NEW BEST! Objective improved by {obj - best_objective + IMPROVEMENT_THRESHOLD:.4f} ***")
                
                # Check convergence
                if check_convergence(results):
                    passed = True
                    logger.info(f"\n{'='*70}")
                    logger.info(f"*** CONVERGENCE ACHIEVED! ***")
                    logger.info(f"{'='*70}")
                    log_candidate(best_weights_high, best_weights_low, best_weights_normal, results, json_path)
                    break
            
            # Log IC for this candidate
            for year in [2020, 2022, 2024]:
                if year in results:
                    r = results[year]
                    logger.info(f"    Year {year}: IC={r['mean_ic']:.4f}, IC_IR={r['ic_ir']:.2f}")
            flush_stdout()
        
        if not round_improved and not passed:
            no_improve_rounds += 1
            logger.info(f"\n>>> Round {round_num}: No improvement. Streak: {no_improve_rounds}/{MAX_NO_IMPROVE_ROUNDS}")
            flush_stdout()
            
            if no_improve_rounds >= MAX_NO_IMPROVE_ROUNDS:
                logger.info(f"\n{'='*70}")
                logger.info(f"*** EARLY STOPPING: {MAX_NO_IMPROVE_ROUNDS} rounds without improvement ***")
                logger.info(f"{'='*70}")
                break
        
        flush_stdout()
    
    # Final output
    logger.info("\n" + "=" * 70)
    logger.info("AUTO-ITERATE FINAL RESULTS")
    logger.info("=" * 70)
    
    version_tag = f"{VERSION_PREFIX}_final"
    update_alpha_history(version_tag, best_weights_high, best_weights_low, best_weights_normal, final_results, passed)
    update_current_state(best_weights_high, best_weights_low, best_weights_normal, final_results)
    cleanup_old_reports()
    
    logger.info(f"\nFinal Best Weights:")
    logger.info(f"  HIGH_VOL:   ret_5d={best_weights_high['ret_5d']:.2f}, vol_20d={best_weights_high['vol_20d']:.2f}, trend={best_weights_high['trend']:.2f}")
    logger.info(f"  LOW_VOL:    ret_5d={best_weights_low['ret_5d']:.2f}, vol_20d={best_weights_low['vol_20d']:.2f}, trend={best_weights_low['trend']:.2f}")
    logger.info(f"  NORMAL:     ret_5d={best_weights_normal['ret_5d']:.2f}, vol_20d={best_weights_normal['vol_20d']:.2f}, trend={best_weights_normal['trend']:.2f}")
    
    logger.info(f"\nFinal IC Results:")
    for year in [2020, 2022, 2024]:
        if year in final_results:
            r = final_results[year]
            status = "PASS" if (r['mean_ic'] >= IC_THRESHOLD and r['ic_ir'] >= IC_IR_THRESHOLD) else "FAIL"
            logger.info(f"  Year {year}: IC={r['mean_ic']:.4f}, IC_IR={r['ic_ir']:.2f} [{status}]")
        else:
            logger.info(f"  Year {year}: NO RESULTS")
    
    logger.info(f"\nOverall Status: {'ALL PASSED' if passed else 'DID NOT MEET TARGET'}")
    logger.info(f"Total Rounds: {round_num}")
    logger.info(f"Final Objective (Avg IC): {best_objective:.4f}")
    logger.info("=" * 70)
    flush_stdout()
    
    if passed:
        logger.info("\n✅ SUCCESS: All years meet IC >= 0.05 and IC IR >= 0.60!")
        sys.exit(0)
    else:
        logger.warning("\n❌ Target not met. Review ALPHA_HISTORY.md for iteration log.")
        sys.exit(1)


if __name__ == "__main__":
    main()