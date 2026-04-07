"""
V158 Fusion Audit Runner - V156 Signal-Smoothing + V157 IC-IR Optimized Weighting
"""

import sys
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}", level="INFO")
logger.add("logs/v158_audit_{time:YYYYMMDD_HHmmss}.log", rotation="10 MB", retention="7 days", level="DEBUG")

from src.alpha_research_v158 import AlphaResearchV158, get_alpha_research
from src.engine.backtest_referee import BacktestReferee

VERSION = "V158"


# 辅助函数用于安全获取嵌套字典的值
def get_gvs_value(stats: dict, key: str, default: float = 0.0) -> float:
    """安全获取 GVS 统计值"""
    if not stats:
        return default
    val = stats.get(key)
    return float(val) if val is not None else default


def get_atg_value(stats: dict, key: str, default: float = 0.0) -> float:
    """安全获取 ATG 统计值"""
    if not stats:
        return default
    val = stats.get(key)
    return float(val) if val is not None else default


def get_icir_value(stats: dict, key: str, default: float = 0.0) -> float:
    """安全获取 ICIR 统计值"""
    if not stats:
        return default
    val = stats.get(key)
    return float(val) if val is not None else default


class V158BacktestReferee(BacktestReferee):
    """V158 回测裁判 - 继承 V103，保持初始资金和费率不变"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.0003,
                 stamp_duty: float = 0.001, slippage: float = 0.0005, top_n: int = 50):
        super().__init__(initial_capital=initial_capital, commission=commission,
                        stamp_duty=stamp_duty, slippage=slippage, top_n=top_n)
        logger.info(f"[{VERSION}] Backtest Referee Initialized (Initial Capital: {initial_capital})")


def load_data(year: int) -> pd.DataFrame:
    """从数据库加载数据"""
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    load_dotenv()
    
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL not found in .env")
    
    engine = create_engine(db_url)
    
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    
    # 动态字段映射：数据库中没有 pe_ttm、pb、amount 列
    query = text("""
        SELECT symbol, trade_date, open, high, low, close, pre_close, 
               volume, turnover_rate, total_mv,
               pct_chg, `change` AS price_change
        FROM stock_daily
        WHERE trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date, symbol
    """)
    
    logger.info(f"[{VERSION}][Data] Loading data for year {year}...")
    df = pd.read_sql_query(query, engine, params={'start_date': start_date, 'end_date': end_date})
    logger.info(f"[{VERSION}][Data] Loaded {len(df)} rows for year {year}")
    
    return df


def run_backtest(signals: pd.DataFrame, returns: pd.DataFrame, year: int) -> dict:
    """运行回测"""
    logger.info(f"[{VERSION}][Backtest] Running backtest for year {year}...")
    
    signals = signals.copy()
    returns = returns.copy()
    
    # 移除 t1_return 避免 merge 冲突
    if 't1_return' in signals.columns:
        signals = signals.drop(columns=['t1_return'])
    
    # 确保 returns 有 t1_return
    if 't1_return' not in returns.columns:
        if 't1_return_period' in returns.columns:
            returns['t1_return'] = returns['t1_return_period']
    
    returns_cols = ['symbol', 'trade_date', 't1_return']
    available_cols = [c for c in returns_cols if c in returns.columns]
    returns = returns[available_cols].copy()
    
    # 填充 NaN
    if 't1_return' in returns.columns:
        returns['t1_return'] = returns['t1_return'].fillna(0)
    
    # 合并
    merged = signals.merge(returns, on=['symbol', 'trade_date'], how='left')
    
    if 't1_return' in merged.columns:
        merged['t1_return'] = merged['t1_return'].fillna(0)
    
    logger.info(f"[{VERSION}][Backtest] Merged shape: {merged.shape}")
    
    # 按日期分组回测
    dates = sorted(merged['trade_date'].unique())
    
    portfolio_value = 100000.0
    daily_values = []
    daily_profits = []
    
    for date in dates:
        day_data = merged[merged['trade_date'] == date]
        
        if len(day_data) < 10:
            daily_values.append(portfolio_value)
            daily_profits.append(0)
            continue
        
        # 选择 Top N
        day_data = day_data.dropna(subset=['score'])
        if len(day_data) < 10:
            daily_values.append(portfolio_value)
            daily_profits.append(0)
            continue
        
        top_n = 50
        selected = day_data.nlargest(min(top_n, len(day_data)), 'score')
        
        if len(selected) == 0:
            daily_values.append(portfolio_value)
            daily_profits.append(0)
            continue
        
        # 等权配置
        weight = 1.0 / len(selected)
        
        # 计算当日收益
        day_profit = 0
        for _, row in selected.iterrows():
            if 't1_return' in row and not pd.isna(row['t1_return']):
                stock_return = row['t1_return']
            else:
                stock_return = 0
            
            position_value = portfolio_value * weight
            profit = position_value * stock_return
            
            # 费用
            commission = position_value * 0.0003
            stamp_duty = abs(profit) * 0.001 if profit < 0 else 0
            
            day_profit += profit - commission - stamp_duty
        
        portfolio_value += day_profit
        daily_values.append(portfolio_value)
        daily_profits.append(day_profit)
        
        if len(daily_profits) <= 30 or len(daily_profits) % 50 == 0:
            logger.info(f"[{VERSION}][Backtest] Date {date}: portfolio_value={portfolio_value:.2f}, daily_profit={day_profit:.2f}")
    
    # 计算指标
    if len(daily_values) < 2:
        return {'total_return': 0, 'annual_return': 0, 'sharpe': 0, 'max_drawdown': 0}
    
    daily_values = np.array(daily_values)
    daily_profits = np.array(daily_profits)
    
    initial = 100000.0
    final = daily_values[-1]
    total_return = (final - initial) / initial
    
    n_days = len(daily_values)
    annual_return = (1 + total_return) ** (252 / max(n_days, 1)) - 1
    
    daily_returns = daily_profits / np.maximum(np.roll(daily_values, 1), initial)
    daily_returns = daily_returns[1:]
    
    if np.std(daily_returns) > 1e-10:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0
    
    peak = np.maximum.accumulate(daily_values)
    drawdown = (daily_values - peak) / np.maximum(peak, 1e-10)
    max_drawdown = np.min(drawdown)
    
    logger.info(f"[{VERSION}][Backtest] Total Return: {total_return*100:.2f}%")
    logger.info(f"[{VERSION}][Backtest] Annual Return: {annual_return*100:.2f}%")
    logger.info(f"[{VERSION}][Backtest] Sharpe Ratio: {sharpe:.2f}")
    logger.info(f"[{VERSION}][Backtest] Max Drawdown: {max_drawdown*100:.2f}%")
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'final_value': final,
        'daily_values': daily_values.tolist(),
        'daily_profits': daily_profits.tolist(),
    }


def run_audit(year: int) -> dict:
    """运行 V158 审计"""
    logger.info(f"[{VERSION}] Running audit for year {year}")
    
    # 加载数据
    df = load_data(year)
    
    # 初始化 Alpha
    alpha = get_alpha_research()
    
    # 计算分数
    result = alpha.compute_score(df)
    
    logger.info(f"[{VERSION}] Score computed: {len(result)} rows")
    
    # 计算 IC
    ics = []
    for date in result['trade_date'].unique():
        day = result[result['trade_date'] == date]
        if len(day) < 20:
            continue
        f = day['score'].fillna(0)
        r = day['t1_return'].fillna(0)
        if len(f) > 10 and np.std(f) > 1e-10:
            ic = np.corrcoef(f.rank(method='average'), r.rank(method='average'))[0, 1]
            if not np.isnan(ic):
                ics.append(ic)
    
    ic_mean = np.mean(ics) if ics else 0
    ic_std = np.std(ics) if ics else 1
    ic_ir = ic_mean / ic_std if ic_std > 1e-10 else 0
    
    logger.info(f"[{VERSION}] T+1 IC: {ic_mean:.4f}")
    logger.info(f"[{VERSION}] IC IR: {ic_ir:.2f}")
    
    # 运行回测
    backtest_result = run_backtest(result, result, year)
    
    # 汇总结果
    audit_result = {
        'version': VERSION,
        'year': year,
        'ic': ic_mean,
        'ic_ir': ic_ir,
        'ic_std': ic_std,
        'total_return': backtest_result['total_return'],
        'annual_return': backtest_result['annual_return'],
        'sharpe': backtest_result['sharpe'],
        'max_drawdown': backtest_result['max_drawdown'],
        'final_value': backtest_result['final_value'],
        'selected_factors': alpha.get_selected_factors(),
        'factor_ics': alpha.get_factor_ics(),
        'gvs_stats': alpha.get_gvs_stats(),
        'atg_stats': alpha.get_atg_stats(),
        'icir_stats': alpha.get_icir_stats(),
    }
    
    return audit_result


def generate_report(audit_result: dict) -> str:
    """生成审计报告"""
    version = audit_result['version']
    year = audit_result['year']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    report_path = Path(f"reports/v{version.lower()}_audit_{year}_{timestamp}.md")
    
    ic_status = "[PASS]" if audit_result['ic'] > 0.09 else "[FAIL]"
    ir_status = "[PASS]" if audit_result['ic_ir'] > 0.7 else "[FAIL]"
    return_status = "[PASS]" if audit_result['total_return'] > 0 else "[FAIL]"
    
    overall = "PASSED" if (ic_status == "[PASS]" and ir_status == "[PASS]" and return_status == "[PASS]") else "FAILED"
    
    report = f"""# {version} Fusion Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
**Year**: {year}
**Architecture**: Referee-Player (V156 Signal-Smoothing + V157 IC-IR)

---

## 1. Executive Summary

| Metric | Target | {version} Actual | Status |
|--------|--------|-----------------|--------|
| T+1 Rank IC | > 0.09 | **{audit_result['ic']:.4f}** | {ic_status} |
| IC IR | > 0.7 | **{audit_result['ic_ir']:.2f}** | {ir_status} |
| Total Return | > 0 | **{audit_result['total_return']*100:.2f}%** | {return_status} |
| Sharpe Ratio | > 1.5 | **{audit_result['sharpe']:.2f}** | {"[OK]" if audit_result['sharpe'] > 1.5 else "[NG]"} |

**Overall Assessment**: **{overall}**

---

## 2. {version} vs V156 vs V157 Comparison

| Metric | V156 | V157 | {version} | V156→{version} Δ |
|--------|------|------|-----------|------------------|
| T+1 IC | 0.0924 | 0.0775 | **{audit_result['ic']:.4f}** | {audit_result['ic'] - 0.0924:+.4f} |
| IC IR | 0.58 | 0.49 | **{audit_result['ic_ir']:.2f}** | {audit_result['ic_ir'] - 0.58:+.2f} |
| Total Return | - | 139.85% | **{audit_result['total_return']*100:.2f}%** | - |

---

## 3. Core Features

### 3.1 GARCH-Like Volatility Scaling (V156)

| Parameter | Value |
|-----------|-------|
| Signal Window | {audit_result['gvs_stats'].get('signal_window', 'N/A') if audit_result['gvs_stats'] else 'N/A'} |
| Shrink Threshold | {audit_result['gvs_stats'].get('shrink_threshold', 'N/A') if audit_result['gvs_stats'] else 'N/A'} |
| Mean Shrink Ratio | {get_gvs_value(audit_result['gvs_stats'], 'mean_shrink_ratio'):.3f} |

### 3.2 Adaptive Threshold Gate (V156)

| Parameter | Value |
|-----------|-------|
| Skewness Threshold | {audit_result['atg_stats'].get('skewness_threshold', 'N/A') if audit_result['atg_stats'] else 'N/A'} |
| High Skew Ratio | {get_atg_value(audit_result['atg_stats'], 'high_skew_ratio'):.2%} |
| Mean Gate Weight | {get_atg_value(audit_result['atg_stats'], 'mean_gate_weight'):.3f} |

### 3.3 IC-IR Optimized Weighting (V157)

| Parameter | Value |
|-----------|-------|
| IC Window | {audit_result['icir_stats'].get('ic_window', 'N/A') if audit_result['icir_stats'] else 'N/A'} |
| Total Weight | {get_icir_value(audit_result['icir_stats'], 'total_weight'):.4f} |

### 3.4 Top Selected Factors

| Factor | IC | Weight |
|--------|-----|--------|
"""
    
    factor_ics = audit_result.get('factor_ics', {})
    icir_weights = audit_result.get('icir_stats', {}).get('weights', {})
    
    for factor in sorted(factor_ics.keys(), key=lambda x: abs(factor_ics.get(x, 0)), reverse=True)[:5]:
        ic = factor_ics.get(factor, 0)
        weight = icir_weights.get(factor, 0)
        report += f"| {factor} | {ic:.4f} | {weight:.4f} |\n"
    
    report += f"""
---

## 4. Backtest Performance

| Metric | Value |
|--------|-------|
| Initial Capital | 100,000 |
| Final Value | {audit_result['final_value']:.2f} |
| Total Return | {audit_result['total_return']*100:.2f}% |
| Annual Return | {audit_result['annual_return']*100:.2f}% |
| Sharpe Ratio | {audit_result['sharpe']:.2f} |
| Max Drawdown | {audit_result['max_drawdown']*100:.2f}% |

---

## 5. Conclusion

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > 0.09 | {audit_result['ic']:.4f} | {ic_status} |
| IC IR | > 0.7 | {audit_result['ic_ir']:.2f} | {ir_status} |
| Total Return | > 0 | {audit_result['total_return']*100:.2f}% | {return_status} |

**{overall}**

---

*Report generated by {version} Fusion Audit System*
"""
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding='utf-8')
    
    # 保存 JSON
    json_path = Path(f"reports/v{version.lower()}_audit_{year}_{timestamp}.json")
    json_path.write_text(json.dumps(audit_result, indent=2, default=str), encoding='utf-8')
    
    logger.info(f"[{VERSION}] Report saved to: {report_path}")
    logger.info(f"[{VERSION}] JSON saved to: {json_path}")
    
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description=f"{VERSION} Fusion Audit Runner")
    parser.add_argument('--year', type=int, default=2024, help='Audit year')
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info(f"{VERSION} Fusion Main Entry - V156 Signal-Smoothing + V157 IC-IR")
    logger.info("=" * 80)
    logger.info(f"【{VERSION} Core Features】")
    logger.info("  - GARCH-Like Volatility Scaling: Enabled (V156)")
    logger.info("  - Adaptive Threshold Gate: Enabled (V156)")
    logger.info("  - ORA 3.0 Nonlinear Mining: Enabled (V156)")
    logger.info("  - IC-IR Optimized Weighting: Enabled (V157)")
    logger.info("  - Rolling PAC: Enabled")
    logger.info("  - Target IR: > 0.7")
    logger.info("  - Target IC: > 0.09")
    logger.info("=" * 80)
    
    audit_result = run_audit(args.year)
    report_path = generate_report(audit_result)
    
    logger.info("=" * 80)
    logger.info(f"{VERSION} Audit Complete!")
    logger.info(f"  Year: {args.year}")
    logger.info(f"  IC: {audit_result['ic']:.4f} (target > 0.09)")
    logger.info(f"  IC IR: {audit_result['ic_ir']:.2f} (target > 0.7)")
    logger.info(f"  Total Return: {audit_result['total_return']*100:.2f}%")
    logger.info(f"  Report: {report_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()