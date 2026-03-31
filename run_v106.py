"""
V106 逻辑门控与非线性动态增强 - 回测运行脚本.

【使用方法】
    python run_v106.py

【输出】
- reports/V106_Summary_Report_*.md - 运行总结报告
- reports/v106_audit_*.json - 审计结果 JSON
"""

import sys
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

import sys
sys.path.insert(0, 'src')

from alpha_research_v106 import AlphaResearchV106, VERSION, get_alpha_research
from engine.backtest_referee import BacktestReferee, get_backtest_referee


def generate_test_data(n_samples: int = 5000, n_symbols: int = 100, n_days: int = 50) -> pd.DataFrame:
    """
    生成模拟测试数据。
    
    Args:
        n_samples: 样本数
        n_symbols: 股票数量
        n_days: 交易日数
        
    Returns:
        测试数据 DataFrame
    """
    np.random.seed(42)
    
    # 生成日期
    base_date = pd.Timestamp('2024-01-01')
    dates = pd.date_range(start=base_date, periods=n_days, freq='B')
    
    # 生成股票代码
    symbols = [f"STOCK_{i:04d}" for i in range(n_symbols)]
    
    # 生成数据 - 使用向量化方法
    # 预先生成所有股票的价格路径
    stock_prices = {}
    for symbol in symbols:
        stock_id = int(symbol.split('_')[1])
        # 每个股票有不同的价格路径
        np.random.seed(stock_id + 42)
        price_path = 100 + np.cumsum(np.random.randn(n_days) * 2)
        stock_prices[symbol] = price_path
    
    # 生成所有数据
    data = []
    for day_idx, date in enumerate(dates):
        for symbol in symbols:
            stock_id = int(symbol.split('_')[1])
            close_price = stock_prices[symbol][day_idx]
            
            # 生成成交量 (对数正态分布)
            volume = np.random.lognormal(mean=10, sigma=1)
            
            # 生成其他字段
            high = close_price * (1 + abs(np.random.randn() * 0.02))
            low = close_price * (1 - abs(np.random.randn() * 0.02))
            open_price = low + np.random.rand() * (high - low)
            
            # 生成 T+1, T+3, T+5 收益 (带有一些可预测性)
            # 让某些股票有持续性收益，模拟真实 Alpha
            alpha_signal = (stock_id % 10 - 4.5) * 0.001  # 某些股票有正的 Alpha，某些负的
            
            t1_return = alpha_signal + np.random.randn() * 0.02
            t3_return = alpha_signal * 3 + np.random.randn() * 0.03
            t5_return = alpha_signal * 5 + np.random.randn() * 0.04
            
            data.append({
                'trade_date': date,
                'symbol': symbol,
                'close': close_price,
                'open': open_price,
                'high': high,
                'low': low,
                'volume': volume,
                'amount': volume * close_price,
                'turnover_rate': np.random.rand() * 0.1,
                'total_mv': np.random.lognormal(mean=22, sigma=1),  # 市值
                'industry_code': f'IND_{stock_id % 10:02d}',
                't1_return': t1_return,
                't3_return': t3_return,
                't5_return': t5_return,
            })
    
    df = pd.DataFrame(data)
    
    logger.info(f"Generated test data: {len(df)} rows, {n_symbols} symbols, {n_days} days")
    
    return df


def run_v106_backtest(use_real_data: bool = False) -> dict:
    """
    运行 V106 回测。
    
    Args:
        use_real_data: 是否使用真实数据 (需要配置数据库)
        
    Returns:
        回测结果字典
    """
    logger.info("=" * 80)
    logger.info(f"[{VERSION}] V106 Backtest Started")
    logger.info("=" * 80)
    
    # 准备数据
    if use_real_data:
        logger.info(f"[{VERSION}] Loading real data from database...")
        # TODO: 从数据库加载真实数据
        # 这里需要从数据库或 Parquet 文件加载
        raise NotImplementedError("Real data loading not implemented yet")
    else:
        logger.info(f"[{VERSION}] Generating test data...")
        df = generate_test_data(n_samples=5000, n_symbols=100, n_days=50)
    
    # 初始化 V106 Alpha 引擎
    logger.info(f"[{VERSION}] Initializing AlphaResearchV106...")
    alpha_engine = get_alpha_research(
        use_gating=True,
        enable_ablation=True,
        auto_heal=True
    )
    
    # 计算 Alpha 评分
    logger.info(f"[{VERSION}] Computing alpha scores...")
    score_df = alpha_engine.compute_score(df)
    
    # 获取消融分析结果
    ablation_results = alpha_engine.get_ablation_results()
    gate_states = alpha_engine.get_gate_states()
    healing_records = alpha_engine.get_healing_records()
    
    # IC 稳定性审计
    logger.info(f"[{VERSION}] Running IC stability audit...")
    ic_audit = alpha_engine.audit_ic_stability(score_df)
    
    # IC 衰减审计
    ic_decay = alpha_engine.audit_ic_decay(score_df)
    
    # 因子相关性分析
    corr_matrix = alpha_engine.analyze_factor_correlation(df)
    
    # 输出结果
    logger.info("=" * 80)
    logger.info(f"[{VERSION}] Backtest Results Summary")
    logger.info("=" * 80)
    logger.info(f"[{VERSION}]   T+1 Mean IC: {ic_audit['t1_ic']['mean_ic']:.4f}")
    logger.info(f"[{VERSION}]   T+1 IC Std: {ic_audit['t1_ic']['ic_std']:.4f}")
    logger.info(f"[{VERSION}]   T+1 IC IR: {ic_audit['t1_ic']['ic_ir']:.2f}")
    logger.info(f"[{VERSION}]   IC Strong: {ic_audit['ic_strong']}")
    logger.info(f"[{VERSION}]   IC Stable: {ic_audit['ic_stable']}")
    logger.info(f"[{VERSION}]   Passed: {ic_audit['passed']}")
    
    # 消融分析结果
    if ablation_results:
        logger.info(f"[{VERSION}] Ablation Analysis:")
        logger.info(f"[{VERSION}]   Linear IC: {ablation_results.get('linear_weighted', 'N/A')}")
        logger.info(f"[{VERSION}]   Gated IC: {ablation_results.get('gated', 'N/A')}")
        logger.info(f"[{VERSION}]   Improvement: {ablation_results.get('improvement_percent', 'N/A')}")
    
    # 门控状态
    if gate_states:
        logger.info(f"[{VERSION}] Gate States:")
        for gate_name, state in gate_states.items():
            logger.info(f"[{VERSION}]   {gate_name}: {state}")
    
    # 生成报告
    logger.info(f"[{VERSION}] Generating summary report...")
    report_path = alpha_engine.generate_v106_report()
    
    # 保存 JSON 结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = f"reports/v106_audit_{timestamp}.json"
    Path("reports").mkdir(parents=True, exist_ok=True)
    
    json_result = {
        'version': VERSION,
        'timestamp': timestamp,
        'ic_metrics': {
            't1_ic': ic_audit['t1_ic'],
            'ic_strong': ic_audit['ic_strong'],
            'ic_stable': ic_audit['ic_stable'],
            'passed': ic_audit['passed'],
        },
        'ic_decay': ic_decay,
        'ablation_results': ablation_results,
        'gate_states': gate_states,
        'healing_records': healing_records,
        'config': {
            'use_gating': alpha_engine.use_gating,
            'enable_ablation': alpha_engine.enable_ablation,
            'auto_heal': alpha_engine.auto_heal,
        }
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, default=str)
    
    logger.info(f"[{VERSION}] JSON result saved to: {json_path}")
    
    # 使用 BacktestReferee 进行完整回测
    logger.info(f"[{VERSION}] Running full backtest with BacktestReferee...")
    
    try:
        referee = get_backtest_referee(alpha_engine, output_dir="reports")
        backtest_result = referee.run_audit(df)
        
        logger.info(f"[{VERSION}] Full backtest complete!")
        logger.info(f"[{VERSION}]   Backtest passed: {backtest_result.get('passed', False)}")
        
        if backtest_result.get('t1_ic'):
            logger.info(f"[{VERSION}]   Backtest T+1 IC: {backtest_result['t1_ic'].get('mean_ic', 0):.4f}")
        
    except Exception as e:
        logger.warning(f"[{VERSION}] BacktestReferee audit failed: {e}")
        logger.info(f"[{VERSION}] Continuing with basic analysis...")
    
    logger.info("=" * 80)
    logger.info(f"[{VERSION}] V106 Backtest Complete")
    logger.info("=" * 80)
    
    return {
        'ic_audit': ic_audit,
        'ic_decay': ic_decay,
        'ablation_results': ablation_results,
        'gate_states': gate_states,
        'healing_records': healing_records,
        'report_path': report_path,
        'json_path': json_path,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V106 逻辑门控与非线性动态增强回测")
    parser.add_argument(
        "--use-real-data",
        action="store_true",
        help="使用真实数据 (需要配置数据库)"
    )
    
    args = parser.parse_args()
    
    result = run_v106_backtest(use_real_data=args.use_real_data)
    
    # 打印最终结论
    print("\n" + "=" * 80)
    print(f"V106 回测完成!")
    print("=" * 80)
    print(f"报告路径：{result['report_path']}")
    print(f"JSON 结果：{result['json_path']}")
    
    if result['ic_audit']['passed']:
        print("\n✓ IC 稳定性审计通过!")
        print(f"  T+1 IC: {result['ic_audit']['t1_ic']['mean_ic']:.4f} (> 0.04)")
        print(f"  IC_Std: {result['ic_audit']['t1_ic']['ic_std']:.4f} (< 0.02)")
    else:
        print("\n✗ IC 稳定性审计未通过")
        if not result['ic_audit']['ic_strong']:
            print(f"  T+1 IC ({result['ic_audit']['t1_ic']['mean_ic']:.4f}) < 0.04")
        if not result['ic_audit']['ic_stable']:
            print(f"  IC_Std ({result['ic_audit']['t1_ic']['ic_std']:.4f}) > 0.02")
    
    if result['ablation_results']:
        improvement = result['ablation_results'].get('improvement', 0)
        if improvement > 0.10:
            print(f"\n✓ 逻辑门控有效! IC 提升 {result['ablation_results'].get('improvement_percent', 'N/A')}")
        else:
            print(f"\n⚠ 逻辑门控提升不足 10% ({result['ablation_results'].get('improvement_percent', 'N/A')})")
    
    print("=" * 80)