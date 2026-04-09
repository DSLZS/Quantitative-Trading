"""
V181 策略运行脚本 - 2023-2024 全量审计

【执行指令】
1. 自动运行 2023-2024 全量审计
2. 输出：
   - [1] 因子正交化矩阵
   - [2] 2023/2024 年度对比表
   - [3] 详细的失败/报错自修记录

【零容忍审计约束】
- 初始资金锁定 100,000
- 严禁在回测中通过调整 buy_limit 来虚增成交概率
- 报错自愈：遇到 MySQL 8.0 保留字冲突或数据缺失，立即调用 DataHealer 补全
- 如果 2023 年 Rank IC 低于 0.05，自动进入 Regime_Switch_Logic
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_research_v181 import (
    V181BacktestRunner,
    VERSION,
    TARGET_IC_2023,
    TARGET_IC_2024,
    TARGET_IR_2023,
    TARGET_IR_2024,
    WARMUP_DAYS,
    WARMUP_YEAR,
)
from loguru import logger
from datetime import datetime


def setup_logger():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )


def run_v181_audit():
    """
    运行 V181 全量审计
    
    Returns:
        Dict: 审计结果
    """
    setup_logger()
    
    print("=" * 80)
    print(f"  V181 策略全量审计 (2023-2024)")
    print("=" * 80)
    print()
    
    # 打印 V181 核心改进
    print("【V181 核心改进】")
    print("  1. 因子清洗：移除 reversion_5 和 liquidity_alpha（纯噪声）")
    print("  2. Gram-Schmidt 正交化：volume_price_contradiction 作为主因子")
    print("  3. Rolling IC-IR 动态权重：W_i = IR_i^2 / sum(IR^2)")
    print("  4. Warm-up Buffer：2022 年最后 60 个交易日，解决 2023 启动黑洞")
    print("  5. 报错自愈：自动处理 MySQL 8.0 保留字冲突和数据缺失")
    print()
    
    # 打印性能目标
    print("【性能目标】")
    print(f"  2024: IC > {TARGET_IC_2024}, IR > {TARGET_IR_2024}")
    print(f"  2023: IC > {TARGET_IC_2023}, IR > {TARGET_IR_2023}")
    print()
    
    # 打印约束条件
    print("【零容忍审计约束】")
    print("  - 初始资金：100,000")
    print("  - 禁止调整 buy_limit 虚增成交概率")
    print("  - 报错自愈：自动处理 MySQL 8.0 保留字冲突和数据缺失")
    print("  - IC < 0.05 自动触发 Regime Switch Logic")
    print()
    
    # 初始化回测运行器
    runner = V181BacktestRunner(
        output_dir='reports',
        initial_capital=100000.0
    )
    
    # 运行跨周期审计
    results = runner.run_cross_cycle_audit(years=[2023, 2024])
    
    # 输出结果摘要
    print("\n" + "=" * 80)
    print("  V181 审计结果摘要")
    print("=" * 80)
    
    # [1] 输出因子正交化矩阵
    print("\n【1. 因子正交化矩阵】")
    print(results['orthogonalization_report'])
    
    # [2] 输出 2023/2024 年度对比表
    print("\n【2. 2023/2024 年度对比表】")
    print(results['comparison_table'])
    
    # [3] 输出报错自修记录
    print("\n【3. 报错自修记录】")
    if results['auto_fix_records']:
        for i, record in enumerate(results['auto_fix_records'], 1):
            print(f"  [{i}] [{record['status']}] {record['action']}: {record['details']}")
    else:
        print("  无报错自修记录")
    
    # 输出 Regime Switch 记录
    if results['regime_switch_log']:
        print("\n【4. Regime Switch Log】")
        for i, record in enumerate(results['regime_switch_log'], 1):
            print(f"  [{i}] {record['timestamp'][:19]} - {record['action']}: {record.get('reason', record.get('factor', ''))}")
    
    # 输出验证结果
    print("\n" + "=" * 80)
    print("  验证结果")
    print("=" * 80)
    
    validation = results['validation_passed']
    
    print(f"\n2023 年:")
    print(f"  IC: {validation['2023']['min_ic_actual']:.4f} (目标 > {TARGET_IC_2023}) - {'✓ PASSED' if validation['2023']['passed'] else '✗ FAILED'}")
    print(f"  IR: {validation['2023']['min_ir_actual']:.2f} (目标 > {TARGET_IR_2023})")
    
    print(f"\n2024 年:")
    print(f"  IC: {validation['2024']['min_ic_actual']:.4f} (目标 > {TARGET_IC_2024}) - {'✓ PASSED' if validation['2024']['passed'] else '✗ FAILED'}")
    print(f"  IR: {validation['2024']['min_ir_actual']:.2f} (目标 > {TARGET_IR_2024})")
    
    print(f"\n总体状态：{'✓ PASSED' if validation['overall_passed'] else '✗ FAILED'}")
    
    # 保存 JSON 结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = Path('reports') / f"v181_audit_{timestamp}.json"
    
    import json
    
    # 序列化结果
    serializable_results = {
        'version': VERSION,
        'timestamp': datetime.now().isoformat(),
        'years': results['years'],
        'validation_passed': {
            '2023': {
                'ic_actual': validation['2023']['min_ic_actual'],
                'ic_target': validation['2023']['min_ic_target'],
                'ir_actual': validation['2023']['min_ir_actual'],
                'ir_target': validation['2023']['min_ir_target'],
                'passed': validation['2023']['passed'],
            },
            '2024': {
                'ic_actual': validation['2024']['min_ic_actual'],
                'ic_target': validation['2024']['min_ic_target'],
                'ir_actual': validation['2024']['min_ir_actual'],
                'ir_target': validation['2024']['min_ir_target'],
                'passed': validation['2024']['passed'],
            },
            'overall_passed': validation['overall_passed'],
        },
        'factor_ics': results['results'].get(2024, {}).get('factor_ics', {}),
        'factor_weights': results['results'].get(2024, {}).get('factor_weights', {}),
        'orthogonalization_matrix': results['orthogonalization_matrix'],
        'auto_fix_count': len(results['auto_fix_records']),
        'regime_switch_count': len(results['regime_switch_log']),
        'nag_stats': results['nag_stats'],
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n审计结果已保存至：{json_path}")
    
    # 返回退出码
    return 0 if validation['overall_passed'] else 1


def main():
    """主函数"""
    try:
        exit_code = run_v181_audit()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"V181 审计失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()