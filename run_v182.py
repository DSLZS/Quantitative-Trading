"""
V182 策略运行脚本 - 2023-2024 全量审计

【执行指令】
1. 自动运行 2023-2024 全量审计
2. 输出：
   - [1] 自我迭代循环历史
   - [2] 2023/2024 年度对比表
   - [3] 因子 IC 和权重

【V182 核心改进】
- 回滚到 V172 核心逻辑
- Lowdin 对称正交化
- IC-Rolling-Significance 加权
- Self-Correction Loop
"""

import sys
import os
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from alpha_research_v182 import (
    V182BacktestRunner,
    VERSION,
    TARGET_IC_2023,
    TARGET_IC_2024,
    TARGET_IR_2023,
    TARGET_IR_2024,
    WARMUP_DAYS,
    WARMUP_YEAR,
    IC_THRESHOLD_FOR_ITERATION,
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


def run_v182_audit():
    """
    运行 V182 全量审计
    
    Returns:
        Dict: 审计结果
    """
    setup_logger()
    
    print("=" * 80)
    print(f"  V182 策略全量审计 (2023-2024)")
    print("=" * 80)
    print()
    
    # 打印 V182 核心改进
    print("【V182 核心改进】")
    print("  1. 回滚到 V172 核心逻辑 - 6 因子配置")
    print("  2. Lowdin 对称正交化 - 防止过度依赖单一主因子")
    print("  3. IC-Rolling-Significance 加权 - 仅对 p-value < 0.05 的因子分配权重")
    print("  4. Self-Correction Loop - IC < 0.08 时自动调整 NAG 阈值")
    print("  5. Warm-up Buffer - 2022 年底 60 天数据")
    print()
    
    # 打印性能目标
    print("【性能目标】")
    print(f"  2024: IC > {TARGET_IC_2024}, IR > {TARGET_IR_2024}")
    print(f"  2023: IC > {TARGET_IC_2023}, IR > {TARGET_IR_2023}")
    print()
    
    # 打印自我迭代参数
    print("【Self-Correction Loop】")
    print(f"  - IC 阈值：{IC_THRESHOLD_FOR_ITERATION}")
    print("  - 最大迭代次数：5")
    print("  - NAG 阈值调整策略：[0.5, 0.6, 0.7, 0.4, 0.3]")
    print()
    
    # 初始化回测运行器
    runner = V182BacktestRunner(
        output_dir='reports',
        initial_capital=100000.0
    )
    
    # 运行跨周期审计
    results = runner.run_cross_cycle_audit(years=[2023, 2024])
    
    # 输出结果摘要
    print("\n" + "=" * 80)
    print("  V182 审计结果摘要")
    print("=" * 80)
    
    # [1] 输出自我迭代循环历史
    print("\n【1. Self-Correction Loop History】")
    if results['iteration_history']:
        for record in results['iteration_history']:
            print(f"  Iteration {record['iteration']} (Year {record['year']}): IC={record['ic']:.4f}, NAG threshold={record['nag_threshold']}", flush=True)
    else:
        print("  无迭代记录")
    
    # [2] 输出 2023/2024 年度对比表
    print("\n【2. 2023/2024 年度对比表】")
    print(results['comparison_table'])
    
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
    json_path = Path('reports') / f"v182_audit_{timestamp}.json"
    
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
        'iteration_history': results['iteration_history'],
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
        exit_code = run_v182_audit()
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"V182 审计失败：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()