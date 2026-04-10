#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V183 回测运行脚本

【V183 核心改进】
1. 回滚到 V172 ORM 逻辑 - 正交残差提取
2. |IC|^1.0 加权 - V172 的核心算法
3. IC-Rolling-Significance 筛选 - 仅对 p-value < 0.05 的因子分配权重
4. Self-Correction Loop - IC < 0.08 时自动调整 NAG 阈值
5. Warm-up Buffer - 2022 年底 60 天数据

【性能目标】
2024: IC > 0.1, IR > 0.6
2023: IC > 0.06, IR > 0.45
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)

def run_v183_audit():
    """运行 V183 审计"""
    from src.alpha_research_v183 import V183BacktestRunner, TARGET_IC_2024, TARGET_IR_2024, TARGET_IC_2023, TARGET_IR_2023
    
    logger.info("=" * 70)
    logger.info("  V183 策略全量审计 (2023-2024)")
    logger.info("=" * 70)
    
    logger.info("""
【V183 核心改进】
  1. 回滚到 V172 ORM 逻辑 - 正交残差提取
  2. |IC|^1.0 加权 - V172 的核心算法
  3. IC-Rolling-Significance 加权 - 仅对 p-value < 0.05 的因子分配权重
  4. Self-Correction Loop - IC < 0.08 时自动调整 NAG 阈值
  5. Warm-up Buffer - 2022 年底 60 天数据

【性能目标】
  2024: IC > 0.1, IR > 0.6
  2023: IC > 0.06, IR > 0.45

【Self-Correction Loop】
  - IC 阈值：0.08
  - 最大迭代次数：5
  - NAG 阈值调整策略：[0.5, 0.6, 0.7, 0.4, 0.3]
""")
    
    runner = V183BacktestRunner(output_dir='reports', initial_capital=100000.0)
    results = runner.run_cross_cycle_audit(years=[2023, 2024])
    
    # 输出详细结果
    logger.info("\n" + "=" * 70)
    logger.info("  V183 审计结果摘要")
    logger.info("=" * 70)
    
    logger.info("\n【1. Self-Correction Loop History】")
    for record in results['iteration_history']:
        logger.info(f"  Iteration {record['iteration']} (Year {record['year']}): IC={record['ic']:.4f}, NAG threshold={record['nag_threshold']}")
    
    logger.info("\n【2. 2023/2024 年度对比表】")
    print(results['comparison_table'], flush=True)
    
    logger.info("\n================================================================================")
    logger.info("  验证结果")
    logger.info("================================================================================")
    
    validation = results['validation_passed']
    
    logger.info("\n2023 年:")
    logger.info(f"  IC: {validation['2023']['min_ic_actual']:.4f} (目标 > {TARGET_IC_2023}) - {'✓ PASSED' if validation['2023']['passed'] else '✗ FAILED'}")
    logger.info(f"  IR: {validation['2023']['min_ir_actual']:.2f} (目标 > {TARGET_IR_2023})")
    
    logger.info("\n2024 年:")
    logger.info(f"  IC: {validation['2024']['min_ic_actual']:.4f} (目标 > {TARGET_IC_2024}) - {'✓ PASSED' if validation['2024']['passed'] else '✗ FAILED'}")
    logger.info(f"  IR: {validation['2024']['min_ir_actual']:.2f} (目标 > {TARGET_IR_2024})")
    
    logger.info(f"\n总体状态：{'✓ PASSED' if validation['overall_passed'] else '✗ FAILED'}")
    
    # 保存 JSON 结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = Path('reports') / f"v183_audit_{timestamp}.json"
    
    json_result = {
        'version': 'V183',
        'timestamp': datetime.now().isoformat(),
        'validation': validation,
        'targets': {
            '2024': {'ic': TARGET_IC_2024, 'ir': TARGET_IR_2024},
            '2023': {'ic': TARGET_IC_2023, 'ir': TARGET_IR_2023},
        },
        'iteration_history': results['iteration_history'],
        'nag_stats': results.get('nag_stats', {}),
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n审计结果已保存至：{json_path}")
    
    return 0 if validation['overall_passed'] else 1


def main():
    """主函数"""
    try:
        exit_code = run_v183_audit()
        return exit_code
    except Exception as e:
        logger.error(f"V183 审计失败：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())