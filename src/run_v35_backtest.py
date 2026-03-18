#!/usr/bin/env python
"""
V35 Truth Engine 回测运行脚本

使用方法:
    python src/run_v35_backtest.py
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from v35_truth_engine import AutoRunner, verify_results, v35_audit
from loguru import logger


def main():
    """运行 V35 真实回测"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("V35 TRUTH ENGINE - 拒绝欺诈，重构真实性")
    logger.info("=" * 80)
    
    # 创建 AutoRunner
    runner = AutoRunner(
        start_date="2025-01-01",
        end_date="2026-03-18",
        initial_capital=100000.0,
    )
    
    try:
        result = runner.run()
        
        # 最终验证
        logger.info("\n" + "=" * 80)
        logger.info("V35 FINAL VERIFICATION")
        logger.info("=" * 80)
        
        verified, errors = verify_results(result)
        
        if verified:
            logger.info("✅ V35 验证通过 - 数据强一致性检查 PASSED")
        else:
            logger.error("❌ V35 验证失败 - 数据强一致性检查 FAILED")
            for err in errors:
                logger.error(f"  - {err}")
        
        # 打印 V34 欺诈消除声明
        logger.info("\n[V34 欺诈逻辑消除声明]")
        for item in result.get('v34_fraud_eliminated', []):
            logger.info(f"  ✅ {item}")
        
        return result
        
    except Exception as e:
        logger.error(f"V35 回测失败：{e}")
        logger.error("V35 设计原则：宁可崩溃，也不要无效数据")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()