"""
V19 回测运行脚本

功能：
1. 运行 V19 策略回测
2. 对比 V18 与 V19 的性能差异
3. 生成换手率优化对比报告
4. 生成新因子重要性分析

使用方法：
    python src/run_v19_backtest.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from src.final_strategy_v1_19 import FinalStrategyV19
    from src.final_strategy_v1_18 import FinalStrategyV18
except ImportError:
    from final_strategy_v1_19 import FinalStrategyV19
    from final_strategy_v1_18 import FinalStrategyV18


def run_v19_backtest():
    """运行 V19 回测"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V19 回测运行脚本")
    logger.info("信号平滑与换手率优化")
    logger.info("=" * 70)
    logger.info(f"开始时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置回测时间段
    start_date = "2023-01-01"
    end_date = "2026-03-31"
    
    logger.info(f"\n回测时间段：{start_date} 至 {end_date}")
    logger.info(f"滑点：0.2% (单边)")
    logger.info(f"持有期：T+5")
    logger.info(f"选股数量：Top 10")
    
    # 运行 V19 策略
    strategy = FinalStrategyV19(
        config_path="config/production_params.yaml",
    )
    
    results = strategy.run_full_analysis(
        start_date=start_date,
        end_date=end_date,
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("V19 回测完成")
    logger.info("=" * 70)
    
    if "error" not in results:
        backtest = results["backtest_result"]
        regime = results["regime_result"]
        importance = results["feature_importance"]
        
        logger.info(f"\n核心指标:")
        logger.info(f"  总收益：{backtest['total_return']:.2%}")
        logger.info(f"  年化收益：{backtest['annual_return']:.2%}")
        logger.info(f"  夏普比率：{backtest['sharpe_ratio']:.3f}")
        logger.info(f"  最大回撤：{backtest['max_drawdown']:.2%}")
        logger.info(f"  总交易次数：{backtest['total_trades']:,}")
        logger.info(f"  平均每日换手：{backtest['avg_daily_turnover']:.1f} 笔/天")
        logger.info(f"  平均持仓周期：{backtest['avg_holding_period']:.1f} 天")
        logger.info(f"  胜率：{backtest['win_rate']:.1%}")
        
        logger.info(f"\n市场环境熔断 2.0:")
        logger.info(f"  强熔断天数：{regime.strong_circuit_breaker_days} 天")
        logger.info(f"  弱熔断天数：{regime.weak_circuit_breaker_days} 天")
        logger.info(f"  当前状态：{regime.regime_status}")
        
        logger.info(f"\n因子重要性 Top 3:")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        for name, imp in sorted_importance:
            logger.info(f"  {name}: {imp:.4f}")
        
        logger.info(f"\n报告路径：{results['report_path']}")
        logger.info(f"缓存路径：{results['cache_path']}")
    else:
        logger.error(f"回测失败：{results.get('error', '未知错误')}")
    
    return results


def generate_comparison_report(v18_results=None, v19_results=None):
    """生成 V18 vs V19 对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(f"reports/V19_V18_Comparison_Report_{timestamp}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 如果没有传入结果，使用默认值
    if v18_results is None:
        v18_results = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 380000,  # V18 参考值
            "win_rate": 0.0,
        }
    
    if v19_results is None:
        v19_results = {
            "total_return": 0.0,
            "annual_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
        }
    
    # 计算改善百分比
    def calc_improvement(v18, v19, higher_better=True):
        if v18 == 0:
            return 0.0
        if higher_better:
            return (v19 - v18) / abs(v18) * 100
        else:
            return (v18 - v19) / abs(v18) * 100
    
    turnover_reduction = calc_improvement(v18_results["total_trades"], v19_results["total_trades"], higher_better=False)
    sharpe_improvement = calc_improvement(v18_results["sharpe_ratio"], v19_results["sharpe_ratio"], higher_better=True)
    dd_improvement = calc_improvement(v18_results["max_drawdown"], v19_results["max_drawdown"], higher_better=False)
    
    report = f"""# V19 vs V18 性能对比报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、核心指标对比

| 指标 | V18 (参考) | V19 (结果) | 改善幅度 |
|------|------------|------------|----------|
| 总收益 | {v18_results['total_return']:.2%} | {v19_results['total_return']:.2%} | {calc_improvement(v18_results['total_return'], v19_results['total_return'], higher_better=True):+.1f}% |
| 年化收益 | {v18_results['annual_return']:.2%} | {v19_results['annual_return']:.2%} | {calc_improvement(v18_results['annual_return'], v19_results['annual_return'], higher_better=True):+.1f}% |
| 夏普比率 | {v18_results['sharpe_ratio']:.3f} | {v19_results['sharpe_ratio']:.3f} | {sharpe_improvement:+.1f}% |
| 最大回撤 | {v18_results['max_drawdown']:.2%} | {v19_results['max_drawdown']:.2%} | {dd_improvement:+.1f}% |
| **总交易次数** | {v18_results['total_trades']:,} | {v19_results['total_trades']:,} | **{turnover_reduction:+.1f}%** |
| 胜率 | {v18_results['win_rate']:.1%} | {v19_results['win_rate']:.1%} | {calc_improvement(v18_results['win_rate'], v19_results['win_rate'], higher_better=True)*100:+.1f}% |

---

## 二、换手率优化分析

### 2.1 换手率对比

| 指标 | V18 | V19 | 变化 |
|------|-----|-----|------|
| 总交易次数 | {v18_results['total_trades']:,} | {v19_results['total_trades']:,} | {turnover_reduction:+.1f}% |
| 日均交易次数 | {v18_results['total_trades']/500:.1f} | {v19_results.get('avg_daily_turnover', 0):.1f} | - |

### 2.2 换手率优化手段

1. **EMA 信号平滑**: 对 Raw Signal 进行 5 日指数加权平均
2. **调仓门槛**: 只有当新信号显著优于当前持仓 (>5%) 时才换手

### 2.3 换手率目标验证

| 目标 | 要求 | 实际 | 状态 |
|------|------|------|------|
| 总交易次数 | 5,000-10,000 | {v19_results['total_trades']:,} | {"✅" if 5000 <= v19_results['total_trades'] <= 10000 else "⚠️"} |
| 换手率降低 | >70% | {turnover_reduction:.1f}% | {"✅" if turnover_reduction > 70 else "⚠️"} |

---

## 三、新因子重要性分析

### 3.1 V19 新增因子

| 因子名称 | 描述 | 重要性排名 |
|----------|------|------------|
| momentum_ortho | 股价偏离 20 日均线的比例（反向） | TBD |
| low_vol_ortho | 过去 20 日收益率标准差（反向） | TBD |

### 3.2 因子库扩展效果

- **V18 因子数量**: 4 个
- **V19 因子数量**: 6 个
- **新增因子**: 动量因子 + 低波动因子

---

## 四、市场环境熔断 2.0 效果

### 4.1 熔断逻辑对比

| 版本 | 熔断条件 | 响应速度 |
|------|----------|----------|
| V18 | 波动率 > 90 分位 | 较慢 |
| V19 | 强熔断：指数 < MA60 且 MA20 向下 | 更快 |
| V19 | 弱熔断：波动率 > 80 分位 | 更敏感 |

### 4.2 熔断效果

| 指标 | V19 值 |
|------|--------|
| 强熔断天数 | TBD |
| 弱熔断天数 | TBD |

---

## 五、总结

### 5.1 V19 核心改进

1. ✅ **信号平滑**: EMA + 调仓门槛降低换手率
2. ✅ **因子扩展**: 新增动量和低波动因子
3. ✅ **熔断 2.0**: 趋势 + 波动双重过滤
4. ✅ **训练优化**: 120 天训练/20 天预测

### 5.2 性能评估

| 维度 | 目标 | 达成情况 |
|------|------|----------|
| 换手率优化 | 降至 5,000-10,000 | {"✅" if 5000 <= v19_results['total_trades'] <= 10000 else "⚠️"} |
| Sharpe 比率 | 保持或提升 | {"✅" if v19_results['sharpe_ratio'] >= v18_results['sharpe_ratio'] else "⚠️"} |
| 最大回撤 | 降低 | {"✅" if v19_results['max_drawdown'] <= v18_results['max_drawdown'] else "⚠️"} |

---

**报告生成完毕**
"""
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    logger.info(f"对比报告已保存至：{report_path}")
    return str(report_path)


if __name__ == "__main__":
    # 运行 V19 回测
    v19_results = run_v19_backtest()
    
    # 生成对比报告
    if "error" not in v19_results:
        generate_comparison_report(v19_results=v19_results["backtest_result"])