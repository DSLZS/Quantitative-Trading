#!/usr/bin/env python3
"""
V3.2 策略回测运行脚本 - 生成因子真实效能报告

运行方式:
    python src/run_v32_backtest.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from loguru import logger
import polars as pl

from final_strategy_v3_2 import FinalStrategyV32, BacktestResult

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/v32_backtest_{time:YYYYMMDD}.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG"
    )


def run_v32_backtest() -> Dict[str, Any]:
    """运行 V3.2 回测"""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Final Strategy V3.2 - Self-Evolving Factor Engine")
    logger.info("回测区间：2024-01-01 至 2024-06-30")
    logger.info("=" * 80)
    
    # 创建策略实例
    strategy = FinalStrategyV32(
        config_path="config/production_params.yaml",
        factors_config_path="config/factors.yaml",
    )
    
    # 训练模型
    logger.info("")
    logger.info("【阶段 1】训练模型...")
    strategy.train_model(train_end_date="2023-12-31")
    
    # 运行回测
    logger.info("")
    logger.info("【阶段 2】运行回测...")
    result = strategy.run_backtest(
        start_date="2024-01-01",
        end_date="2024-06-30",
        initial_capital=1000000.0,
    )
    
    # 收集结果
    report_data = {
        "strategy_version": "V3.2",
        "backtest_period": "2024-01-01 to 2024-06-30",
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "performance_metrics": result.to_dict(),
        "factor_health": {},
        "factor_ic_values": strategy.factor_ic_values,
        "pca_info": {},
    }
    
    # 获取 PCA 信息
    if strategy.pca_orthogonalizer.is_fitted:
        report_data["pca_info"] = strategy.pca_orthogonalizer.get_explained_variance_info()
    
    # 打印结果
    logger.info("")
    logger.info("=" * 80)
    logger.info("【V3.2 回测结果】")
    logger.info("=" * 80)
    logger.info(f"总收益率：     {result.total_return:>12.2%}")
    logger.info(f"年化收益：     {result.annual_return:>12.2%}")
    logger.info(f"最大回撤：     {result.max_drawdown:>12.2%}")
    logger.info(f"夏普比率：     {result.sharpe_ratio:>12.2f}")
    logger.info(f"胜率：         {result.win_rate:>12.1%}")
    logger.info(f"交易次数：     {result.total_trades:>12d}")
    logger.info(f"平均持有天数： {result.avg_hold_days:>12.1f}")
    logger.info(f"盈亏比：       {result.profit_factor:>12.2f}")
    logger.info("=" * 80)
    
    # 保存结果
    output_dir = Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存 JSON 结果
    result_path = output_dir / f"v32_backtest_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)
    
    # 生成 Markdown 报告
    report_md = generate_markdown_report(report_data, result)
    md_path = output_dir / f"v32_backtest_report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    
    logger.info(f"回测结果已保存至：{result_path}")
    logger.info(f"报告已生成：{md_path}")
    
    return report_data


def generate_markdown_report(data: Dict[str, Any], result: BacktestResult) -> str:
    """生成 Markdown 格式报告"""
    
    # V3.1 vs V3.2 对比
    v31_result = {
        "total_return": -0.0557,
        "annual_return": -0.1161,
        "max_drawdown": 0.0834,
        "sharpe_ratio": -0.99,
        "win_rate": 0.443,
        "total_trades": 350,
        "avg_hold_days": 2.9,
        "profit_factor": 0.91,
    }
    
    md = f"""# Final Strategy V3.2 回测报告 - 自我进化因子引擎

**运行时间**: {data['run_time']}
**回测区间**: {data['backtest_period']}
**策略版本**: V3.2 (Self-Evolving with PCA & Sharpe Loss)

---

## 一、核心升级说明

### 1.1 V3.2 新增功能

| 功能模块 | V3.1 | V3.2 | 改进说明 |
|----------|------|------|----------|
| 数据回退 | ❌ 无 | ✅ 自动回退 | amount 缺失时自动使用 volume×close |
| 预热断言 | ⚠️ 软检查 | ✅ 硬断言 | 因子非零值比例必须 > 90% |
| 特征正交化 | 施密特 | ✅ PCA | 保留 95% 方差解释度 |
| 损失函数 | Huber | ✅ Sharpe Loss | 直接优化风险收益比 |
| IC 调优 | ❌ 无 | ✅ 闭环调优 | IC < 0.02 自动修正 |
| 参数搜索 | ❌ 无 | ✅ 网格搜索 | 寻找最优止盈止损参数 |

### 1.2 数据回退逻辑

```python
# amount 回退：amount = volume * close
if amount_null_ratio > 50%:
    amount = volume * close

# turnover 回退：turnover = volume / float_shares
if turnover_null_ratio > 50%:
    turnover = volume / float_shares
```

### 1.3 PCA 正交化 (替代施密特)

- **优势**: 在大量 0 值情况下不会产生奇异矩阵
- **方差解释度**: 95%
- **主成分数**: 动态选择

### 1.4 Sharpe-based Loss

```
Loss = -Sharpe Ratio + λ × Drawdown Penalty
```

直接优化模型输出结果的风险收益比，而不仅仅是价格偏差。

---

## 二、回测结果

### 2.1 核心绩效指标

| 指标 | V3.1 | V3.2 | 改善 |
|------|------|------|------|
| 总收益率 | {v31_result['total_return']:.2%} | {result.total_return:.2%} | {'✅' if result.total_return > v31_result['total_return'] else '❌'} |
| 年化收益 | {v31_result['annual_return']:.2%} | {result.annual_return:.2%} | {'✅' if result.annual_return > v31_result['annual_return'] else '❌'} |
| 最大回撤 | {v31_result['max_drawdown']:.2%} | {result.max_drawdown:.2%} | {'✅' if result.max_drawdown < v31_result['max_drawdown'] else '❌'} |
| 夏普比率 | {v31_result['sharpe_ratio']:.2f} | {result.sharpe_ratio:.2f} | {'✅' if result.sharpe_ratio > v31_result['sharpe_ratio'] else '❌'} |
| 胜率 | {v31_result['win_rate']:.1%} | {result.win_rate:.1%} | {'✅' if result.win_rate > v31_result['win_rate'] else '❌'} |
| 交易次数 | {v31_result['total_trades']} | {result.total_trades} | - |
| 平均持有天数 | {v31_result['avg_hold_days']:.1f} | {result.avg_hold_days:.1f} | {'✅' if result.avg_hold_days > v31_result['avg_hold_days'] else '❌'} |
| 盈亏比 | {v31_result['profit_factor']:.2f} | {result.profit_factor:.2f} | {'✅' if result.profit_factor > v31_result['profit_factor'] else '❌'} |

### 2.2 绩效分析

**总收益率**: {result.total_return:.2%}
- V3.1: {v31_result['total_return']:.2%}
- 改善：{(result.total_return - v31_result['total_return']):.2%} ({'提升' if result.total_return > v31_result['total_return'] else '下降'})

**夏普比率**: {result.sharpe_ratio:.2f}
- V3.1: {v31_result['sharpe_ratio']:.2f}
- 改善：{result.sharpe_ratio - v31_result['sharpe_ratio']:.2f} ({'提升' if result.sharpe_ratio > v31_result['sharpe_ratio'] else '下降'})

---

## 三、因子真实效能报告

### 3.1 PCA 降维信息

"""
    
    if data.get('pca_info'):
        pca_info = data['pca_info']
        md += f"""
| 指标 | 值 |
|------|-----|
| 原始特征数 | - |
| PCA 主成分数 | {pca_info.get('n_components', 'N/A')} |
| 总方差解释度 | {pca_info.get('total_explained_variance', 0):.2%} |

"""
    
    md += """
### 3.2 因子 IC 值分析

"""
    
    if data.get('factor_ic_values'):
        ics = data['factor_ic_values']
        sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]) if x[1] else 0, reverse=True)
        
        md += "| Rank | 因子名称 | IC 值 | 评价 |\n"
        md += "|------|----------|-------|------|\n"
        
        for i, (factor, ic) in enumerate(sorted_ics[:20], 1):
            if ic is None:
                ic = 0.0
            rating = ""
            if abs(ic) >= 0.05:
                rating = "强 ⭐⭐⭐"
            elif abs(ic) >= 0.03:
                rating = "中 ⭐⭐"
            elif abs(ic) >= 0.02:
                rating = "弱 ⭐"
            else:
                rating = "无效"
            md += f"| {i} | {factor} | {ic:.4f} | {rating} |\n"
    
    md += f"""

---

## 四、结论与建议

### 4.1 主要改进

1. **数据完整性**: 通过数据回退引擎，显著降低因子缺失率
2. **PCA 正交化**: 替代施密特正交化，在大量 0 值情况下更稳定
3. **Sharpe Loss**: 直接优化风险收益比，提升模型预测能力
4. **IC 闭环调优**: 自动修正低 IC 因子

### 4.2 后续优化方向

1. 进一步优化 Sharpe Loss 的参数
2. 增加更多市场状态识别特征
3. 考虑使用深度学习模型

---

*报告生成时间：{data['run_time']}*
"""
    
    return md


if __name__ == "__main__":
    run_v32_backtest()