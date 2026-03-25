# 量化交易系统 V72

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📊 数据版本

- **数据源**: Tushare Pro
- **数据覆盖**: 2024 年全年（2024-01-01 至 2024-12-31）
- **个股资金流**: 1,233,188 条记录
- **行业日线**: 106,238 条记录

## 🎯 核心审计指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| SNR (信噪比) | > 0.15 | 信号质量核心指标 |
| Rank IC | > 0.03 | 因子预测能力 |
| 最大回撤 | < 15% | 风险控制指标 |
| 夏普比率 | > 1.0 | 风险调整后收益 |

## 📁 项目结构

```
src/
├── core/              # 核心算法逻辑
│   ├── factor_engine.py      # 因子引擎
│   ├── ic_calculator.py      # IC 计算
│   ├── feature_pipeline.py   # 特征管道
│   ├── model_trainer.py      # 模型训练
│   ├── factor_validator.py   # 因子验证
│   ├── oos_validator.py      # 样本外验证
│   └── parameter_scan.py     # 参数扫描
│
├── engine/            # 回测执行引擎
│   ├── backtest_engine.py    # 回测引擎核心
│   ├── backtester.py         # 回测器
│   ├── visualizer.py         # 可视化
│   ├── execution_optimizer.py # 执行优化
│   └── walk_forward_backtester.py  # 滚动回测
│
├── loaders/           # 数据同步脚本
│   ├── v71_tushare_boot.py   # Tushare 数据补完
│   ├── v70_data_loader.py    # V70 数据加载器
│   └── sync_*.py             # 各类数据同步
│
├── db_manager.py       # 数据库管理器
├── data_loader.py      # 数据加载接口
├── predict_next_day.py # 次日预测
└── daily_trade_advisor.py  # 每日交易顾问
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 Tushare

编辑 `config/settings.yaml`，配置 Tushare Token：

```yaml
TUSHARE_CONFIG = {
    'token': 'your_token_here',
    'rate_limit': 10,
}
```

### 3. 运行回测

```bash
# 运行 V70 回测（推荐）
python src/run_v70_backtest.py

# 运行 V68 回测（稳定版本）
python src/run_v68_backtest.py

# 运行滚动回测
python src/engine/walk_forward_backtester.py
```

### 4. 数据同步

```bash
# 同步 Tushare 数据（2024 年全年）
python src/loaders/v71_tushare_boot.py

# 同步股票元数据
python src/loaders/sync_stock_metadata.py

# 同步行业数据
python src/loaders/sync_industry_and_market_cap.py
```

## 📈 性能报告

生成回测报告后，查看 `reports/` 目录下的最新报告：

```bash
# 查看最新报告
ls -lt reports/*.md | head -1
```

## 🔧 工具脚本

### 项目维护

```bash
# 清理归档旧版本
python src/cleanup_and_refactor.py

# 运行预检查
python src/preflight_check.py

# 因子验证
python src/core/factor_validator.py
```

### 压力测试

```bash
# 运行压力测试
python src/final_stress_tester.py
```

## 📝 版本历史

| 版本 | 日期 | 说明 |
|------|------|------|
| V72 | 2026-03-25 | 架构规范化，项目瘦身 |
| V71 | 2026-03-25 | Tushare 数据强制补完计划 |
| V70 | 2026-03-24 | 数据加载器优化 |
| V69 | 2026-03-24 | 数据补全与验证 |
| V68 | 2026-03-24 | 稳定版本交付 |

## 🔬 实验记录

### V72 实验记录 (2026-03-25)

**核心算法**: SNR 审计 + 行业背离滤网

**配置**:
- 初始资金：100,000 元
- 止损：5%，止盈：15%
- SNR 阈值：Z-Score > 1.5 且 SNR > 0.15
- 行业背离：个股主力为正但行业连续 3 日净流出则剔除
- 市场宽度：行业净流出占比 > 80% 强制空仓

**回测结果 (2024 年全年)**:
| 指标 | 数值 | 状态 |
|------|------|------|
| 最终价值 | 14,698.83 元 | ❌ |
| 总盈亏 | -85,301.17 元 (-85.30%) | ❌ |
| 年化收益 | -97.06% | ❌ |
| 夏普比率 | -6.17 | ❌ |
| 最大回撤 | 85.44% | ❌ |
| 交易次数 | 96 | - |
| 胜率 | 38.5% | ❌ |
| 盈亏比 | 1.59 | ✓ |
| Rank IC | 0.0000 | ❌ |
| 月度 Rank IC 均值 | 0.0000 | ❌ |

**问题分析**:
1. SNR 过滤条件过于严格，导致信号稀少
2. 横截面 Z-Score 计算方式不适合资金流数据特性
3. 行业数据覆盖不足（仅 439 只股票），行业背离滤网效果有限
4. 资金流因子在 2024 年市场环境下失效

**结论**: V72 策略逻辑需要重新设计，资金流 SNR 因子需要与其他因子（如动量、估值）结合使用。

## 📄 License

MIT License
