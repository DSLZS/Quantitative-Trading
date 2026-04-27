# CURRENT_STATE.md - 代码库快照摘要

> **最后更新**: 2026-04-27  
> **当前版本**: V218  
> **状态**: ❌ V218未通过IC阈值（2024年IC < 0.05）

---

## 基本信息

| 项目 | 值 |
|------|-----|
| **最后成功编译版本** | V218 |
| **编译时间** | 2026-04-27 |
| **当前因子公式** | `score = W_t * score_rev + (1 - W_t) * score_mom` |
| **权重计算** | `W_t = sigmoid(1.5 * volatility_norm + 1.0 * trend_strength)` |
| **市场状态分类** | CRISIS / TREND / NORMAL |

---

## 当前因子公式

### 反转得分 (Score_Rev)
```
score_rev = 0.4 * reversal_5d_rank + 0.3 * reversal_10d_rank + 0.3 * reversal_20d_rank
```

### 动量得分 (Score_Mom)
```
score_mom = 0.5 * momentum_20d_rank + 0.3 * mid_momentum_rank + 0.2 * volatility_adjusted_momentum
```

### 最终得分
```
final_score = W_t * score_rev + (1 - W_t) * score_mom
```

### 市场状态判定
```
volatility_20 = std(ret_20)
trend_20 = ma(close, 20) / close

if volatility_20 > threshold_high and trend_20 < 0.95:
    state = CRISIS  # 高波动 + 下跌 -> 反转策略
elif volatility_20 < threshold_low and trend_20 > 1.05:
    state = TREND   # 低波动 + 上涨 -> 动量策略
else:
    state = NORMAL  # 正常区间
```

---

## 数据依赖

### 数据库表

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| `stock_daily` | 日行情数据 | trade_date, symbol, open, high, low, close, pre_close, pct_chg, volume, amount, turnover_rate, industry_code, total_mv, is_st |
| `stock_fund_flow` | 资金流数据 | trade_date, symbol, net_main_amount, net_main_rate |

### CSV/Parquet字段需求

| 字段 | 类型 | 用途 |
|------|------|------|
| trade_date | int (YYYYMMDD) | 交易日期 |
| symbol | str | 股票代码 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| close | float | 收盘价 |
| pre_close | float | 前收盘价 |
| pct_chg | float | 涨跌幅 |
| volume | float | 成交量 |
| amount | float | 成交额 |
| turnover_rate | float | 换手率 |
| industry_code | str | 行业代码 |
| total_mv | float | 总市值 |
| is_st | bool | 是否ST |
| net_main_amount | float | 主力净流入 |
| net_main_rate | float | 主力净流入占比 |

---

## 回测配置（锁定）

| 参数 | 值 | 来源 |
|------|-----|------|
| 初始资金 | 100,000 | backtest_referee.py |
| 佣金率 | 0.03% (万分之三) | backtest_referee.py |
| 印花税率 | 0.1% (千分之一) | backtest_referee.py |
| 滑点率 | 0.05% (万分之五) | backtest_referee.py |
| 总费率 | 0.18% (单边) | 计算值 |
| 持仓数量 | 50只 | backtest_referee.py |
| 单股仓位 | 2% | backtest_referee.py |
| IC阈值 | 0.05 | backtest_referee.py |
| IC IR阈值 | 0.60 | backtest_referee.py |

---

## 已知未解决问题

### 🔴 严重问题

| 问题 | 描述 | 影响 |
|------|------|------|
| **2024年IC为负** | 反转/动量因子在2024年失去预测能力 | V218未通过 |
| **门控失效** | 市场状态门控在牛市未生效 | 策略选择性失效 |
| **线性局限** | 线性组合可能已达到极限 | 需要非线性方法 |

### 🟡 待优化问题

| 问题 | 描述 | 优先级 |
|------|------|--------|
| **因子库单一** | 仅使用价格和成交量因子 | 高 |
| **市场状态粗糙** | 仅用波动率和趋势划分 | 中 |
| **无行业控制** | 未进行行业中性化 | 低 |
| **无市值控制** | 未控制市值效应 | 低 |

### 🟢 改进建议

| 建议 | 描述 | 预期效果 |
|------|------|----------|
| 引入LightGBM | 使用非线性模型合成因子 | 提升2024年IC |
| 另类数据 | 集成新闻舆情、分析师预期 | 增加alpha来源 |
| 波动率调整动量 | 动量/波动率比值 | 改善动量因子稳定性 |
| 截面离散度 | 截面收益率离散度 | 捕捉市场情绪 |

---

## 核心文件清单

### 必须保留（核心代码）

| 文件 | 用途 | 状态 |
|------|------|------|
| `src/alpha_model_v218.py` | Alpha模型（Player） | ✅ 保留 |
| `src/backtest_engine.py` | 回测引擎（Referee） | ✅ 保留 |
| `src/engine/backtest_referee.py` | 不可变裁判引擎 | ✅ 保留 |
| `src/engine/__init__.py` | 模块初始化 | ✅ 保留 |
| `run_v218.py` | 运行入口 | ✅ 保留 |
| `config/factors.yaml` | 因子配置 | ✅ 保留 |
| `config/settings.yaml` | 系统配置 | ✅ 保留 |

### 辅助文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `scripts/diagnose_ic.py` | IC诊断 | ✅ 保留 |
| `README.md` | 项目文档 | ✅ 已更新 |
| `ALPHA_HISTORY.md` | 历史日志 | ✅ 已创建 |
| `CURRENT_STATE.md` | 当前快照 | ✅ 本文件 |
| `TODOS.md` | 改进方向 | ✅ 已创建 |
| `PROJECT_ARCHITECTURE.md` | 架构文档 | ✅ 已创建 |

---

## 环境要求

| 项目 | 值 |
|------|-----|
| Python | 3.13.x |
| 数据库 | MySQL 8.0+ |
| 内存 | 建议 16GB+ |
| 磁盘 | 建议 50GB+（数据缓存） |

---

*本文件由AI Assistant于2026-04-27生成，作为代码库的快照摘要。*