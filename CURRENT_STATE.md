# CURRENT_STATE.md - 代码库快照摘要

> **最后更新**: 2026-04-29  
> **当前版本**: V227  
> **状态**: ❌ V227未通过IC阈值（2020=0.0363, 2022=0.0574, 2024=0.0255）

---

## 基本信息

| 项目 | 值 |
|------|-----|
| **当前活跃版本** | V227 |
| **编译时间** | 2026-04-29 |
| **当前因子公式** | 极端超卖(50%) + 放量下跌(25%) + 低波动(25%) |
| **特征标准化** | 截面百分位排名 (percentile rank) |
| **最终输出** | `rank(pct=True)` 排名百分比 |

---

## 当前因子公式（V227 - Extreme Reversal with Volume Confirmation）

### 特征工程（3个因子）
```
反转特征:   ret_5d = close / close.shift(5) - 1
f_extreme_os = -ret_5d (跌幅越大信号越强)

量价特征:   vol_ratio = volume / volume.rolling(5).mean()
            is_down_day = (close < open).astype(float)
f_vol_surge = vol_ratio * is_down_day (下跌日放量)

波动率特征: vol_20d = close.pct_change().rolling(20).std()
f_low_vol = -vol_20d (低波动=正信号)
```

### 固定权重方案
```python
W_EXTREME_OS = 0.50    # 极端超卖 (核心信号)
W_VOL_SURGE = 0.25     # 放量下跌确认
W_LOW_VOL = 0.25       # 低波动率
```

### 得分计算
```python
score_raw = W_EXTREME_OS * f_extreme_os_rank + W_VOL_SURGE * f_vol_surge_rank + W_LOW_VOL * f_low_vol_rank
score = score_raw.groupby('trade_date').rank(pct=True)
```

---

## V227 回测结果

| 年份 | T+1 IC | IC IR | 年化收益 | 最大回撤 | 状态 |
|------|--------|-------|----------|----------|------|
| 2020 | 0.0363 | 0.29 | 60.13% | -17.92% | ❌ |
| 2022 | 0.0574 | 0.45 | 25.80% | -35.74% | ❌ (IR<0.60) |
| 2024 | 0.0255 | 0.12 | -55.37% | -53.92% | ❌ |
| **平均** | **0.0397** | - | - | - | ❌ |

**关键发现**: 2022 IC=0.0574 首次超过 0.05 阈值！但 IC IR 未达标(0.45 < 0.60)

---

## 数据依赖

### 数据库表

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| `stock_daily` | 日行情数据 | trade_date, symbol, open, high, low, close, pre_close, pct_chg, volume, amount, turnover_rate, industry_code, total_mv, is_st |

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
| **IC 强度不足** | 2020/2024 IC < 0.05 | 无法达到目标阈值 |
| **2024年收益 -55.37%** | 权重在牛市环境中方向错误 | 大幅亏损 |
| **IC IR 全部不达标** | 所有年份 IC IR < 0.60 | V227未通过 |

### 🟡 待优化问题

| 问题 | 描述 | 优先级 |
|------|------|--------|
| **2024年持续失效** | 从V218到V227，2024年IC始终无法达标 | 高 |
| **线性组合性能上限** | 线性方法可能无法捕捉复杂市场模式 | 中 |
| **OHLCV因子信息饱和** | 200+轮迭代后价格/成交量因子可能已达上限 | 高 |

---

## 核心文件清单

### 必须保留（核心代码）

| 文件 | 用途 | 状态 |
|------|------|------|
| `src/alpha_model_v227.py` | V227 Alpha模型（当前版本） | ✅ 当前版本 |
| `src/data_healer.py` | 数据修复模块 | ✅ 保留 |
| `src/backtest_engine.py` | 回测引擎 | ✅ 保留 |
| `src/engine/backtest_referee.py` | 不可变裁判引擎 | ✅ 保留 |
| `run_v227.py` | V227 运行入口 | ✅ 当前版本 |

---

## V227 vs 历史版本对比

### 各版本 IC 对比

| 版本 | 2020 IC | 2022 IC | 2024 IC | Avg IC | 状态 |
|------|---------|---------|---------|--------|------|
| V225R2 (资金流) | 0.0303 | 0.0458 | 0.0243 | 0.0334 | ❌ |
| V226 | 0.0104 | 0.0307 | 0.0017 | 0.0143 | ❌ |
| **V227** | **0.0363** | **0.0574** | **0.0255** | **0.0397** | ❌ |

### 结论
- V227 是当前最佳版本：2022 IC 首次突破 0.05 阈值
- 但 2024 年仍然失效，是系统性盲区
- OHLCV 因子可能已饱和，需要引入另类数据

**目标**：IC ≥ 0.05 且 IC IR ≥ 0.60

*本文件由AI Assistant于2026-04-29更新，反映V227版本状态。*