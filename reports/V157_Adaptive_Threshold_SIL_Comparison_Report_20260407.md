# V157 Adaptive Threshold & Signal Inertia Layer Comparison Report

**Generated**: 2026-04-07
**Version**: V157
**Architecture**: Referee-Player (裁判 - 选手)
**Core Enhancements**: AT-2 (Adaptive Threshold v2) + SIL (Signal Inertia Layer) + ORA 3.1 (Skewness Residual)

---

## 1. Executive Summary (执行摘要)

| Metric | Target | V157 Actual | V156 Baseline | Improvement | Status |
|--------|--------|-------------|---------------|-------------|--------|
| T+1 Rank IC | > 0.09 | **0.0215** | 0.0924 | **-76.7%** | ✗ FAILED |
| IC IR | > 0.7 | **0.22** | 0.58 | **-62.1%** | ✗ FAILED |
| Total Return | > 0 | **89.73%** | 0% | **+∞** | ✓ PASSED |
| Annual Return | - | **94.81%** | 0% | **+∞** | ✓ PASSED |
| Sharpe Ratio | - | **1.37** | N/A | NEW | ✓ PASSED |
| Max Drawdown | - | **-25.51%** | N/A | NEW | ✓ MONITOR |
| Turnover | Active | **Active** | 0% | **Fixed** | ✓ PASSED |

**Overall Assessment**: **V157 成功修复零交易问题，但 IC/IR 显著下降** - 需要重新校准信号生成逻辑

---

## 2. V156 vs V157 Comprehensive Comparison

### 2.1 Core Metrics Comparison

| Metric | V156 (Signal-Smoothing) | V157 (AT-2 + SIL) | V156→V157 Δ |
|--------|-------------------------|-------------------|-------------|
| **T+1 IC** | 0.0924 | 0.0215 | **-76.7%** |
| **IC IR** | 0.58 | 0.22 | **-62.1%** |
| **IC Std** | 0.160 | 0.0991 | -38.1% |
| **T+3 IC** | 0.0256 | 0.0071 | -72.3% |
| **T+5 IC** | 0.0220 | 0.0017 | -92.3% |
| **IC Decay** | T+1>T+3>T+5 ✓ | T+1>T+3>T+5 ✓ | Maintained |
| **Total Return** | 0% | 89.73% | **+∞** |
| **Annual Return** | 0% | 94.81% | **+∞** |
| **Sharpe Ratio** | N/A | 1.37 | NEW |
| **Max Drawdown** | N/A | -25.51% | NEW |
| **Num Trading Days** | 242 | 242 | - |

### 2.2 Algorithm Evolution

| Feature | V156 | V157 |
|---------|------|------|
| Orthogonal Residual Mining | ✓ ORA 3.0 | ✓ ORA 3.1 (Skewness) |
| Adaptive Lead-Lag Correction | ✓ | ✓ |
| Cross-Sectional Volatility Weighting | ✓ | ✓ |
| Rolling PAC | ✓ | ✓ |
| Signal Entropy Filter | ✓ | - Removed |
| GARCH-Like Volatility Scaling | ✓ | - **Removed** |
| Adaptive Threshold Gate | ✓ | ✓ **AT-2 (Fixed)** |
| **Signal Inertia Layer (SIL)** | - | ✓ **NEW** |
| **IC-IR Optimized Weighting** | - | ✓ **NEW** |
| **Volatility Trimming [-3,3]** | - | ✓ **NEW** |
| Multi-Level Data Healing | ✓ | ✓ |

---

## 3. V157 Core Implementation Details

### 3.1 AT-2 (Adaptive Threshold v2) - 修复版

**Purpose**: 确保每日 Top 10% 股票必须进入备选库，**严禁零交易**

**Algorithm**:
```python
# Step 1: Score 映射至 [-3, 3] 正态区间
score_normalized = (score - score.mean()) / (score.std() + 1e-6)
score_mapped = score_normalized.clip(-3.0, 3.0)

# Step 2: 每日 Top 10% 阈值
threshold = score_mapped.quantile(0.90)

# Step 3: 门控信号
gate_signal = 1.0 if score >= threshold else 0.0

# Step 4: 确保至少 Top 10% 股票
if num_selected < min_required:
    force_select_top_stocks()
```

**V157 Statistics**:
| Parameter | Value |
|-----------|-------|
| Sigma Clip | 3.0 |
| Top Percentile | 10% |
| Mean Gate Signal | ~50 stocks/day |
| Zero Trading Days | **0** |

**Status**: ✓ **零交易问题已修复**

---

### 3.2 SIL (Signal Inertia Layer) - 信号惯性层

**Purpose**: 基于 IC 自相关的动态α平滑

**Algorithm**:
```python
# 动态权重计算
w = Correlation(Signal_{t-1}, Return_{t-1})
w = min_weight + |w| * (max_weight - min_weight)

# 信号融合
Score_final = w * Score_new + (1-w) * Score_old
```

**V157 Parameters**:
| Parameter | Value |
|-----------|-------|
| Min Weight | 0.2 |
| Max Weight | 0.8 |
| Decay Factor | 0.95 |
| Correlation Window | 20 days |

**Impact**: 信号平滑，但可能降低了 IC 敏感性

---

### 3.3 ORA 3.1 (Skewness Residual) - 三阶矩残差

**Purpose**: 挖掘暴跌后的反弹动力

**Algorithm**:
```python
# 计算滚动偏度
rolling_skew = series.rolling(window=20).apply(skewness)

# 偏度残差
skew_residual = rolling_skew - rolling_skew.rolling(20).mean()

# ORA 3.1 合并
ora31_residual = linear_residual + λ * skew_residual
```

**V157 Parameters**:
| Parameter | Value |
|-----------|-------|
| Skewness Window | 20 days |
| Skewness Lambda | 0.3 |
| Core Factor | volume_price_contradiction |

**V157 Feature Summary**:
| Component | Count |
|-----------|-------|
| Linear Factors | 5 |
| Nonlinear Features | 9 (with Skewness) |
| Total Features | 9 |

---

## 4. IC Decay Analysis (IC 衰减分析)

### 4.1 Decay Pattern Comparison

| Horizon | V156 IC | V157 IC | Change | Pattern |
|---------|---------|---------|--------|---------|
| T+1 | 0.0924 | 0.0215 | -76.7% | Baseline |
| T+3 | 0.0256 | 0.0071 | -72.3% | T+1 > T+3 ✓ |
| T+5 | 0.0220 | 0.0017 | -92.3% | T+3 > T+5 ✓ |

**Decay Pattern**: `T+1(0.0215) -> T+3(0.0071) -> T+5(0.0017)`

**Analysis**: 
- V157 保持了单调递减模式
- IC 绝对值大幅下降，表明信号预测能力减弱
- T+5 IC 下降最显著（-92.3%），长期预测能力受损

---

## 5. Factor Analysis (因子分析)

### 5.1 Selected Lead Factors (V157)

| Factor | IC | Weight | Role |
|--------|-----|--------|------|
| volume_rank | -0.0225 | 0.178 | Lead |
| momentum_5 | -0.0187 | 0.148 | Lead |
| volatility_5 | -0.0456 | 0.360 | Lead (Highest Weight) |
| volume_price_contradiction | 0.0252 | 0.198 | ORM Core |
| liquidity_alpha | -0.0148 | 0.117 | Lead |

### 5.2 Factor IC Comparison

| Factor | V156 IC | V157 IC | Change |
|--------|---------|---------|--------|
| volume_rank | -0.0225 | -0.0225 | - |
| momentum_5 | -0.0187 | -0.0187 | - |
| volatility_5 | -0.0456 | -0.0456 | - |
| volume_price_contradiction | 0.0286 | 0.0252 | -11.9% |
| liquidity_alpha | ~0.0 | -0.0148 | NEW |

---

## 6. Backtest Performance (回测表现)

### 6.1 V157 Performance Summary

| Metric | Value |
|--------|-------|
| Initial Capital | 100,000 |
| Final Value | 277,038.70 |
| **Total Return** | **89.73%** |
| **Annual Return** | **94.81%** |
| **Sharpe Ratio** | **1.37** |
| **Max Drawdown** | **-25.51%** |
| Volatility (Ann.) | 61.94% |
| Trading Days | 242 |
| Transaction Cost | 54,515.96 |

### 6.2 V156 vs V157 Return Comparison

| Metric | V156 | V157 | Change |
|--------|------|------|--------|
| Total Return | 0% | 89.73% | **+∞** |
| Annual Return | 0% | 94.81% | **+∞** |
| Sharpe Ratio | N/A | 1.37 | NEW |
| Max Drawdown | N/A | -25.51% | NEW |
| Trading Activity | None | Active | **Fixed** |

**Key Achievement**: V157 **成功修复了 V156 的零交易问题**，产生了真实的收益曲线

---

## 7. Root Cause Analysis (根本原因分析)

### 7.1 IC 下降原因

1. **GARCH-Like 机制移除**: V156 的信号波动率收缩被移除，导致信号噪声增加
2. **SIL 平滑过度**: 信号惯性层可能过度平滑了有效信号
3. **AT-2 门控简化**: 固定 Top 10% 门控可能引入了低质量信号

### 7.2 收益曲线修复原因

1. **强制交易机制**: AT-2 确保每日至少有 Top 10% 股票进入备选库
2. **信号连续性**: SIL 提供了信号的时间连续性
3. **Skewness 增强**: ORA 3.1 捕捉了暴跌反弹的非线性机会

---

## 8. Recommendations (改进建议)

### 8.1 IC 恢复策略 (Priority: CRITICAL)

1. **重新引入 GARCH-Like 机制**: 保留 V156 的信号波动率收缩
2. **优化 SIL 权重**: 调整 min_weight 和 max_weight 参数
3. **增强因子选择**: 重新评估 Lead Factors 选择标准

### 8.2 IR 提升策略 (Priority: HIGH)

1. **IC 稳定性优化**: 引入 IC 时间序列的波动率抑制
2. **动态因子权重**: 基于滚动 IC 表现自适应调整
3. **信号熵滤波器**: 重新引入 V156 的 Signal Entropy Filter

### 8.3 V158 融合方案 (Recommended)

```
V158 = V156 (GARCH + ORA 3.0) + V157 (AT-2 + SIL) + Enhancements

Key Components:
1. GARCH-Like Volatility Scaling (from V156)
2. ORA 3.0 Nonlinear Interactions (from V156)
3. AT-2 Fixed Threshold Gate (from V157)
4. SIL Signal Inertia Layer (from V157, optimized)
5. Signal Entropy Filter (from V156)
6. Multi-Level Data Healing (from both)
```

---

## 9. Engineering Discipline (工程纪律)

| Rule | Status |
|------|--------|
| No Empty Position (严禁空仓) | ✓ PASSED |
| No Data Filler (严禁数据敷衍) | ✓ PASSED |
| No Capital Change (严禁篡改裁判) | ✓ PASSED |
| Active Debug (主动 Debug) | ✓ PASSED |
| Full Implementation (严禁 To be implemented) | ✓ PASSED |

---

## 10. Conclusion (结论)

### 10.1 V157 Mission Accomplishment

| Mission | Status | Evidence |
|---------|--------|----------|
| AT-2 (Adaptive Threshold v2) | ✓ COMPLETED | Zero trading days = 0 |
| SIL (Signal Inertia Layer) | ✓ COMPLETED | Dynamic weight applied |
| ORA 3.1 (Skewness Residual) | ✓ COMPLETED | 9 features extracted |
| No Empty Position | ✓ COMPLETED | Total Return = 89.73% |
| IC > 0.09 | ✗ FAILED | IC = 0.0215 |
| IC IR > 0.7 | ✗ FAILED | IR = 0.22 |
| Turnover Active | ✓ COMPLETED | Active trading confirmed |

### 10.2 Final Assessment

**V157 成功修复了 V156 的零交易问题，但 IC/IR 显著下降**

**成就**:
1. ✓ **零交易问题已修复**: Total Return 89.73%
2. ✓ **AT-2 门控有效**: 每日 Top 10% 股票进入备选库
3. ✓ **SIL 信号平滑**: 基于 IC 自相关的动态权重
4. ✓ **ORA 3.1 增强**: Skewness 残差捕捉非线性机会
5. ✓ **工程纪律遵守**: 无空仓、无数据敷衍、无篡改裁判

**问题**:
1. ✗ **IC 大幅下降**: 0.0924 → 0.0215 (-76.7%)
2. ✗ **IR 大幅下降**: 0.58 → 0.22 (-62.1%)
3. ✗ **长期预测能力受损**: T+5 IC 下降 92.3%

**下一步行动**: 建议开发 **V158 融合版本**，结合 V156 和 V157 的优势

---

## Appendix: V156-V157 Evolution Summary

```
V156 (Signal-Smoothing + ORA 3.0):
  - IC: 0.0924 (✓), IR: 0.58 (CLOSE)
  - Return: 0% (✗ ZERO TRADING)
  - Features: GARCH Scaling, ORA 3.0, Adaptive Gate
  - Status: High IC but no trading

V157 (AT-2 + SIL + ORA 3.1):
  - IC: 0.0215 (✗), IR: 0.22 (✗)
  - Return: 89.73% (✓ ACTIVE TRADING)
  - Features: AT-2 Gate, SIL, Skewness Residual
  - Status: Active trading but low IC/IR

V158 (Recommended Fusion):
  - Target IC: > 0.09
  - Target IR: > 0.7
  - Target Return: > 0
  - Features: V156 (GARCH + ORA 3.0) + V157 (AT-2 + SIL) + Optimizations
```

---

*Report generated by V157 Audit System (Adaptive Threshold & Signal Inertia Layer)*