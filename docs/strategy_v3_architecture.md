# Final Strategy V3.0 - 架构重构文档

## 概述

本文档详细说明了从 V2.6（规则驱动）到 V3.0（模型驱动）的核心重构逻辑，解释了新架构如何提供更强的**泛化预测能力**。

---

## 一、架构对比

### V2.6（旧版）- 规则驱动架构

```
原始因子 → 硬编码规则 → 交易信号
         - if atr_rank > 0.7: multiply by 1.3
         - if hold_days > 10: sell
         - if loss > 5%: stop_loss
```

**问题**：
1. 因子间存在多重共线性，信息冗余
2. 规则是人工设定的，无法适应市场变化
3. 预测目标是简单收益率，未考虑风险
4. 对异常值敏感，极端行情下表现差

### V3.0（新版）- 模型驱动架构

```
原始因子 → 正交化 → 特征合成 → GMM 门控 → 集成模型 → 交易信号
           ↓           ↓           ↓          ↓
        去相关     非线性特征   状态感知    Ridge+LGBM
```

**优势**：
1. 因子相互独立，减少噪声
2. 自动学习市场状态，动态调整权重
3. 三屏障碍法标注，学习风险调整后收益
4. Huber Loss 对异常值鲁棒

---

## 二、核心算法详解

### 1. 因子正交化 (Factor Orthogonalization)

**数学原理** - Gram-Schmidt 正交化：

对于因子矩阵 X 的每一列 x_i：
1. 计算 x_i 在之前所有正交化向量上的投影
2. 从 x_i 中减去这些投影，得到正交分量

```python
for j in range(n_features):
    v = X_normalized[:, j].copy()
    for i in range(j):
        R[i, j] = np.dot(Q[:, i], X_normalized[:, j])
        v = v - R[i, j] * Q[:, i]  # 减去投影
    norm = np.linalg.norm(v)
    if norm > tolerance:
        Q[:, j] = v / norm  # 归一化
```

**金融意义**：
- 动量因子和波动率因子往往存在负相关
- 正交化后，每个因子提供**独立信息**
- 减少模型过拟合风险

**示例**：
```
原始：momentum_5 与 volatility_20 相关系数 = -0.65
正交化后：momentum_5_ortho 与 volatility_20_ortho 相关系数 ≈ 0
```

---

### 2. 非线性特征合成 (Feature Synthesis)

**核心逻辑**：

1. 生成二阶交叉特征：`x_i * x_j`
2. 计算每个特征与目标的互信息 (Mutual Information)
3. 选择 Top N 个高 MI 特征

**互信息计算**：
```python
def _compute_mutual_information(self, x, y):
    x_discrete = self._discretize(x, n_bins=10)
    y_discrete = self._discretize(y, n_bins=10)
    mi = mutual_info_score(x_discrete, y_discrete)
    return mi
```

**金融意义**：
- `动量 × 流动性` = 流动性调整的动量效应
- `波动率 / 成交量` = 单位成交量的价格波动
- 捕捉因子间的**非线性交互效应**

---

### 3. 智能环境感知门控 (Gating Mechanism)

**GMM 聚类原理**：

高斯混合模型假设数据来自多个高斯分布的混合：
```
P(x) = Σ_k π_k * N(x | μ_k, Σ_k)
```

**市场状态识别**：
```python
self.gmm = GaussianMixture(
    n_components=4,  # 4 种状态
    covariance_type='full',
    n_init=5,
)
```

**状态相关权重学习**：
```python
for state_id in range(n_components):
    mask = labels == state_id
    X_state = X[mask]
    y_state = y[mask]
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_state, y_state)
    self.state_weights[state_id] = ridge.coef_
```

**4 种隐状态**：
| 状态 ID | 名称 | 特征 | 有效因子类型 |
|--------|------|------|-------------|
| 0 | CALM | 低波动、平稳 | 所有因子 |
| 1 | BULL | 上涨趋势 | 动量因子 |
| 2 | BEAR | 下跌趋势 | 防御因子 |
| 3 | VOLATILE | 高波动 | 波动率因子 |

---

### 4. 三屏障碍法标注 (Triple Barrier Method)

**核心逻辑**：

1. **上屏障**：价格触及 `entry_price * (1 + 2σ)` → 标签=2
2. **下屏障**：价格触及 `entry_price * (1 - 2σ)` → 标签=0
3. **时间屏障**：持有 N 天后退出 → 标签=1

```python
for t in range(min(len(prices[i]), self.time_barrier)):
    if prices[i, t] >= upper:
        labels[i] = 2  # 上屏障击中
        break
    elif prices[i, t] <= lower:
        labels[i] = 0  # 下屏障击中
        break
    elif t == self.time_barrier - 1:
        labels[i] = 1  # 时间到期
```

**金融意义**：
- 模型学习的是"**在风险调整后能否获利**"
- 而非简单的"明天涨还是跌"
- 自动考虑止损和止盈

---

### 5. 感知回撤的损失函数 (Huber Drawdown Loss)

**Huber Loss 公式**：
```
L(r, r̂) = { 0.5 * (r - r̂)²           if |r - r̂| ≤ δ
         { δ * (|r - r̂| - 0.5 * δ)   otherwise
```

**回撤惩罚项**：
```
L_total = L_Huber + λ * max(0, -r̂)²
```

**代码实现**：
```python
def __call__(self, y_true, y_pred):
    residual = y_true - y_pred
    abs_residual = np.abs(residual)
    huber_loss = np.where(
        abs_residual <= self.delta,
        0.5 * residual ** 2,
        self.delta * (abs_residual - 0.5 * self.delta)
    )
    drawdown_penalty = np.maximum(0, -y_pred) ** 2
    return np.mean(huber_loss) + self.drawdown_weight * np.mean(drawdown_penalty)
```

**优势**：
- 对异常值（极端行情）更鲁棒
- 鼓励模型避免预测极端负收益

---

## 三、为什么 V3.0 具有更强的泛化能力？

### 1. 去除了人工规则的过拟合

| V2.6 规则 | V3.0 替代方案 |
|----------|--------------|
| `if atr_rank > 0.7: weight *= 1.3` | GMM 自动学习状态相关权重 |
| `hold_days = 10` | 三屏障碍法动态退出 |
| `stop_loss = -5%` | Huber Loss 自动处理异常值 |

### 2. 特征工程的质变

| 维度 | V2.6 | V3.0 |
|------|------|------|
| 特征独立性 | ❌ 因子相关 | ✅ 正交化 |
| 非线性 | ❌ 仅线性 | ✅ 二阶交叉 |
| 特征选择 | ❌ 人工设定 | ✅ 互信息排序 |

### 3. 预测目标的升级

| 方面 | V2.6 | V3.0 |
|------|------|------|
| 目标 | T+1 收益率 | 三屏障碍法标注 |
| 风险考虑 | ❌ 无 | ✅ 波动率调整 |
| 退出逻辑 | ❌ 固定天数 | ✅ 路径依赖 |

### 4. 损失函数的鲁棒性

| 特性 | V2.6 (MSE) | V3.0 (Huber) |
|------|-----------|-------------|
| 异常值敏感 | ❌ 敏感 | ✅ 鲁棒 |
| 回撤惩罚 | ❌ 无 | ✅ 有 |

---

## 四、使用示例

### 训练模型

```python
from final_strategy_v3_0 import FinalStrategyV30

strategy = FinalStrategyV30(
    config_path="config/production_params.yaml",
    factors_config_path="config/factors.yaml",
)

# 训练模型（使用 2022-2023 数据）
strategy.train_model(train_end_date="2023-12-31")
```

### 运行回测

```python
result = strategy.run_backtest(
    start_date="2024-01-01",
    end_date="2024-06-30",
    initial_capital=1000000.0,
)

print(f"总收益率：{result.total_return:.2%}")
print(f"夏普比率：{result.sharpe_ratio:.2f}")
print(f"最大回撤：{result.max_drawdown:.2%}")
```

### 每日预测

```python
# 获取当日数据
daily_data = db.read_sql("SELECT * FROM stock_daily WHERE trade_date = '2024-03-15'")

# 计算因子
daily_data = factor_engine.compute_factors(daily_data)

# 模型预测
daily_data = strategy.predict(daily_data)

# 获取预测分数
top_stocks = daily_data.sort("predict_score", descending=True).head(10)
```

---

## 五、配置参数说明

### 正交化配置
```python
ORTHOGONALIZATION_ENABLED = True
GRAM_SCHMIDT_TOLERANCE = 1e-10  # 正交化容差
```

### 特征合成配置
```python
FEATURE_SYNTHESIS_ENABLED = True
MAX_INTERACTION_ORDER = 2  # 二阶交叉
TOP_N_FEATURES_BY_MI = 30  # 保留 Top 30 特征
```

### 环境门控配置
```python
GATING_MECHANISM_ENABLED = True
GMM_N_COMPONENTS = 4  # 4 种市场状态
```

### 三屏障碍法配置
```python
TBR_UPPER_BARRIER = 2.0   # 上屏障：2 倍波动率
TBR_LOWER_BARRIER = -2.0  # 下屏障：-2 倍波动率
TBR_TIME_BARRIER = 5      # 时间屏障：5 天
```

### 损失函数配置
```python
HUBER_DELTA = 1.0         # Huber Loss 的 delta
DRAWDOWN_PENALTY_WEIGHT = 0.1  # 回撤惩罚权重
```

---

## 六、性能对比（预期）

| 指标 | V2.6 | V3.0（预期） |
|------|------|-------------|
| 年化收益 | ~15% | ~20%+ |
| 夏普比率 | ~1.0 | ~1.5+ |
| 最大回撤 | ~25% | ~15% |
| 胜率 | ~55% | ~60%+ |
| 平均持有天数 | 10 | 3-7（动态） |

---

## 七、总结

V3.0 的核心创新在于：

1. **从规则到学习**：不再依赖人工设定的阈值和规则，而是让模型从数据中学习
2. **从线性到非线性**：通过特征合成捕捉因子间的交互效应
3. **从静态到动态**：GMM 门控使模型能够根据市场状态自动调整
4. **从价格到风险**：三屏障碍法让模型学习风险调整后的收益

这种架构设计使得 V3.0 在**未见过的市场环境下**（如极端行情、风格切换）具有更强的适应能力和泛化能力。