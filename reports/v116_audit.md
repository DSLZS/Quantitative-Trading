# V116 Alpha Audit Report - 真实环境下的因子进化

**Generated**: 2026-04-01
**Version**: V116 真实环境下的因子进化
**Architecture**: Referee-Player (裁判 - 选手)

---

## 1. Executive Summary (执行摘要)

| Metric | Target | V115 Actual | V116 Target | Status |
|--------|--------|-------------|-------------|--------|
| T+1 Rank IC | > 0.03 | -0.0078 | > 0.03 | V115 FAILED |
| IC IR | > 0.4 | N/A | > 0.4 | V115 FAILED |
| Data Source | Real | Mock | Real | V115 VIOLATION |
| Fitness Function | IC+ICIR-|Skew| Simple IC | IC+ICIR-|Skew| | V116 ENHANCED |
| Auto-Flip | Required | No | Yes | V116 ADDED |
| Regime Aware | Required | Basic | Enhanced | V116 ENHANCED |

**V115 Overall Assessment**: **FAILED ✗** - Negative IC + Mock Data Violation

---

## 2. V115 失败根因分析

### 2.1 负 IC 诊断 (-0.0078)

V115 的 T+1 Rank IC 为 **-0.0078**，这是一个彻底的失败。根因分析如下：

#### 原因 1: 遗传算法过拟合 Mock 数据
```
V115 代码分析:
- DataHealingEngineV115 默认生成 Mock 数据
- generate_mock_data() 使用纯随机游走生成价格
- t1_return = np.random.normal(0, 0.02) 完全随机
- 遗传算法在随机噪声上"学习"出了虚假模式
```

#### 原因 2: 适应度函数设计缺陷
```python
# V115 适应度函数 (缺陷)
def evaluate_factor(self, factor, df):
    # 仅使用 IC 绝对值
    factor.ic_score = self.evaluate_factor(...)  # 只返回 IC
    # 没有考虑 IC 稳定性 (ICIR)
    # 没有考虑因子偏度 (Skewness)
```

**问题**: 单纯追求 IC 强度会导致:
1. 过拟合短期噪声
2. 忽略因子分布的偏度风险
3. 无法区分真实信号和随机巧合

#### 原因 3: 缺少方向纠偏机制
- V115 没有 Auto-Flip 逻辑
- 当因子 IC 稳定为负时，无法自动纠正
- 导致"反向因子"污染最终评分

### 2.2 Mock 数据工程违规

V115 的 `DataHealingEngineV115` 存在严重的工程违规:

```python
# V115 违规代码
def generate_mock_data(self, n_stocks=100, n_days=60):
    # 生成完全随机的价格和收益
    ret = np.random.normal(0, 0.02)
    t1_return = np.random.normal(0, 0.02)  # 纯随机
```

**违规性质**:
1. 在正式审计报告中使用了伪造数据
2. 通过 `--mock-data` 参数逃避真实数据环境问题
3. 生成的 Mock 数据不具备真实 A 股的统计特性

---

## 3. V116 改进方案

### 3.1 适应度函数重构

V116 采用综合适应度函数:

```python
# V116 适应度函数 (增强)
def evaluate_factor(self, factor, df):
    ic_score = float(np.mean(ic_scores))
    icir_score = self._calculate_icir(ic_scores)  # IC 稳定性
    skewness = self._calculate_skewness(factor_values)  # 偏度风险
    
    # 综合适应度：IC + ICIR - |Skewness|
    fitness_score = ic_score + icir_score - abs(skewness)
    
    return ic_score, icir_score, skewness, fitness_score
```

**改进效果**:
| 组件 | 作用 | 权重 |
|------|------|------|
| IC | 预测强度 | +1.0 |
| ICIR | 稳定性惩罚 | +1.0 |
| |Skewness| | 偏度风险惩罚 | -1.0 |

### 3.2 Auto-Flip 方向纠偏

V116 引入自动方向纠正机制:

```python
def auto_flip_direction(self, factor, df, threshold_days=5):
    """
    如果一个强特征的 IC 稳定为负，自动对其取反 (-1 * Factor)
    """
    negative_ic_days = 0
    total_valid_days = 0
    
    for date in unique_dates[-threshold_days:]:
        ic = self._calculate_ic(factor_day, label_day)
        if ic < -0.01:  # 显著负相关
            negative_ic_days += 1
    
    # 如果大部分天数为负 IC，触发翻转
    if negative_ic_days / total_valid_days >= 0.6:
        return flipped_factor  # -1 * original
```

**触发条件**:
- 最近 5 天中 ≥60% 的天数 IC < -0.01
- 自动创建翻转因子：`-1 * (original_expression)`

### 3.3 算子扩充 (捕捉机构行为)

V116 新增 6 个时序算子:

| 算子 |  arity | 窗口 | 描述 | 机构行为捕捉 |
|------|-------|------|------|-------------|
| Ts_Correlation | 3 | 10/20/30 | 时序相关性 | 量价关系分析 |
| Ts_Regression_Slope | 2 | 10/20/30 | 回归斜率 | 趋势强度识别 |
| Ts_Covariance | 3 | 10/20/30 | 时序协方差 | 资产联动分析 |
| Ts_Rank | 2 | 10/20/30 | 时序排名 | 相对位置判断 |
| WMA | 2 | 5/10/20 | 加权移动平均 | 近期权重倾斜 |
| EMA | 2 | 5/10/20 | 指数移动平均 | 平滑趋势跟踪 |

**示例因子**:
```
Ts_Correlation(close, volume, 20)  # 量价相关性
Ts_Regression_Slope(close, 20)     # 价格趋势斜率
```

### 3.4 场景感知增强 (Regime Awareness)

V116 完善 MarketRegimeDetector:

```python
# 场景策略映射
MEAN_REVERSION_FACTORS = ['reversion_5', 'volatility_20', 'liquidity_stress_5']
TREND_FOLLOWING_FACTORS = ['momentum_10', 'order_flow_imbalance_5']

def get_dynamic_weights(self, factor_names, df):
    regime = self.detect_regime(df)
    
    if regime == 'high_vol_small_cap':
        # 高波动小盘：强制启用均值回归
        for f in factor_names:
            if f in MEAN_REVERSION_FACTORS:
                weights[f] = 2.0  # 权重加倍
            elif f in TREND_FOLLOWING_FACTORS:
                weights[f] = 0.5  # 权重减半
    
    elif regime == 'low_vol_large_cap':
        # 低波动大盘：启用趋势跟踪
        for f in factor_names:
            if f in TREND_FOLLOWING_FACTORS:
                weights[f] = 2.0
            elif f in MEAN_REVERSION_FACTORS:
                weights[f] = 0.5
```

**四象限场景**:
| 场景 | 波动率 | 市值 | 启用策略 |
|------|--------|------|---------|
| high_vol_small_cap | 高 | 小盘 | 均值回归 ×2 |
| low_vol_large_cap | 低 | 大盘 | 趋势跟踪 ×2 |
| high_vol_large_cap | 高 | 大盘 | 均衡配置 |
| low_vol_small_cap | 低 | 小盘 | 均衡配置 |

### 3.5 真实数据强制 (禁用 Mock)

V116 严格禁用 Mock 数据:

```python
class RealDataLoader:
    """真实数据加载器 - 禁用 Mock 数据"""
    
    def load_data(self, start_date, end_date):
        # 1. 尝试从数据库加载
        if self.engine:
            df = self._load_from_database(...)
            if df is not None and not df.empty:
                return df
        
        # 2. 尝试从 Parquet 文件加载
        parquet_files = list(Path("data/parquet").glob("*.parquet"))
        if parquet_files:
            df = pd.concat([pd.read_parquet(pf) for pf in parquet_files])
            return df
        
        # 3. 数据不可用 - 抛出错误而非生成 Mock
        raise DataHealingError(
            "No real data available. Please configure DATABASE_URL or add Parquet files."
        )
```

**强制策略**:
- 数据库连接失败 → 尝试 Parquet
- Parquet 也不可用 → 抛出 `DataHealingError`
- **严禁**生成 Mock 数据用于正式审计

---

## 4. V116 代码架构

### 4.1 核心组件

```
src/alpha_research_v116.py
├── SymbolicOperatorLibrary    # 算子库 (含 6 个新算子)
├── GeneticFactorMiner         # 遗传因子挖掘器
│   ├── evaluate_factor()      # IC + ICIR - |Skewness|
│   ├── auto_flip_direction()  # Auto-Flip 逻辑
│   └── apply_operator()       # 算子实现
├── MarketRegimeDetector       # 场景感知检测器
│   ├── detect_regime()        # 四象限检测
│   └── get_dynamic_weights()  # 场景感知权重
├── RealDataLoader             # 真实数据加载器
│   └── load_data()            # 数据库 → Parquet → Error
└── AlphaResearchV116          # 主引擎
    └── compute_score()        # 统一评分接口
```

### 4.2 与 V115 对比

| 组件 | V115 | V116 | 改进 |
|------|------|------|------|
| 数据源 | Mock | Real | 强制真实数据 |
| 适应度 | IC only | IC+ICIR-|Skew| | 综合评估 |
| 方向纠偏 | 无 | Auto-Flip | 自动纠正 |
| 算子库 | 基础 | +6 个新算子 | 机构行为捕捉 |
| 场景感知 | 基础检测 | 策略映射 | 均值回归/趋势跟踪 |

---

## 5. 运行说明

### 5.1 环境配置

```bash
# 1. 配置数据库 (必需)
echo "DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/quant" > .env

# 2. 安装依赖
pip install pandas numpy sqlalchemy pymysql loguru pyyaml python-dotenv

# 3. 准备 Parquet 数据 (可选备选)
# 将数据文件放入 data/parquet/ 目录
```

### 5.2 运行回测

```bash
# 通过 main.py 运行 (统一入口)
python main.py --year 2024 --version 116

# 或多年份
python main.py --all --version 116
```

### 5.3 预期输出

```
[V116][AlphaResearch] V116 Alpha Research Engine Initialized
[V116][GeneticFactorMiner] Initialized
[V116][MarketRegimeDetector] Initialized
[V116][RealDataLoader] Database connected successfully
[V116][Auto-Flip] Flipping factor: Ts_Correlation(close, volume, 20) (negative IC for 4/5 days)
[V116][RegimeAware] High Vol/Small Cap detected - enabling mean reversion factors
```

---

## 6. 验收标准

| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.03 | 核心预测能力指标 |
| IC IR | > 0.4 | 稳定性指标 |
| IC Decay | 单调递减 | 无前视偏差 |
| Data Source | Real | 严禁 Mock 数据 |
| Auto-Flip | Enabled | 方向纠偏有效 |
| Regime Aware | Enabled | 场景感知有效 |

---

## 7. 线性相关性瓶颈分析

如果 V116 运行后 IC 仍无法达到 0.03，需要进行以下分析:

### 7.1 因子线性相关性检测

```python
# 计算因子间相关系数矩阵
factor_corr = df[factor_columns].corr()

# 识别高相关因子对
high_corr_pairs = []
for i, f1 in enumerate(factor_columns):
    for f2 in factor_columns[i+1:]:
        if abs(factor_corr.loc[f1, f2]) > 0.7:
            high_corr_pairs.append((f1, f2, factor_corr.loc[f1, f2]))
```

### 7.2 瓶颈诊断

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| IC < 0.01 | 因子同质化严重 | 增加正交化步骤 |
| ICIR < 0.3 | 因子不稳定 | 延长回测周期 |
| 高相关对 > 10 | 信息冗余 | 特征选择/降维 |

### 7.3 禁止美化结果

**严禁**以下行为:
- ❌ 修改回测时间段来美化 IC
- ❌ 剔除表现差的日期
- ❌ 使用未来函数

**必须**如实汇报:
- ✓ 因子库的线性相关性瓶颈
- ✓ 真实数据下的预测能力上限
- ✓ 需要进一步优化的方向

---

## 8. 结论

### V115 失败总结

1. **负 IC (-0.0078)**: 遗传算法在 Mock 数据上过拟合
2. **Mock 数据违规**: 通过 `--mock-data` 逃避真实数据环境
3. **适应度缺陷**: 仅优化 IC 强度，忽略稳定性和偏度
4. **缺少纠偏**: 无 Auto-Flip 机制

### V116 改进承诺

1. **综合适应度**: IC + ICIR - |Skewness|
2. **Auto-Flip**: 自动纠正反向因子
3. **算子扩充**: Ts_Correlation, Ts_Regression_Slope 等
4. **场景感知**: 高波动小盘→均值回归，低波动大盘→趋势跟踪
5. **真实数据**: 严禁 Mock，数据库/Parquet 双保险

### 下一步行动

1. 配置 DATABASE_URL 或准备 Parquet 数据
2. 运行 `python main.py --year 2024 --version 116`
3. 根据实际 IC 结果进行因子库优化

---

*Report generated by V116 Alpha Research Engine (Real Data Environment)*