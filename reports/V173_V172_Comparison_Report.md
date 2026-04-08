# V173 vs V172 性能对比报告

**报告日期**: 2026-04-08  
**版本**: V173 Industrial-Grade Turnover & Stability Enhancement  
**状态**: ✅ 代码实现完成

---

## 1. 执行摘要

### 任务背景
V172 达到了 IC > 0.1 的里程碑，但存在三大工业级隐患：
1. **换手率过高**：潜在冲击成本会吃掉所有利润
2. **代码脚本碎片化**：run_vXXX.py 满天飞
3. **缺乏极端行情下的止损保护**

### V173 硬性目标
| 指标 | V172 基准 | V173 目标 | 状态 |
|------|----------|----------|------|
| Rank IC | 0.1072 | > 0.09 (下降≤5%) | ⏳ 待回测 |
| IC IR | 0.62 | > 0.55 | ⏳ 待回测 |
| Turnover | Baseline | ↓20% | ⏳ 待回测 |
| Max Drawdown | Baseline | ↓15% | ⏳ 待回测 |

---

## 2. V173 核心改进

### 2.1 SignalSmoothingV2 (EMA 平滑)

**核心逻辑**:
```
Score_t = α × Raw_Score_t + (1-α) × Score_{t-1}
```

| 参数 | 值 | 说明 |
|------|-----|------|
| EMA Alpha | 0.3 | 30% 新信号 + 70% 旧信号 |
| 目标换手率降低 | 20% | 通过信号平滑实现 |
| IC 下降容忍度 | ≤5% | 保持核心预测力 |

**预期效果**:
- 降低无效调仓
- 提高信号稳定性
- 减少交易成本

### 2.2 Volatility-Adjusted Position (ATR 动态调仓)

**核心逻辑**:
```python
if ATR_Ratio > 1.5:      # 高波动
    Top_K = 5            # 收缩仓位
elif ATR_Ratio < 0.67:   # 低波动
    Top_K = 15           # 增加分散度
else:                    # 正常波动
    Top_K = 10           # 基础仓位
```

| 参数 | 值 |
|------|-----|
| ATR Window | 5 |
| High Vol Threshold | 1.5 |
| Top_K Base | 10 |
| Top_K Min | 5 |
| Top_K Max | 15 |

**预期效果**:
- 市场剧震时收缩仓位，降低回撤
- 市场平稳时增加分散度，提高收益

### 2.3 TSM vs CSM 差异因子

**核心逻辑**:
```
TSM (Time-Series Momentum) = Close_t / Close_{t-20} - 1
CSM (Cross-Sectional Momentum) = Rank(Momentum) / N_Stocks
Divergence = TSM - (CSM - 0.5) × 2
```

| 参数 | 值 |
|------|-----|
| TSM Window | 20 |
| CSM Window | 20 |

**经济意义**:
- **正值**: 个股强于市场，可能有独立逻辑
- **负值**: 个股弱于市场，可能被错杀
- **捕捉 2025 年风格切换**: 从β行情转向α行情

### 2.4 SQL Healer (数据自愈)

**功能**:
1. 主动检测 pe_ttm/pb 缺失
2. 从 SQL 数据库补全数据
3. 中位数/行业均值填充

**自愈流程**:
```
Missing Columns → SQL Query → Merge & Fill → NaN Repair
```

### 2.5 IC 衰减分析表

**告警机制**:
- T+1 到 T+3 IC 衰减超过 50% → ⚠️ ALERT
- T+1 到 T+5 IC 衰减超过 50% → ⚠️ ALERT

**输出格式**:
```
╔═══════════════════════════════════════════════════════════╗
║              V173 IC DECAY ANALYSIS TABLE                  ║
╠═══════════════════════════════════════════════════════════╣
║  Horizon    IC Value    Decay from T+1    Status          ║
╠═══════════════════════════════════════════════════════════╣
║  T+1        XXXX        baseline          ✓               ║
║  T+3        XXXX        XX.X%             ✓ / ⚠️ ALERT    ║
║  T+5        XXXX        XX.X%             ✓ / ⚠️ ALERT    ║
╚═══════════════════════════════════════════════════════════╝
```

---

## 3. 架构归一化

### 3.1 废弃脚本
以下脚本已被废弃，所有逻辑集成到 `main.py --version 173`:
- ~~run_v173.py~~ ❌
- ~~run_v172.py~~ ⚠️ (保留用于对比)
- ~~run_v171.py~~ ❌
- ...

### 3.2 统一接口
```bash
# 运行 V173 回测
python main.py --version 173 --year 2024

# 运行多年份审计
python main.py --version 173 --all
```

---

## 4. V173 vs V172 因子对比

### 4.1 因子池

| 因子 | V172 | V173 | 变化 |
|------|------|------|------|
| momentum_5 | ✓ | ✓ | 保持 |
| volatility_5 | ✓ | ✓ | 保持 |
| volume_price_contradiction | ✓ | ✓ | 保持 |
| liquidity_alpha | ✓ | ✓ | 保持 |
| reversion_5 | ✓ | ✓ | 保持 |
| **tsm_csm_divergence** | ❌ | ✓ | **新增** |

### 4.2 核心参数对比

| 参数 | V172 | V173 | 变化 |
|------|------|------|------|
| PAC Window | 15 | 15 (自适应) | 保持 |
| Lead-Lag Threshold | 1.3 | 1.3 | 保持 |
| IC Power | 1.0 | 1.0 | 保持 |
| **EMA Alpha** | N/A | **0.3** | **新增** |
| **ATR Window** | N/A | **5** | **新增** |
| **TSM/CSM Window** | N/A | **20** | **新增** |

---

## 5. 预期性能改善

### 5.1 换手率改善

| 指标 | V172 | V173 预期 | 改善 |
|------|------|----------|------|
| 日度信号换手率 | 100% | ~80% | ↓20% |
| 月度调仓频率 | ~20 次 | ~16 次 | ↓20% |
| 冲击成本 | Baseline | -20% | 改善 |

### 5.2 回撤改善

| 场景 | V172 | V173 预期 | 改善 |
|------|------|----------|------|
| 正常市场 | Baseline | Baseline | - |
| 高波动市场 | Baseline | -15% | 改善 |
| 极端行情 | Baseline | -20% | 显著改善 |

### 5.3 IC 稳定性

| 指标 | V172 | V173 预期 | 变化 |
|------|------|----------|------|
| T+1 Rank IC | 0.1072 | 0.095-0.105 | -5% ~ -2% |
| IC IR | 0.62 | 0.55-0.60 | -10% ~ -3% |
| IC 衰减 | 单调 | 单调 | 保持 |

---

## 6. 运行指南

### 6.1 环境准备
```bash
# 确保数据库连接
export DATABASE_URL="mysql+pymysql://user:pass@host:3306/db"

# 或使用 .env 文件
```

### 6.2 运行回测
```bash
# 单一年份
python main.py --version 173 --year 2024

# 多年份
python main.py --version 173 --all

# 指定 Parquet 数据
python main.py --version 173 --parquet data/parquet/stock_data_2024.parquet
```

### 6.3 输出文件
- `reports/v173_audit_2024_*.md` - 审计报告
- `reports/v173_audit_2024_*.json` - JSON 结果
- `reports/v173_reflection_*.json` - 反思报告

---

## 7. 验收标准

| 指标 | 目标值 | 判定标准 |
|------|--------|----------|
| T+1 Rank IC | > 0.09 | 下降不超过 5% |
| IC IR | > 0.55 | 工业级稳定性 |
| Turnover | ↓20% | 换手率改善 |
| IC Decay | 单调 | T+1 > T+3 > T+5 |
| 代码归一化 | main.py 集成 | 废弃 run_vXXX.py |

---

## 8. 风险提示

1. **过拟合风险**: V173 参数在 2024 年数据上优化，需验证 OOS 性能
2. **市场状态依赖**: ATR 动态调仓效果依赖市场波动率 regime
3. **因子拥挤**: volatility_5 权重过高可能带来风险

---

## 9. 后续建议

1. **OOS 验证**: 在 2023 年数据上验证 V173 性能
2. **参数稳定性**: 测试 EMA α在 0.2-0.4 范围内的鲁棒性
3. **实盘准备**: 考虑交易成本和冲击成本的精确建模

---

## 10. 交付清单

- [x] `src/alpha_research_v173.py` - 核心 Alpha 研究模块
- [x] `main.py` - V173Runner 集成
- [x] `reports/V173_V172_Comparison_Report.md` - 本报告
- [ ] `reports/v173_audit_2024_*.md` - 回测报告 (待运行)
- [ ] `reports/V173_Final_Comparison_Report.md` - 最终对比报告 (待回测)

---

**签署**: 资深量化项目负责人 (V173)  
**日期**: 2026-04-08  
**状态**: ✅ 代码实现完成 - 待回测验证