# Current State - 2026-05-01

## Latest Version: V235 (Value + Reversal) - FINAL

### Strategy
- **Value Factor**: Price-to-MA250 ratio (proxy for 1/PB)
- **Reversal Factor**: Negative 5-day return
- **Combined**: Value(0.5) + Reversal(0.5)
- **Portfolio**: Top 50 stocks, equal-weighted

### Results
| Year | Portfolio | Benchmark | Excess | IR |
|------|-----------|-----------|--------|-----|
| 2020 | -34.73% | 39.64% | -74.37% | -2.35 |
| 2022 | -18.87% | -23.29% | +4.42% | 0.13 |
| 2024 | -5.72% | 16.52% | -22.24% | -0.45 |

### Factor IC
| Year | Mean IC | IC IR |
|------|---------|-------|
| 2020 | 0.0301 | 0.23 |
| 2022 | 0.0429 | 0.31 |
| 2024 | 0.0318 | 0.16 |

### Conclusion
V235转向基本面因子（价值+反转），放弃纯OHLCV因子挖掘。**所有年份FAIL**。

#### 230+轮迭代总结

经过234轮迭代（V1-V235），A股日频因子策略已达到信息上限：

1. **OHLCV因子极限**: 最佳Avg IC ~0.04 (V227/V225R2)，无法突破0.05
2. **2024年系统性失效**: 无论何种因子组合，2024年牛市环境中年化收益均为负
3. **线性组合饱和**: 2-3个因子的线性组合已无法提供更多alpha
4. **LightGBM过拟合**: 非线性模型在截面预测中表现更差

#### 建议方向

1. **停止A股日频策略迭代**：230+轮已证明OHLCV无法产生稳定超额
2. **转向另类数据**：新闻情绪、分析师预期、社交媒体情绪
3. **考虑周频/月频策略**：降低频率可能减少噪声
4. **探索微观结构**：订单流、买卖价差、盘口深度（需要tick数据）

### Project Status: ⏸️ PAUSED - Need New Data Sources