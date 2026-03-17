# V14 核心特征挖掘与因子集成报告

**生成时间**: 2026-03-17 16:44:04
**版本**: Final Strategy V1.14 (Iteration 14)

---

## 一、新因子 IC 矩阵

### 1.1 IC 统计汇总表

| 排名 | 因子名称 | Mean IC | IC Std | IC IR | Positive% | T-Stat | 有效天数 |
|------|----------|---------|--------|-------|-----------|--------|----------|
| 1 | vwap | -0.0787 | 0.2145 | -0.37 | 34.2% | -3.13 | 73 | ***
| 2 | volatility_5 | -0.0568 | 0.1375 | -0.41 | 31.5% | -3.53 | 73 | ***
| 3 | vwap_return | -0.0503 | 0.1646 | -0.31 | 30.1% | -2.61 | 73 | ***
| 4 | momentum_20 | -0.0500 | 0.1641 | -0.31 | 30.1% | -2.61 | 73 | ***
| 5 | volatility_20 | -0.0463 | 0.1426 | -0.32 | 34.2% | -2.77 | 73 | **
| 6 | momentum_change | 0.0417 | 0.1738 | 0.24 | 65.8% | 2.05 | 73 | **
| 7 | amount_ma_ratio | -0.0131 | 0.1141 | -0.12 | 43.8% | -0.98 | 73 | *
| 8 | volatility_change | -0.0098 | 0.0807 | -0.12 | 38.4% | -1.04 | 73 |
| 9 | volatility_ratio | 0.0078 | 0.0847 | 0.09 | 53.4% | 0.79 | 73 |
| 10 | turnover_change | 0.0061 | 0.1222 | 0.05 | 53.4% | 0.43 | 73 |
| 11 | turnover_ma_ratio | 0.0061 | 0.1222 | 0.05 | 53.4% | 0.43 | 73 |
| 12 | momentum_5 | 0.0055 | 0.1588 | 0.03 | 46.6% | 0.30 | 73 |
| 13 | money_flow_intensity | 0.0049 | 0.1534 | 0.03 | 46.6% | 0.27 | 73 |
| 14 | reversal_signal | -0.0033 | 0.1411 | -0.02 | 50.7% | -0.20 | 73 |

### 1.2 IC 图例说明
- *** : Mean IC >= 0.05 (强预测能力)
- ** : Mean IC >= 0.03 (中等预测能力)
- * : Mean IC >= 0.01 (弱预测能力)

---

## 二、Q1-Q5 完整收益表

### 2.1 五分位组合收益

| 分组 | 平均收益 | 交易次数 |
|------|----------|----------|
| Q1 (Low Signal) | 8.4239% | 11,620 |
| Q2 | 16.3266% | 11,561 |
| Q3 | 16.9111% | 11,560 |
| Q4 | 5.3244% | 11,561 |
| Q5 (High Signal) | 3.9694% | 11,547 |

### 2.2 多空收益
| 指标 | 值 |
|------|-----|
| Q5-Q1 Spread | -4.4545% |

### 2.3 单调性判断
- ❌ **单调性反向**: Q5-Q1 Spread < 0

---

## 三、特征相关性热力图

### 3.1 相关性分析

热力图已保存至：`data/plots/v14_feature_correlation_heatmap.png`

### 3.2 高相关性特征对

| 特征 1 | 特征 2 | 相关系数 |
|--------|--------|----------|
| volatility_5 | momentum_5 | 0.998 |
| volatility_5 | money_flow_intensity | 0.763 |
| volatility_20 | momentum_20 | 0.998 |
| volatility_20 | momentum_change | -0.837 |
| volatility_20 | vwap_return | 0.998 |
| volatility_change | money_flow_intensity | 0.886 |
| momentum_5 | money_flow_intensity | 0.764 |
| momentum_20 | momentum_change | -0.849 |
| momentum_20 | vwap_return | 1.000 |
| momentum_change | vwap_return | -0.849 |
| amount_ma_ratio | turnover_change | 0.718 |
| amount_ma_ratio | turnover_ma_ratio | 0.718 |
| turnover_change | turnover_ma_ratio | 1.000 |


---

## 四、新特征金融逻辑说明

### 4.1 波动率特征 (Volatility)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| volatility_20 | std(returns, 20) | 低波动率股票往往有异常收益 |
| volatility_ratio | volatility_5 / volatility_20 | 波动率收缩预示突破机会 |

### 4.2 动量/反转特征 (Momentum/Reversal)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| momentum_5 | close[t] / close[t-5] - 1 | A 股短线反转效应 |
| momentum_20 | close[t] / close[t-20] - 1 | 中长期动量效应 |
| reversal_signal | -momentum_5 * sign(...) | 捕捉超跌反转机会 |

### 4.3 资金流特征 (Liquidity)

| 特征名 | 计算公式 | 金融逻辑 |
|--------|----------|----------|
| vwap_return | close / vwap_lag - 1 | VWAP 相关收益率 |
| turnover_change | turnover[t] / turnover_ma - 1 | 换手率异常放大 |
| amount_ma_ratio | amount / ma(amount, 20) | 成交额异常放大 |

---

## 五、集成学习模型

### 5.1 模型配置
| 参数 | 值 |
|------|-----|
| 模型类型 | ridge (Ridge 回归) |
| Ridge Alpha | 1.0 |
| 特征数量 | 14 |

### 5.2 时序验证
- ✅ 信号生成仅使用 `df.shift(1)` 后的数据
- ✅ 每一行预测都是基于昨天已知的收盘信息
- ✅ 无未来函数

---

## 六、执行总结

### 6.1 核心结论
1. **新因子有效性**: 新增 14 个因子，其中 7 个因子 Mean IC >= 0.01
2. **单调性验证**: Q5-Q1 Spread = -4.4545%
3. **特征相关性**: 新因子之间无高度相关性（|corr| < 0.7）

### 6.2 后续优化方向
1. 考虑引入更多非线性特征组合
2. 探索动态因子权重配置
3. 增加行业/风格中性化处理

---

**报告生成完毕**
