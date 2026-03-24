# V67 量化交易系统 - 最终交付报告

## 报告信息

- **系统版本**: V67.0
- **交付日期**: 2026-03-24
- **系统类型**: SPI 信号质量审计驱动的量化交易系统

---

## 一、交付清单

| 文件名 | 类型 | 功能描述 |
|--------|------|----------|
| `src/v67_data_filler.py` | 核心模块 | 数据暴力填充工具 |
| `src/v67_core.py` | 核心模块 | 带 SPI 审计的预测引擎 |
| `src/v67_engine.py` | 核心模块 | 回测引擎（拒绝降级） |
| `src/run_v67_backtest.py` | 运行脚本 | 一键运行回测 |

---

## 二、核心功能实现

### 2.1 数据强制填充协议 (v67_data_filler.py)

#### ✅ 暴力抓取协议
- [x] **严禁一次性抓取全部个股** - 采用单只股票逐个抓取模式
- [x] **批次 - 等待模式** - 每抓取 20 只股票，强制 sleep 5 秒
- [x] **落库验证** - 每写入一条数据，立即执行 SELECT COUNT(*) 确认
- [x] **连续失败退出** - 连续 3 次写入失败，sys.exit(1) 并打印具体 API 报错信息

#### ✅ 报错透明化分类
```python
错误类型分类:
- "timeout"      → 网络超时，请检查网络连接
- "no_data"      → API 返回空数据，可能是非交易日
- "ip_blocked"   → IP 可能被封禁，请更换 IP 或使用代理
- "connection_error" → 网络连接错误
- "unknown"      → 未知错误，请检查日志详情
```

#### ✅ 必须拉取的数据
- [x] 2024 年全年的 stock_individual_fund_flow_rank_em（资金流排名表）
- [x] 2024 年全年的 stock_board_industry_cons_em（行业成分）

#### ✅ 输出格式
```
[SUCCESS] 000001.SZ written. Total rows now: XXX
```

---

### 2.2 核心算法：SPI 信号纯度驱动 (v67_core.py)

#### ✅ 预测引擎
| 功能 | 实现状态 | 说明 |
|------|----------|------|
| 5 日远期收益率 | ✅ 已实现 | `forward_return_5d = (close.shift(-5) - close) / close` |
| IC 值计算 | ✅ 已实现 | `IC = Corr(Signal, Forward Return)` |
| SPI 约束 | ✅ 已实现 | `SPI = IC / IC_Std`，如果 SPI < 0.1，重新调整 RS-ZScore 权重 |

#### ✅ SPI 自适应权重调整
```python
if spi_value < 0.05:
    rs_z_score_weight = 2.0  # 大幅增加权重
elif spi_value < 0.1:
    rs_z_score_weight = 1.5  # 适度增加权重
```

#### ✅ 机构踪迹逻辑
| 因子 | 公式 | 阈值 |
|------|------|------|
| 主力净额/流通市值 | `main_force_ratio = net_main_amount / mv` | > 0.1% |
| 主力净额/成交量 | `main_force_volume_ratio = |net_main_amount| / volume` | 比值上升 |

---

### 2.3 杜绝偷懒与伪造 (审计条款)

#### ✅ 数据熔断升级
```python
检查项目                    阈值           实际值
stock_fund_flow 行数    ≥ 1,000,000    待填充
stock_industry_daily 行数 ≥ 100,000     待填充
```

**熔断触发时输出**:
```
数据不足，请运行 data_filler
详细原因:
  - 数据不足：stock_fund_flow 仅有 XXX 行，需要 1,000,000 行
  - 数据不足：stock_industry_daily 仅有 XXX 行，需要 100,000 行
```

#### ✅ 手续费锁定
```python
# V67 费率配置 - 总计 0.2% (写死，严禁修改)
V67_COMMISSION_RATE = 0.0003    # 佣金万 3
V67_MIN_COMMISSION = 5.0        # 最低佣金 5 元
V67_SLIPPAGE_BUY = 0.001        # 买入滑点 0.1%
V67_SLIPPAGE_SELL = 0.001       # 卖出滑点 0.1%
V67_STAMP_DUTY = 0.0005         # 印花税 0.05%
V67_TRANSFER_FEE = 0.00001      # 过户费 0.001%
V67_FRICTION_COST = 0.002       # 0.2% 总计 (写死)
```

#### ✅ 报错透明化
```python
# 不准用 fill_null(0) 掩盖
def log_missing_data(self, symbol: str, trade_date: str, field_name: str):
    missing_info = f"缺失数据：symbol={symbol}, trade_date={trade_date}, field={field_name}"
    self._missing_data_log.append(missing_info)
    logger.warning(f"V67: {missing_info}")
```

---

## 三、使用方法

### 3.1 运行数据填充
```bash
# 填充 2024 年全年数据
python src/run_v67_backtest.py --fill-data
```

### 3.2 验证数据充足性
```bash
# 只验证数据，不运行回测
python src/run_v67_backtest.py --verify-only
```

### 3.3 运行回测
```bash
# 运行完整回测（2024 年全年）
python src/run_v67_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
```

### 3.4 完整流程
```bash
# 一键完成：填充数据 + 验证 + 回测
python src/run_v67_backtest.py --fill-data --start-date 2024-01-01 --end-date 2024-12-31
```

---

## 四、SPI 审计指标

### 4.1 SPI 阈值配置
| 指标 | 阈值 | 说明 |
|------|------|------|
| SPI_TARGET | 0.1 | 目标 SPI 值 |
| SPI_MIN | 0.05 | 最低容忍 SPI 值 |

### 4.2 IC 审计目标
| 指标 | 目标值 | 说明 |
|------|--------|------|
| IC_TARGET_MEAN | 0.02 | 2024 年 IC 均值目标 > 0.02 |

### 4.3 审计报告输出
```
============================================================
V67 SPI 信号质量审计表
============================================================
统计天数：250
----------------------------------------
Mean SPI: 0.1234 (目标：>0.1)
Min SPI:  0.0567 (最低容忍：0.05)
Max SPI:  0.2345
----------------------------------------
✓ SPI 达标：Mean SPI (0.1234) >= 目标 (0.1)
============================================================
```

---

## 五、回测报告指标

### 5.1 核心指标
| 指标 | 说明 |
|------|------|
| 总收益率 | 回测期间总收益率 |
| 年化收益率 | 年化收益率 |
| 最大回撤 | 最大回撤百分比 |
| 夏普比率 | 风险调整后收益 |
| AE 指标 | Alpha-Efficiency 指标 |

### 5.2 交易统计
| 指标 | 说明 |
|------|------|
| 交易次数 | 总交易次数 |
| 胜率 | 盈利交易占比 |
| 盈亏比 | 平均盈利/平均亏损 |

### 5.3 SPI 审计
| 指标 | 说明 |
|------|------|
| Mean SPI | 平均 SPI 值 |
| Min SPI | 最小 SPI 值 |
| Max SPI | 最大 SPI 值 |
| SPI Pass Ratio | SPI 达标天数占比 |

---

## 六、系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      V67 量化交易系统                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ v67_data_filler │    │   Database      │                │
│  │                 │───▶│   Manager       │                │
│  │ - 批次等待模式  │    │                 │                │
│  │ - 落库验证      │    └────────┬────────┘                │
│  └─────────────────┘             │                          │
│                                  ▼                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ v67_core        │    │   Data Cache    │                │
│  │                 │◀───│                 │                │
│  │ - SPI 计算      │    │                 │                │
│  │ - IC 审计       │    └─────────────────┘                │
│  │ - 机构踪迹      │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ v67_engine      │    │   SPI Monitor   │                │
│  │                 │───▶│                 │                │
│  │ - 数据熔断      │    │ - 实时监控      │                │
│  │ - 回测执行      │    │ - 警告输出      │                │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 七、关键代码片段

### 7.1 批次等待模式
```python
# 批次等待：每 20 只股票强制 sleep 5 秒
batch_count += 1
if batch_count % V67_BATCH_SIZE == 0:
    logger.info(f"V67: 【批次等待】已处理 {batch_count} 只股票，等待 {V67_BATCH_WAIT_SECONDS} 秒...")
    time.sleep(V67_BATCH_WAIT_SECONDS)
```

### 7.2 落库验证
```python
def _verify_write(self, table_name: str, symbol: str, trade_date: str) -> bool:
    query = f"SELECT COUNT(*) as cnt FROM {table_name} WHERE symbol = '{symbol}' AND trade_date = '{trade_date}'"
    result = self.db.read_sql(query)
    cnt = int(result['cnt'][0])
    return cnt > 0
```

### 7.3 SPI 计算
```python
def _compute_ic_and_spi(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    # 计算 IC (相关系数)
    ic_by_date = df.group_by('trade_date').agg([
        pl.corr('composite_score', 'forward_return_5d').alias('ic')
    ])
    
    ic_mean = float(np.mean(ic_values))
    ic_std = float(np.std(ic_values, ddof=1))
    
    # SPI = IC / IC_Std
    spi = ic_mean / ic_std if ic_std > self.EPSILON else 0.0
    
    return result, {'ic_mean': ic_mean, 'ic_std': ic_std, 'spi': spi}
```

### 7.4 数据熔断检查
```python
def _check_data_sufficiency(self) -> Tuple[bool, str]:
    # 检查 stock_fund_flow 行数
    fund_flow_query = "SELECT COUNT(*) as cnt FROM stock_fund_flow"
    fund_flow_rows = int(self.db.read_sql(fund_flow_query)['cnt'][0])
    
    if fund_flow_rows < V67_MIN_FUND_FLOW_ROWS:
        return (False, "数据不足，请运行 data_filler")
    
    return (True, "数据充足")
```

---

## 八、测试建议

### 8.1 数据填充测试
```bash
# 测试单只股票数据获取
python -c "from v67_data_filler import V67DataFetcher; f = V67DataFetcher(); print(f.fetch_fund_flow_rank_em('2024-01-01'))"
```

### 8.2 数据验证测试
```bash
# 验证数据充足性
python src/run_v67_backtest.py --verify-only
```

### 8.3 回测测试
```bash
# 测试回测引擎（缩短日期范围）
python src/run_v67_backtest.py --start-date 2024-01-01 --end-date 2024-01-31
```

---

## 九、常见问题

### Q1: 数据填充失败怎么办？
**A**: 检查报错类型：
- `timeout`: 网络超时，请检查网络连接或稍后重试
- `ip_blocked`: IP 可能被封禁，请更换 IP 或使用代理
- `no_data`: API 返回空数据，可能是非交易日

### Q2: 数据熔断触发怎么办？
**A**: 运行数据填充器：
```bash
python src/run_v67_backtest.py --fill-data
```

### Q3: SPI 未达标怎么办？
**A**: 系统会自动调整 RS-ZScore 权重，如果持续未达标，建议：
- 检查策略逻辑
- 降低交易频率
- 重新优化信号权重

---

## 十、交付确认

| 检查项 | 状态 | 说明 |
|--------|------|------|
| v67_data_filler.py | ✅ 已交付 | 数据暴力填充工具 |
| v67_core.py | ✅ 已交付 | SPI 驱动预测核心 |
| v67_engine.py | ✅ 已交付 | 回测引擎 |
| run_v67_backtest.py | ✅ 已交付 | 运行脚本 |
| 批次等待模式 | ✅ 已实现 | 每 20 只股票 sleep 5 秒 |
| 落库验证 | ✅ 已实现 | SELECT COUNT(*) 确认 |
| 连续失败退出 | ✅ 已实现 | 连续 3 次失败 sys.exit(1) |
| 数据熔断 | ✅ 已实现 | 100 万行阈值检查 |
| 手续费锁定 | ✅ 已实现 | 0.2% 写死 |
| 报错透明化 | ✅ 已实现 | 打印缺失数据的日期和股票代码 |
| SPI 审计 | ✅ 已实现 | SPI = IC/IC_Std 监控 |
| 机构踪迹 | ✅ 已实现 | 主力净额/流通市值 |

---

## 十一、下一步操作

1. **运行数据填充**:
   ```bash
   python src/run_v67_backtest.py --fill-data
   ```

2. **验证数据充足性**:
   ```bash
   python src/run_v67_backtest.py --verify-only
   ```

3. **运行回测**:
   ```bash
   python src/run_v67_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
   ```

---

**报告生成时间**: 2026-03-24

**系统版本**: V67.0

**交付状态**: ✅ 完成