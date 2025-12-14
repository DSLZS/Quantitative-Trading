# Quantitative-Trading
quant_trader/
├── config/
│   └── settings.py
├── outputs/                    # 【新增】存放回测结果和图表
├── scripts/
│   ├── daily_update.py         # 已有，我们将修改它
│   └── run_backtest.py         # 【新增】本次策略回测的启动脚本
└── src/
    ├── backtest/               # 【新增】回测层（简单的回测引擎）
    │   └── engine.py
    ├── data/                  # 已有，数据层
    │   ├── database_manager.py # 已有，适配了你的表结构，需小改
    │   └── data_fetcher.py     # 已有
    └── strategies/               # 【新增】策略层（仅放一个双均线策略）
        └── dual_ma.py