# Quantitative-Trading
quant_trader/
├── config/
│   └── settings.py
├── outputs/                      # 存放运行结果和图表
├── scripts/
│   ├── daily_update.py           # 拉取股票数据启动脚本
│   ├── run_backtest.py           # 策略回测的启动脚本
|   └── run_factor_analysis.py    # 因子批量测试启动脚本
└── src/
    ├── backtest/                 # 回测层（简单的回测引擎）
    │   └── engine.py
    ├── data/                  
    │   ├── database_manager.py   # 数据库相关操作
    │   └── data_fetcher.py       # 数据读取
    ├── strategies/               # 策略层
    |   └── dual_ma.py            # 双均线策略
    └── factors/
        ├── factor_analyzer.py    # 因子测试
        └── factor_library.py     # 因子工厂