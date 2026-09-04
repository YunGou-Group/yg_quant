# 贡献

先跑测试：

```powershell
python -m pytest tests
```

Akshare 真实网络探测默认跳过。需要时：

```powershell
$env:AKSHARE_LIVE=1; python -m pytest tests/test_akshare_live.py -q
```

扩展示例（都会被自动扫到，不必改入口）：

| 想加 | 放哪里 |
|------|--------|
| 数据源 | `DailyUpdates/data_fetcher/data_sources/`，继承 `DataSourceBase`，声明 `fetch_slice` |
| 因子 | `DailyUpdates/factor_updates/factors/`，继承 `BaseFactor` |
| 评估指标 | `FactorEvaluates/metrics/`，继承 `BaseMetric` |
| 股票池 | `Universes/`，继承 `Universe` |
| 策略 | `Strategies/`，继承 `StrategyEngine.strategy.Strategy` |
| 仓位分配 | `StrategyEngine/allocators/`，并在 `REGISTRY` 登记 |
| 业绩归因 | `StrategyEngine/attribution/`（`run_attribution` 合同保持不变） |

请不要提交 `data/`、`.env`、token，或把密钥写进配置快照。
