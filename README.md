# yg_quant

A 股日频多因子研究平台：因子评估、回测与归因，并可对接迅投 QMT 实盘。

仅供研究使用，不构成投资建议。

从行情入库、算因子、网页评估，到回测和归因。实盘把同一套 `score` 写成 QMT 可读的 JSON。内置小市值策略作参考：周五收盘给名单，下个交易日开盘成交；因子 TopK策略仅作为策略实现的演示。

Python 3.10–3.12。MIT，见 [LICENSE](LICENSE)。

## 效果展示

**因子总览**

![因子总览](docs/screenshots/factor-eval.jpg)

**深度报告**

![深度报告](docs/screenshots/factor-detail.jpg)

**策略回测**

![小市值回测净值](docs/screenshots/backtest-nav.jpg)

**业绩归因**

![业绩归因](docs/screenshots/attribution.jpg)

## 目录

| 包 | 做什么 |
|----|--------|
| `DailyUpdates` | 行情入库、因子增量 |
| `FactorEvaluates` | 因子评估 |
| `StyleCrowding` | 风格拥挤 |
| `frontend` / `App` | 网页与后端 |
| `StrategyEngine` | 回测、仓位、实盘桥、归因 |
| `Strategies` | 策略 |
| `Universes` | 股票池 |
| `qmt_scripts` | 拷进迅投 QMT 的脚本 |

入口用 `python -m`。

## 安装

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

拉行情需要 `TUSHARE_TOKEN`，写进 `.env` 即可。前端：`cd frontend && npm install`。

数据默认在仓库根 `data/`，已 gitignore。环境变量一览见文末。

## 更新行情

```powershell
python -m DailyUpdates.data_fetcher.main_scheduler
```

空库从 `20050101` 重建，之后按最后交易日增量。增删数据集改 [`main_scheduler.py`](DailyUpdates/data_fetcher/main_scheduler.py) 里的 `dataset_config`。默认 Tushare；也可把 `data_source` 改成 `Akshare`。

库里是未复权价加一列 `adj_factor`。读的时候默认后复权：`原始价 × adj_factor`。对照原始价设 `YG_QUANT_PRICE_ADJUST=none`。

## 更新因子

```powershell
python -m DailyUpdates.factor_updates.factor_updater
python -m DailyUpdates.factor_updates.factor_updater --only "style_*"
python -m DailyUpdates.factor_updates.factor_updater --rebuild
```

改过实现或复权口径后必须 `--rebuild`，再重跑全库评估。因子放在 `DailyUpdates/factor_updates/factors/`，继承 `BaseFactor`。

## 网页

```powershell
python -m App
```

```powershell
cd frontend
npm run dev
```

打开 `http://127.0.0.1:5173`。评估和回测的日志看跑 `python -m App` 的终端。

## 因子评估

远期收益：`fwd_ret_Nd[T] = open[T+1+N] / open[T+1] - 1`。

```powershell
python -m FactorEvaluates.run_batch --start 2010-01-01
```

新指标放进 `FactorEvaluates/metrics/`。

## 股票池

评估、回测、实盘、风格拥挤共用 `--universe`，不改底层数据。规则按当时信息：ST 用历史简称，指数用最近成分快照。

内置：`all`，指数 `hs300` / `zz500` / `zz1000` / `sz50` / `cyb` / `zzhz` / `sme` / `kcb50`，板块 `main` / `gem` / `star` / `bse`。新池在 `Universes/` 加一个 `Universe` 子类。

## 策略回测

信号在收盘 T 可得，目标仓位 T+1 开盘成交。策略只实现 `score(ctx) -> TargetHoldings`。

```powershell
python -m StrategyEngine --mode backtest --start 2010-01-01 --factor alpha001 --n 50
python -m StrategyEngine --mode backtest --strategy small_cap --start 2023-01-01
python -m StrategyEngine --mode backtest --strategy small_cap --allocator min_vol
python -m StrategyEngine --mode live --strategy small_cap --cash 500000 --account 资金账号
python -m StrategyEngine --mode attribute --run small_cap_20260902_120000
```

`--allocator` 默认等权。实盘写出 `data/strategy_runs/qmt_orders.json`，客户端脚本见 [`qmt_scripts/`](qmt_scripts/README.md)。归因读已有回测快照，不重跑策略。

新策略、新分配器怎么加，见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 可拓展

数据源、因子、指标、股票池、策略都是放对目录、继承基类就会被扫到。仓位分配器还要在 `REGISTRY` 登记一行。

| 要加 | 放哪里 |
|------|--------|
| 数据源 | `DailyUpdates/data_fetcher/data_sources/` |
| 数据集 | `main_scheduler.py` 的 `dataset_config` |
| 因子 | `DailyUpdates/factor_updates/factors/` |
| 评估指标 | `FactorEvaluates/metrics/` |
| 股票池 | `Universes/` |
| 策略 | `Strategies/` |
| 仓位分配 | `StrategyEngine/allocators/` |

## StyleCrowding

```powershell
python -m StyleCrowding run --start 2018-01-01
python -m StyleCrowding incremental
```

网页：**因子分析 → 风格拥挤**。详见 [`StyleCrowding/README.md`](StyleCrowding/README.md)。

## 测试

```powershell
python -m pytest tests
```

Akshare 真实接口默认跳过，设 `AKSHARE_LIVE=1` 才跑。

## 第三方

- Alpha101 按 Kakushadze, *101 Formulaic Alphas*, 2016（[arXiv:1601.00991](https://arxiv.org/abs/1601.00991)）。
- 风格轴是研究用实现，不是 MSCI 官方因子。

---

## 附录

### 数据目录

```text
data/
  yg_quant.db
  factors/
  factor_eval/
  style_crowding/
  strategy_runs/
```

覆盖顺序：构造参数 > 环境变量 > `data/`。

| 变量 | 默认 |
|------|------|
| `YG_QUANT_DATA_DIR` | `<仓库根>/data` |
| `YG_QUANT_DB_PATH` | `<数据根>/yg_quant.db` |
| `YG_QUANT_FACTOR_DIR` | 库同级 `factors/` |
| `YG_QUANT_EVAL_DIR` | 库同级 `factor_eval/` |
| `YG_QUANT_STYLE_CROWDING_DIR` | `<数据根>/style_crowding/` |
| `YG_QUANT_START_DATE` / `YG_QUANT_END_DATE` | 空库从 `20050101` 到当天 |
| `YG_QUANT_FACTOR_ONLY` | 全部因子 |
| `YG_QUANT_FACTOR_REBUILD` | 增量 |
| `YG_QUANT_FACTOR_WORKERS` | `8` |
| `YG_QUANT_PRICE_ADJUST` | `hfq` |
| `YG_QUANT_BETA_INDEX` | `index_SH000300` |

### 库表

行情在 `market_data`，因子在 `data/factors/` 的 bin，不进 SQLite。其余是行业、股票资料、财务、指数成分等 sidecar。财务截面对齐用 `ann_date`。
