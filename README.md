# yg_quant

A 股日频多因子研究平台：因子评估、回测与归因，并可对接迅投 QMT 实盘。

从 Tushare / Akshare 入库、计算因子、做研究评估，再到网页上回测和归因。实盘只负责把同一套 `score` 写成迅投 QMT 可读的 JSON，不是自带柜台的交易系统。

数据源、因子、评估指标、股票池、策略、仓位分配都按同一套插件合同扩展：继承基类、放到指定目录，调度器 / CLI / 网页会自动发现，不必改引擎入口。

仓库内置 **小市值** 策略作为参考实现（[`Strategies/small_cap.py`](Strategies/small_cap.py)）：周五收盘给名单，下个交易日（通常周一）开盘成交；回测、归因、实盘共用这一份 `score`。另有因子 TopK。换自己的策略时，照 `score(ctx)` 合同写即可。

- **DailyUpdates**：行情 / 财务 / 行业入库（SQLite）+ 因子增量（Qlib 风格 bin）
- **FactorEvaluates**：单因子评估、全库批评估、CNE5-lite 风格中性化
- **StyleCrowding**：Barra Style 拥挤监测（独立风控，不绑日更 batch）
- **frontend**：网页（Vue 3 + Vite）
- **App**：网页后端（FastAPI）。评估、回测等 HTTP 都挂在这里，领域实现仍在各自的包
- **StrategyEngine**：日频运行时。合同是 `score(ctx) → 目标权重`；`backtest/` 做模拟撮合；`allocators/` 是仓位分配；`live/` 按分配资金写出 QMT 对接 JSON；`attribution/` 读取某次回测做业绩归因
- **Strategies**：策略库。参考实现是小市值，另有因子 TopK。回测与实盘共用同一套 `score`
- **Universes**：asof 股票池（ST / 指数成分 / 板块）。评估、回测、实盘、风格拥挤共用；**不**过滤数据拉取
- **qmt_scripts**：拷进**迅投 QMT** 客户端的脚本（不在仓库里执行）

库内一律 `from DailyUpdates...` / `from FactorEvaluates...` / `from StrategyEngine...` / `from Strategies...` / `from Universes...` / `from App...`。入口可用 `python -m`，也可直接跑对应 `.py`（会自动把仓库根加入 `sys.path`）。

Python 3.10–3.12。MIT License，见 [LICENSE](LICENSE)。

本仓库是研究工具，**不是**投资建议，也**不是**迅投 QMT / MSCI Barra 的官方产品。拉行情须遵守 Tushare / Akshare 的使用条款；你需要自备 `TUSHARE_TOKEN`。截图里的回测数字只是本地样例，不是收益承诺。

## 效果展示

因子分析与策略回测是并列入口。下面是本地网页的效果。

**因子总览**（全库 RankIC / IR）：

![因子总览](docs/screenshots/factor-eval.jpg)

**深度报告**（单因子 IC 序列、分布与分层）：

![深度报告](docs/screenshots/factor-detail.jpg)

**策略回测**（默认参考策略 `small_cap`，等权，净值相对基准）：

![小市值回测净值](docs/screenshots/backtest-nav.jpg)

**业绩归因**（交易 / 持仓洋葱树，以及行业 Brinson、本地十风格）：

![业绩归因](docs/screenshots/attribution.jpg)

## 可拓展

研究链路每一层都是插件。放对目录、继承基类，调度器、命令行和网页会扫到，不必改 `__main__` 或 FastAPI 路由。仓位分配器例外：还要在 `REGISTRY` 登记一行。

| 要加 | 放哪里 | 合同 | 出现在 |
|------|--------|------|--------|
| 数据源 | `DailyUpdates/data_fetcher/data_sources/` | `DataSourceBase` + `fetch_slice` | `dataset_config` 的 `data_source`（类名去掉 `DataSource` 后缀：`AkshareDataSource` → `Akshare`） |
| 数据集 | [`main_scheduler.py`](DailyUpdates/data_fetcher/main_scheduler.py) 的 `dataset_config` | `data_type` / `api_name` / `fields` | 调度器对照 `data/dataset_config.json` 自动 install |
| 因子 | `DailyUpdates/factor_updates/factors/`（含子目录） | `BaseFactor.calculate` | 增量更新、评估、回测 `ctx.factor` |
| 评估指标 | `FactorEvaluates/metrics/` | `BaseMetric.compute` | 网页勾选、批评估 |
| 股票池 | `Universes/` | `Universe` + `name` | `--universe`、下拉 |
| 策略 | `Strategies/` | `Strategy.score`，以及 `from_cli` / `panel_kwargs` / `run_tag` | `--strategy`、网页 |
| 仓位分配 | `StrategyEngine/allocators/` | `Allocator.size` + `REGISTRY` | `--allocator` |
| 归因分解 | `StrategyEngine/attribution/` | 保持 `run_attribution` | `--mode attribute`、网页洋葱树 |

最小形态（其余按旁边的参考实现抄）：

```python
# 数据源：DailyUpdates/data_fetcher/data_sources/foo_data_source.py
class FooDataSource(DataSourceBase):
    fetch_slice = FetchSlice.BY_DATE  # 或 BY_SYMBOL / BY_PANEL
    def fetch_data(self, config, start_date, end_date):
        return pd.DataFrame(...)  # 引擎按 fetch_slice 选窗口，不必改调度器
# dataset_config 里写 data_source: "Foo"
```

```python
# 因子：DailyUpdates/factor_updates/factors/my_factor.py
class MyFactor(BaseFactor):
    def __init__(self):
        self.name = "my_factor"
        self.dependencies = ["close"]
        super().__init__()
    def calculate(self, data, start_date=None):
        ...  # 返回含 date / symbol / 因子值列的 DataFrame
```

```python
# 指标：FactorEvaluates/metrics/my_metric.py
class MyMetric(BaseMetric):
    name = "my_metric"
    def compute(self, ctx, params):
        return MetricResult(scalars={...})  # 网页勾选框按 name 出现
```

```python
# 策略：Strategies/my_strategy.py
class MyStrategy(Strategy):
    name = "my_strategy"
    @classmethod
    def from_cli(cls, args, allocator): ...
    @classmethod
    def panel_kwargs(cls, args) -> dict: ...
    @classmethod
    def run_tag(cls, args) -> str: ...
    def score(self, ctx):
        return TargetHoldings(...)  # 股票 → 权重；None / 全 0 为空仓
```

股票池更短：在 `Universes/` 加一个 `Universe` 子类、设 `name`（可选 `index_codes` / `boards`）即可。扩展示例清单见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 安装

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env   # 可选；把 TUSHARE_TOKEN 写进去或只设环境变量
$env:TUSHARE_TOKEN = "你的 token"   # 拉行情时必需
```

前端：

```powershell
cd frontend
npm install
```

## 数据目录

默认都在仓库根 `data/`，按仓库根解析，不依赖当前工作目录。`data/` 已 gitignore，克隆仓库不会带数据。

```text
data/
  yg_quant.db                 # SQLite
  dataset_config.json    # 数据集快照（调度器自动维护）
  factors/               # 因子 bin：calendars/ + features/{SH600000}/*.day.bin
  factor_eval/           # 评估摘要、jobs、全库 runs
  style_crowding/        # Style 拥挤 published/
  strategy_runs/         # 回测净值 CSV、成交明细、指标 JSON；实盘 qmt_orders.json
```

覆盖顺序：**构造参数 > 环境变量 > `data/`**。已有 `data/qmt.db` 且尚未改名为 `yg_quant.db` 时，默认仍会用旧文件。

| 变量 | 作用 | 默认 |
|------|------|------|
| `YG_QUANT_DATA_DIR` | 数据根 | `<仓库根>/data` |
| `YG_QUANT_DB_PATH` | SQLite | `<数据根>/yg_quant.db` |
| `YG_QUANT_FACTOR_DIR` | 因子 bin | 与库同级的 `factors/` |
| `YG_QUANT_EVAL_DIR` | 评估输出 | 与库同级的 `factor_eval/` |
| `YG_QUANT_STYLE_CROWDING_DIR` | Style 拥挤发布 | `<数据根>/style_crowding/` |
| `YG_QUANT_START_DATE` | 日更可选起点 `YYYYMMDD` | 空库从 `20050101` |
| `YG_QUANT_END_DATE` | 日更截止日 `YYYYMMDD` | 当天 |
| `YG_QUANT_FACTOR_ONLY` | 只更新匹配的因子名 | 全部 |
| `YG_QUANT_FACTOR_START_DATE` | 因子首次写入起点 | 写满可用历史 |
| `YG_QUANT_FACTOR_REBUILD` | `1` 时全量重算因子 | 增量 |
| `YG_QUANT_FACTOR_WORKERS` | 因子增量**计算**线程数 | `8` |
| `YG_QUANT_FACTOR_WRITE_WORKERS` | 同时写入几个因子（Windows 建议 `1`） | `1` |
| `YG_QUANT_PRICE_ADJUST` | 读层复权：`hfq` / `none` | `hfq` |
| `YG_QUANT_BETA_INDEX` | 风格 Beta 对照指数 | `index_SH000300` |
| `YG_QUANT_BIN_WRITE_WORKERS` | 单个因子内部的 bin 写线程 | Windows `8` / 其他 `16` |
| `YG_QUANT_JOB_TTL_SECONDS` | 网页任务保留秒数 | `3600` |
| `YG_QUANT_JOB_KEEP_FINISHED` | 网页已完成任务条数 | `20` |
| `YG_QUANT_JOB_MAX_ACTIVE` | 网页同时排队/运行上限 | `8` |

只改库位置时设 `YG_QUANT_DB_PATH` 即可，因子默认仍在该库同级 `factors/`。因子另放再设 `YG_QUANT_FACTOR_DIR`。

## 更新行情

```powershell
python -m DailyUpdates.data_fetcher.main_scheduler
```

首次从 `20050101` 重建；之后从库内最后交易日继续。主键 `(symbol, trade_date)` 更新，不会插重复行。

任一核心数据集拉取失败会让当日整体失败并终止调度（退出码 1），因子更新不会在行情残缺时继续跑。

在 [`main_scheduler.py`](DailyUpdates/data_fetcher/main_scheduler.py) 的 `dataset_config` 里增删数据集后，调度器对照 `data/dataset_config.json` 自动 install / 清理 / 增量。

默认数据源是 Tushare（需要 `TUSHARE_TOKEN`）。同一套 `data_type` / `api_name` 也可以改成 `Akshare`，不需要 token：

```python
'daily': {
    'data_source': 'Akshare',
    'data_type': 'daily',
    'api_name': 'daily',
    'fields': ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount'],
    'primary_key': ['ts_code', 'trade_date'],
    'pause_seconds': 0.2,
}
```

Akshare 多数接口按股票循环，全市场历史会比 Tushare 慢很多。涨跌停价按昨收和板块幅度推算；指数成分权重是中证最新快照，不是逐日调样。调度器按数据源类上的 `fetch_slice` 选窗口：`BY_DATE`（Tushare 全市场日频）按自然日循环写入，`BY_SYMBOL` / `BY_PANEL`（Akshare 等）把整个日期区间一次交给数据源，避免按日把全市场请求重复成千上万次。新数据源只要在类上声明切片类型，不必改引擎。

## 复权

库里存的是 Tushare `daily` 的**原始未复权价**，另外单独落一列 `adj_factor`（Tushare `adj_factor` 接口，按交易日批量拉）。复权在**读取层**完成，按官方口径算后复权：

```
后复权价 = 原始价 × adj_factor
```

后复权的历史值不随后续分红变化，所以增量入库和 Bin 因子的时间轴都不会因为一次分红而整体漂移（前复权会）。

| 字段 | 处理 |
| --- | --- |
| `open` `high` `low` `close` `pre_close` `up_limit` `down_limit` | `× adj_factor` |
| `vol` | `÷ adj_factor`，使 `vwap = amount / vol` 也是后复权价 |
| `amount` `pct_chg` `total_mv` `circ_mv` `pb` `pe` `ps` | 不变（本身已是除权口径或与复权无关） |
| 指数行 `index_*` | 无因子，按 `1.0` 处理 |

因子计算、因子评估、策略回测、风格拥挤默认全部走后复权。用 `YG_QUANT_PRICE_ADJUST=none` 可以退回原始价做对照。

## 更新因子

```powershell
python -m DailyUpdates.factor_updates.factor_updater
```

只重建风格描述子：

```powershell
python -m DailyUpdates.factor_updates.factor_updater --only "style_*"
```

切换复权口径或修改因子实现后，必须整段覆盖重算，否则新旧口径会拼在同一条序列里：

```powershell
python -m DailyUpdates.factor_updates.factor_updater --rebuild
```

`--rebuild` 等价于 `YG_QUANT_FACTOR_REBUILD=1`。重算完还要重跑一次全库评估，旧 run 的结果仍是旧口径。改过行业中性 Alpha（PIT 行业而不是最新分类）后，相关 `alpha*` 也必须 `--rebuild`。

实现放在 `DailyUpdates/factor_updates/factors/`，继承 `BaseFactor` 后自动发现。

CNE5-lite / Barra10 在 `factors/risk/`，落盘 9 个 `style_*` 原始描述子（不做截面 z-score）。**NLSIZE 不写 Bin**，评估时在当前股票池上由 `z(Size)³` 对 Size 残差得到。这是研究用的十条风格轴，不是 MSCI 官方复合因子。

## 网页

两个进程：后端出日志，前端热更新。

```powershell
python -m App
```

```powershell
cd frontend
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`（`/api` 转到 8765）。评估、回测的后台输出看跑 `python -m App` 的那个终端，不要打开 8765 当页面——那会把 JS/CSS 访问日志刷满。

首页可选择**因子分析**或**策略回测**（效果见上方截图）。因子评估：选好因子和参数后点「开始计算」，摘要写入 `data/factor_eval/{factor}.json`。策略回测页默认策略是内置小市值，只是可视化：选参数后把结果画成净值曲线和指标；命令行回测仍是 `python -m StrategyEngine`，网页会去读同一目录 `data/strategy_runs/` 里的快照。载入某次回测后可点「业绩归因」，走 `GET/POST /api/backtest/runs/{id}/attribution`。

## 因子评估

远期收益契约 B：`fwd_ret_Nd[T] = open[T+1+N] / open[T+1] - 1`。

全库批评估：

```powershell
python -m FactorEvaluates.run_batch --start 2010-01-01
```

新增指标：在 `FactorEvaluates/metrics/` 增加 `BaseMetric` 子类即可出现勾选框。勾选 **pure_ic** 时会在当前股票池上组 \(X_T\)（9 个 `style_*` 标准化 + 现算 NLSIZE + 申万一级），再对残差做 RankIC。

## 股票池

评估、回测、实盘、风格拥挤在读面板时套同一套 `--universe` mask，**不**改 SQLite / 因子 bin。mask 只约束新开仓候选（`ctx.universe()`）；已有持仓不会因为调出成分而被引擎自动清掉。

规则是 **asof PIT**：ST 用 `stock_namechange` 当时简称（不用今天的名字否决整段历史）；指数用最近一期成分快照；上市/退市按 `list_date` / `delist_date`；只要求**当日**有开盘，不偷看 T+1 停牌。因此旧评估/回测数字会和改规则之前不同。

`namechange` 为空时跳过 ST 过滤并打 warning，不会回退当前简称。指数池若缺 `index_constituent` 行会直接报错，把对应指数加进 `main_scheduler.py` 的 `universe_index_weight.index_list`。

内置：`all`，指数 `hs300` / `zz500` / `zz1000` / `sz50` / `cyb` / `zzhz` / `sme` / `kcb50`，板块 `main` / `gem` / `star` / `bse`。新池：在 `Universes/` 增加一个 `Universe` 子类（设 `name`，可选 `index_codes` / `boards`），CLI 和网页下拉会扫到。

```powershell
python -m StrategyEngine --mode backtest --universe hs300 --start 2010-01-01 --factor alpha001 --n 50
```

## 策略回测

契约与评估相同：信号在收盘 T 可得，**目标仓位在 T+1 开盘成交**。策略只实现 `score(ctx) -> TargetHoldings`（股票→权重），不返回买卖动作。`None` / 全 0 表示空仓。标的列表和目标权重不变时，引擎维持当前股数，不因股价涨跌再配平。账户快照由引擎写入下一个 `ctx.position` / `ctx.cash`。选出候选后可用仓位插件分配权重（默认等权）。`get_allocator("max_sharpe")` 这类名字对应 `StrategyEngine/allocators/` 里的日频目标（均值方差 / 路径风险 / HRP）；每个调仓日只对**当前候选**解一次，μ / Σ 用 asof 前 `lookback` 根（默认 252），不偷看全样本。

```python
from StrategyEngine.allocators import MinVol, get_allocator

w = MinVol().size(ctx, picks)
# 默认 lookback=252、max_weight=1（不限制单票上限）
# 或 get_allocator("min_vol", lookback=252, max_weight=0.2)
```

`--allocator` 可选：`equal`、`min_vol`、`max_sharpe`、`max_quadratic_utility`、`efficient_return`、`efficient_risk`、`multiobjective`、`min_l2`、`min_semivariance`、`min_cvar`、`min_cdar`、`hrp`。需要 μ 的目标默认用几何历史收益（`--mu-model arithmetic|ema` 可换）。未接入：CLA 全前沿、非凸 max_sharpe、整手 MILP、Black-Litterman。

新仓位分配器放在 `StrategyEngine/allocators/`，在 `REGISTRY` 登记后任意策略都可注入。带 `lookback` 的优化器会让引擎多装一段行情供 `ctx.history` 使用，净值仍从 `--start` 起算。新策略继承 `StrategyEngine.strategy.Strategy`，实现抽象方法 `score`，并在类上声明 `name`，加上 `from_cli` / `panel_kwargs` / `run_tag`，`StrategyEngine` 会扫到，不必改 `__main__`。

```powershell
python -m StrategyEngine --mode backtest --start 2010-01-01 --factor alpha001 --n 50
python -m StrategyEngine --mode backtest --strategy small_cap --start 2023-01-01
python -m StrategyEngine --mode backtest --strategy small_cap --allocator min_vol
python -m StrategyEngine --mode backtest --strategy small_cap --allocator max_sharpe --mu-model arithmetic
python -m StrategyEngine --mode backtest --strategy small_cap --anti-tail --start 2023-01-01
python -m StrategyEngine --mode live --strategy small_cap --cash 500000 --account 资金账号
python -m StrategyEngine --mode attribute --run small_cap_20260902_120000
```

命令行必须带 `--mode backtest`、`--mode live` 或 `--mode attribute`。`--mode live` 与回测走同一套 `score` / `--allocator`，用 `--cash` 作为分配给该策略的资金，把目标仓换成 QMT 手数，写入 `data/strategy_runs/qmt_orders.json`（临时文件再替换；没有下一交易日则拒绝写出）。客户端脚本在 [`qmt_scripts/`](qmt_scripts/README.md)，复制到大 QMT 模型交易，9:15 读这份 JSON 下单。只调 JSON 列出的代码，同账户其他持仓不动；`execute_on` 必须是当天。

`--mode attribute` 不跑策略：读取 `data/strategy_runs/` 里某次回测的净值与成交，按收盘盯市还原持仓，调用 [`StrategyEngine/attribution`](StrategyEngine/attribution) 做交易/持仓分解、行业 Brinson，以及本地 CNE5-lite 十风格 + 行业 + 市场联动 + 特异收益，写出 `*_attribution.json`。也可以 `python -m StrategyEngine.attribution --run <快照id>`。网页回测页用同一份结果画洋葱收益树和风格表。缺少风格段的文件不算已归因，需重算。

净值 CSV、成交明细和指标快照 JSON 默认写到 `data/strategy_runs/`（同一时间戳前缀：`*.csv` / `*_trades.csv` / `*.json`）。成交为 T+1 开盘、涨跌停用表内限价、印花税仅卖出。JSON 里是年化、最大回撤、波动、夏普、换手、超额等，方便把多种 allocator 并排对比。

**参考策略是小市值**（`Strategies/small_cap.py`），网页回测和上面的命令行示例都走它：**周五收盘给名单，下个交易日（通常周一）开盘成交**；周五休市则用当周最后一个交易日。用来对照引擎合同，不是推荐组合。

## 数据库

- `market_data`：行情及基本面（含可选 `up_limit` / `down_limit`）
- `instruments`：标的及其数据起止日期
- `trade_calendar`：已有交易日
- `dataset_registry`：数据集配置及更新状态
- `industry_classify`：申万行业分类（[index_classify](https://tushare.pro/document/2?doc_id=181)）
- `industry_member`：申万行业成分（[index_member_all](https://tushare.pro/document/2?doc_id=335)）
- `stock_basic`：股票名称 / 上市日等（[stock_basic](https://tushare.pro/document/2?doc_id=25)）
- `stock_namechange`：历史简称（[namechange](https://tushare.pro/document/2?doc_id=100)），股票池 ST 按 asof 用这张表
- `financial_indicator`：财务指标（`fina_indicator` / `fina_indicator_vip`）。成长 / 杠杆等截面对齐用 **`ann_date`（公告日）**，不要用 `end_date`
- `index_constituent`：指数成分权重（[index_weight](https://tushare.pro/document/2?doc_id=96)）

因子数值不进 SQLite，只在 `data/factors/`。

### 可配置的 `data_type`

| data_type | 典型 api_name | 落库 |
|-----------|---------------|------|
| `daily` | `daily` / `daily_basic` / `stk_limit` | `market_data` |
| `index` | `index_daily` | `market_data`（`index_*`） |
| `industry` | `index_classify` / `index_member_all` | 行业表 |
| `stock_info` | `stock_basic` | `stock_basic` |
| `financial` | `fina_indicator_vip` | `financial_indicator` |
| `index_constituent` | `index_weight` | `index_constituent` |

## StyleCrowding（风格拥挤）

独立风控模块，不绑 `main_scheduler` 与日更 batch：

```powershell
python -m StyleCrowding run --start 2018-01-01
python -m StyleCrowding incremental
```

网页：**因子分析 → 风格拥挤**（`/style-crowding`）。详见 [`StyleCrowding/README.md`](StyleCrowding/README.md)。

## 测试

```powershell
python -m pytest tests
```

Akshare 真实接口探测默认跳过。需要时设 `AKSHARE_LIVE=1`。扩展示例见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 第三方与口径

- Alpha101 按 Kakushadze, *101 Formulaic Alphas*, 2016（[arXiv:1601.00991](https://arxiv.org/abs/1601.00991)）实现；本仓库做了面板适配、PIT 行业中性和 Numba 加速。非整数窗口四舍五入为整数。
- CNE5-lite / Barra10 是研究用风格轴，**不是** MSCI 官方复合因子。
- 迅投 QMT 只出现在实盘桥（`qmt_scripts/`、`600000.SH` 代码）。研究仓本体叫 **yg_quant**。
