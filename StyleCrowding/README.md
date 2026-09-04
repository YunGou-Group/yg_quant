# StyleCrowding

Barra **Style** 拥挤监测：独立于 `FactorEvaluates/batch` 与 `main_scheduler` 的风控模块。

## 边界

| 做 | 不做 |
|---|---|
| 9 个 `style_*` 目标 × group_10/5 × positive/reverse × 指标序列（动量核心仅 21 日） | 并入 alpha 全库 batch |
| 独立 CLI + HTTP + `/style-crowding` 前端 | 依赖外部 Parquet 因子库 |
| prior-Z / 因果分位 / 事件分析 | 修改 alpha `factor_crowding` |

## 数据目录

默认 `<data>/style_crowding/published/`，可用 `YG_QUANT_STYLE_CROWDING_DIR` 覆盖。

```text
published/
  manifest.json
  market_exposure.json
  event_analysis.json
  {style}/group_{10|5}/{positive|reverse}/{indicator}.parquet
```

## 运行

日更完成后可自行执行（不绑调度器）：

```powershell
python -m StyleCrowding run --start 2018-01-01
python -m StyleCrowding incremental
python -m StyleCrowding run --skip pairwise   # 调试时跳过成对相关
```

## 实现说明

- 风格因子收益：`StyleReturnEngine` WLS（同日 close 收益）
- 特异收益：Context 内本地 WLS 残差 `(T×N)`
- 目标数：默认 9 个 bin 风格；`include_nlsize=True` 可扩第 10 项

## HTTP

- `GET /api/style-crowding/bootstrap`
- `GET /api/style-crowding/series?style=&indicator=`
- `POST /api/style-crowding/run`
