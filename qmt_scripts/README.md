# QMT 客户端脚本

这里只放**拷进大 QMT「模型研究 / 模型交易」**的脚本，不跑仓库里的 Python。

研究仓用 `python -m StrategyEngine --mode live` 写出对接 JSON；客户端脚本读这份文件，在 9:15 集合竞价报单。

| 文件 | 作用 |
| --- | --- |
| [`json_exec.py`](json_exec.py) | 读 `qmt_orders.json`，9:15–9:20 只对 JSON 列出的代码调仓 |

## 用法

1. 仓库里生成委托（资金是分配给该策略的额度，allocator 与回测相同）：

```powershell
python -m StrategyEngine --mode live --strategy small_cap --cash 500000 --account 你的资金账号
```

默认写到 `data/strategy_runs/qmt_orders.json`。可用 `--qmt-json` 改路径。

2. 把 `json_exec.py` 复制到 QMT 模型研究，把文件顶部的 `ORDER_FILE` 改成研究仓写出的**绝对路径**（仓库里默认是空字符串），必要时再改 `ACCOUNT`。
3. 每个交易日 **9:15 前** 启动模型交易。脚本等到 9:15 才 `passorder`（竞价开始前交易所不收单）。
4. JSON 必须带当天的 `execute_on`，缺省或过期文件不会下单。只调文件里列出的代码；同账户其他持仓（人工仓 / 别的策略）不动。仍建议挂独立资金账号。

代码用仓库符号 `SH600000`；JSON 里已转成 QMT 的 `600000.SH`。
