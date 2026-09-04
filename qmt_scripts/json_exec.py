#coding:utf-8
"""大 QMT 模型交易：读研究仓写出的对接 JSON，9:15 集合竞价按目标手数调仓。

用法：复制到客户端「模型研究」，改 ORDER_FILE / ACCOUNT，挂到模型交易。
定时器驱动，不要靠 handlebar（竞价阶段没有 K 线）。
定时器里 passorder 的 quickTrade 必须为 2。

只调 JSON 列出的代码；同账户其他持仓不动。execute_on 必须等于今天，缺省不交易。
"""
import json
import os
import time

# 研究仓 `python -m StrategyEngine --mode live` 写出的 JSON。
# 拷到 QMT 后改成你本机的绝对路径，例如 r"D:\data\yg_quant\strategy_runs\qmt_orders.json"
ORDER_FILE = r""
ACCOUNT = ""  # 资金账号；JSON 里有 account 时优先用文件
START = "091500"
END = "092000"

_done = False


def init(C):
    C.run_time("on_timer", "1nSecond", "2019-10-14 09:00:00")
    if not ORDER_FILE:
        print("json_exec: 请把 ORDER_FILE 改成 qmt_orders.json 的绝对路径")
    print("json_exec init", ORDER_FILE or "(未设置)")


def handlebar(C):
    pass


def execute_on_today(payload, today=None):
    """execute_on 必填且必须是今天；缺省或空字符串一律拒绝。"""
    today = str(today or time.strftime("%Y%m%d")).replace("-", "")[:8]
    raw = payload.get("execute_on") if isinstance(payload, dict) else None
    if raw is None or str(raw).strip() == "":
        return False
    return str(raw).replace("-", "")[:8] == today


def target_from_payload(payload):
    """目标手数。holdings 优先；否则用 orders。volume=0 表示该代码要清到 0。"""
    target = {}
    if not isinstance(payload, dict):
        return target
    for row in payload.get("holdings") or []:
        code = str(row.get("code") or "").strip()
        if not code:
            continue
        volume = int(row.get("volume") or 0)
        if volume < 0:
            volume = 0
        target[code] = target.get(code, 0) + volume
    if target:
        return target
    for row in payload.get("orders") or []:
        code = str(row.get("code") or "").strip()
        volume = int(row.get("volume") or 0)
        side = str(row.get("side") or "buy").lower()
        if not code or volume <= 0:
            continue
        signed = volume if side == "buy" else -volume
        target[code] = target.get(code, 0) + signed
    return target


def rebalance_deltas(target, current):
    """只对 target 里的代码做差额。current 里多出来的票不出现在结果中。"""
    current = current or {}
    out = []
    for code in sorted(target):
        want = int(target.get(code, 0))
        have = int(current.get(code, 0))
        delta = want - have
        if delta != 0:
            out.append((code, delta))
    return out


def on_timer(C):
    global _done
    now = time.strftime("%H%M%S")
    if _done or now < START or now > END:
        return
    payload = _load()
    if payload is None:
        return
    if not execute_on_today(payload):
        raw = payload.get("execute_on")
        print("execute_on 缺失或不是今天，拒绝下单:", raw)
        return
    if _send(C, payload):
        _done = True


def _load():
    if not ORDER_FILE:
        print("ORDER_FILE 未设置")
        return None
    if not os.path.isfile(ORDER_FILE):
        print("找不到对接文件", ORDER_FILE)
        return None
    with open(ORDER_FILE, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _send(C, payload):
    account = str(payload.get("account") or ACCOUNT).strip()
    if not account:
        print("资金账号为空")
        return False
    current = _positions(account)
    if current is None:
        return False
    target = target_from_payload(payload)
    for code, delta in rebalance_deltas(target, current):
        if delta > 0:
            passorder(23, 1101, account, code, 12, -1, delta, "json_exec", 2, "json", C)
            print("buy", code, delta)
        else:
            passorder(24, 1101, account, code, 12, -1, -delta, "json_exec", 2, "json", C)
            print("sell", code, -delta)
    return True


def _field(row, *names):
    for name in names:
        if isinstance(row, dict):
            val = row.get(name)
        else:
            val = getattr(row, name, None)
        if val is not None and val != "":
            return val
    return None


def _positions(account):
    try:
        rows = get_trade_detail_data(account, "stock", "position")
    except Exception as exc:
        print("读持仓失败，本次不下单：", exc)
        return None
    held = {}
    if not rows:
        return held
    for row in rows:
        code = str(_field(row, "m_strInstrumentID", "stockcode") or "")
        market = str(_field(row, "m_strExchangeID", "exchange") or "")
        if code and market and "." not in code:
            code = code + "." + market
        vol = _field(row, "m_nVolume", "volume")
        if not code or vol is None:
            continue
        held[code] = held.get(code, 0) + int(vol)
    return held
