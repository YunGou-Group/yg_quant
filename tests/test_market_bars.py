"""入库前行情清洗：占位价、OHLC、成交量、adj 跳变。"""

import numpy as np
import pandas as pd

from DailyUpdates.data_fetcher.preprocessing.market_bars import sanitize_market_bars


def _bars(**cols):
    return pd.DataFrame(cols)


def test_tiny_open_nulls_prices_index_untouched():
    frame = _bars(
        symbol=["BJ920227", "index_SH000300"],
        date=["2022-02-25", "2022-02-25"],
        open=[0.01, 0.01],
        high=[10.0, 4000.0],
        low=[10.0, 3900.0],
        close=[10.0, 3950.0],
        pre_close=[9.0, 3940.0],
    )
    out = sanitize_market_bars(frame)
    stock = out[out["symbol"] == "BJ920227"].iloc[0]
    assert pd.isna(stock["open"])
    assert pd.isna(stock["close"])
    index = out[out["symbol"] == "index_SH000300"].iloc[0]
    assert index["open"] == 0.01
    assert index["close"] == 3950.0


def test_adj_jump_in_batch_and_via_prev_adj():
    batch = _bars(
        symbol=["SH600000", "SH600000"],
        date=["2026-01-05", "2026-01-06"],
        adj_factor=[1.0, 10.0],
    )
    out = sanitize_market_bars(batch)
    assert out.loc[out["date"] == "2026-01-06", "adj_factor"].isna().all()
    assert float(out.loc[out["date"] == "2026-01-05", "adj_factor"].iloc[0]) == 1.0

    nxt = _bars(symbol=["SH600000"], date=["2026-01-07"], adj_factor=[80.0])
    out2 = sanitize_market_bars(nxt, prev_adj=pd.Series({"SH600000": 10.0}))
    assert out2["adj_factor"].isna().all()

    ok = sanitize_market_bars(nxt, prev_adj=pd.Series({"SH600000": 79.0}))
    assert float(ok["adj_factor"].iloc[0]) == 80.0


def test_ohlc_cross_vol_inf_and_dedup():
    frame = _bars(
        symbol=["SH600000", "SH600000", "SZ000001"],
        date=["2026-01-06", "2026-01-06", "2026-01-06"],
        open=[10.0, 10.0, 5.0],
        high=[9.0, 11.0, 5.2],
        low=[9.5, 9.8, 4.9],
        close=[10.5, 10.2, np.inf],
        vol=[0.0, 100.0, 10.0],
        amount=[1.0, 1000.0, -3.0],
        adj_factor=[1.0, 1.0, 0.0],
    )
    out = sanitize_market_bars(frame)
    assert len(out) == 2
    kept = out.set_index("symbol")
    sh = kept.loc["SH600000"]
    assert float(sh["high"]) == 11.0
    assert float(sh["vol"]) == 100.0
    sz = kept.loc["SZ000001"]
    assert pd.isna(sz["close"])
    assert pd.isna(sz["amount"])
    assert pd.isna(sz["adj_factor"])


def test_fill_limits_by_board_date_and_keep_official():
    frame = _bars(
        symbol=["SH600000", "SZ300001", "SZ300001", "SZ302132", "SH688001", "BJ430047"],
        date=[
            "2019-06-03",
            "2019-06-03",
            "2020-08-24",
            "2020-08-24",
            "2020-08-24",
            "2022-01-10",
        ],
        pre_close=[10.0, 10.0, 10.0, 10.0, 20.0, 8.0],
        up_limit=[11.0, np.nan, np.nan, np.nan, np.nan, np.nan],
        down_limit=[9.0, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    out = sanitize_market_bars(frame).set_index(["symbol", "date"])
    main = out.loc[("SH600000", "2019-06-03")]
    assert float(main["up_limit"]) == 11.0
    assert float(main["down_limit"]) == 9.0
    gem_old = out.loc[("SZ300001", "2019-06-03")]
    assert float(gem_old["up_limit"]) == 11.0
    assert float(gem_old["down_limit"]) == 9.0
    gem_new = out.loc[("SZ300001", "2020-08-24")]
    assert float(gem_new["up_limit"]) == 12.0
    assert float(gem_new["down_limit"]) == 8.0
    gem_302 = out.loc[("SZ302132", "2020-08-24")]
    assert float(gem_302["up_limit"]) == 12.0
    star = out.loc[("SH688001", "2020-08-24")]
    assert float(star["up_limit"]) == 24.0
    bse = out.loc[("BJ430047", "2022-01-10")]
    assert float(bse["up_limit"]) == 10.4
    assert float(bse["down_limit"]) == 5.6


def test_fill_limits_st_asof_and_ipo_nolimit():
    frame = _bars(
        symbol=["SZ000001", "SZ000001", "SH688001", "SZ300023", "SH600182"],
        date=["2020-01-06", "2021-01-06", "2020-07-24", "2021-09-06", "2019-06-03"],
        pre_close=[10.0, 10.0, 30.0, 10.0, 10.0],
    )
    namechange = pd.DataFrame(
        {
            "symbol": [
                "SZ000001",
                "SZ000001",
                "SZ300023",
                "SH600182",
                "SZ000002",
            ],
            "name": ["*ST平安", "平安银行", "*ST宝德", "S佳通", "万科退"],
            "start_date": [
                "2019-01-01",
                "2020-06-01",
                "2021-04-28",
                "2006-10-09",
                "2021-01-01",
            ],
            "end_date": ["2020-05-31", "", "2022-06-07", "2021-04-25", ""],
        }
    )
    list_dates = pd.Series({"SH688001": "2020-07-22"})
    out = sanitize_market_bars(
        frame, namechange=namechange, list_dates=list_dates
    ).set_index(["symbol", "date"])
    st = out.loc[("SZ000001", "2020-01-06")]
    assert float(st["up_limit"]) == 10.5
    assert float(st["down_limit"]) == 9.5
    later = out.loc[("SZ000001", "2021-01-06")]
    assert float(later["up_limit"]) == 11.0
    ipo = out.loc[("SH688001", "2020-07-24")]
    assert pd.isna(ipo["up_limit"])
    assert pd.isna(ipo["down_limit"])
    gem_st = out.loc[("SZ300023", "2021-09-06")]
    assert float(gem_st["up_limit"]) == 12.0
    s_share = out.loc[("SH600182", "2019-06-03")]
    assert float(s_share["up_limit"]) == 10.5


def test_fill_limits_gem_approval_first_day_44():
    frame = _bars(symbol=["SZ300001"], date=["2018-06-01"], pre_close=[10.0])
    list_dates = pd.Series({"SZ300001": "2018-06-01"})
    out = sanitize_market_bars(frame, list_dates=list_dates).iloc[0]
    assert float(out["up_limit"]) == 14.4
    assert float(out["down_limit"]) == 6.4


def test_fill_limits_delist_uses_board_not_five_percent():
    frame = _bars(symbol=["SZ000002"], date=["2021-06-01"], pre_close=[10.0])
    namechange = pd.DataFrame(
        {
            "symbol": ["SZ000002"],
            "name": ["万科退"],
            "start_date": ["2021-01-01"],
            "end_date": [""],
        }
    )
    out = sanitize_market_bars(frame, namechange=namechange).iloc[0]
    assert float(out["up_limit"]) == 11.0
    assert float(out["down_limit"]) == 9.0


def test_fill_limits_main_ipo_first_day_and_reg_nolimit():
    frame = _bars(
        symbol=["SZ001215", "SZ001215", "SH601001", "SH601001"],
        date=["2021-09-06", "2021-09-07", "2023-04-10", "2023-04-18"],
        pre_close=[15.71, 15.71, 10.0, 10.0],
    )
    list_dates = pd.Series({"SZ001215": "2021-09-06", "SH601001": "2023-04-10"})
    calendar = [
        "2021-09-06",
        "2021-09-07",
        "2023-04-10",
        "2023-04-11",
        "2023-04-12",
        "2023-04-13",
        "2023-04-14",
        "2023-04-17",
        "2023-04-18",
    ]
    out = sanitize_market_bars(
        frame, list_dates=list_dates, calendar=calendar
    ).set_index(["symbol", "date"])
    first = out.loc[("SZ001215", "2021-09-06")]
    assert float(first["up_limit"]) == 22.62
    assert float(first["down_limit"]) == 10.05
    second = out.loc[("SZ001215", "2021-09-07")]
    assert float(second["up_limit"]) == 17.28
    assert float(second["down_limit"]) == 14.14
    reg_early = out.loc[("SH601001", "2023-04-10")]
    assert pd.isna(reg_early["up_limit"])
    reg_later = out.loc[("SH601001", "2023-04-18")]
    assert float(reg_later["up_limit"]) == 11.0


def test_fill_limits_skips_index_and_missing_preclose():
    frame = _bars(
        symbol=["index_SH000300", "SH600000"],
        date=["2020-08-24", "2020-08-24"],
        pre_close=[4000.0, np.nan],
        close=[4010.0, 10.0],
    )
    out = sanitize_market_bars(frame)
    index = out.loc[out["symbol"] == "index_SH000300"].iloc[0]
    assert "up_limit" not in out.columns or pd.isna(index.get("up_limit", np.nan))
    stock = out.loc[out["symbol"] == "SH600000"].iloc[0]
    assert "up_limit" not in out.columns or pd.isna(stock.get("up_limit", np.nan))
