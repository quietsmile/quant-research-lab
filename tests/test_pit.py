"""统一 point_in_time 的陈旧度护栏 + list_date 门控单测（构造数据）。"""
import pandas as pd
import pytest

import quantlab.data.tushare_adapter as tsa
import quantlab.data.pit as pit


def _snap(as_of_rows):
    return pd.DataFrame(as_of_rows)


def test_staleness_guard(monkeypatch):
    # A 报告期新鲜(180天)，B 陈旧(2年) —— 都已公告
    snap = _snap([
        {"symbol": "600519", "report_period": pd.Timestamp("2019-09-30"),
         "announce_date": pd.Timestamp("2019-10-30"), "roe": 20.0},
        {"symbol": "688981", "report_period": pd.Timestamp("2017-09-30"),
         "announce_date": pd.Timestamp("2017-11-14"), "roe": 5.0},
    ])
    monkeypatch.setattr(tsa, "point_in_time", lambda a, s=None: snap)
    monkeypatch.setattr(tsa, "load_listing", lambda: (_ for _ in ()).throw(FileNotFoundError()))
    res = pit.point_in_time("2020-03-31", source="tushare", max_stale_days=365, list_gate=False)
    assert set(res["symbol"]) == {"600519"}        # 陈旧的 B 被丢

    res2 = pit.point_in_time("2020-03-31", source="tushare", max_stale_days=None, list_gate=False)
    assert set(res2["symbol"]) == {"600519", "688981"}   # 关闭护栏则保留


def test_list_date_gate(monkeypatch):
    snap = _snap([
        {"symbol": "600519", "report_period": pd.Timestamp("2019-12-31"),
         "announce_date": pd.Timestamp("2020-03-31"), "roe": 30.0},
        {"symbol": "688981", "report_period": pd.Timestamp("2019-12-31"),
         "announce_date": pd.Timestamp("2020-03-31"), "roe": 8.0},
    ])
    listing = pd.DataFrame({"symbol": ["600519", "688981"],
                            "list_date": [pd.Timestamp("2001-08-27"), pd.Timestamp("2020-07-16")]})
    monkeypatch.setattr(tsa, "point_in_time", lambda a, s=None: snap)
    monkeypatch.setattr(tsa, "load_listing", lambda: listing)
    # as_of 早于中芯上市日 → 中芯被门控
    res = pit.point_in_time("2020-03-31", source="tushare", max_stale_days=None, list_gate=True)
    assert set(res["symbol"]) == {"600519"}
    # as_of 晚于上市日 → 都保留
    res2 = pit.point_in_time("2021-03-31", source="tushare", max_stale_days=None, list_gate=True)
    assert set(res2["symbol"]) == {"600519", "688981"}


def test_delist_date_gate(monkeypatch):
    """退市股在退市后不应被纳入(但退市前在其上市期间应纳入)。"""
    snap = _snap([
        {"symbol": "600519", "report_period": pd.Timestamp("2023-12-31"),
         "announce_date": pd.Timestamp("2024-03-31"), "roe": 30.0},
        {"symbol": "000003", "report_period": pd.Timestamp("2023-12-31"),
         "announce_date": pd.Timestamp("2024-03-31"), "roe": 1.0},  # 已退市
    ])
    listing = pd.DataFrame({
        "symbol": ["600519", "000003"],
        "list_date": [pd.Timestamp("2001-08-27"), pd.Timestamp("1991-01-01")],
        "delist_date": [pd.NaT, pd.Timestamp("2024-04-26")],
    })
    monkeypatch.setattr(tsa, "point_in_time", lambda a, s=None: snap)
    monkeypatch.setattr(tsa, "load_listing", lambda: listing)
    # 退市日(2024-04-26)之后查询 → 000003 被排除
    res = pit.point_in_time("2024-06-01", source="tushare", max_stale_days=None, list_gate=True)
    assert set(res["symbol"]) == {"600519"}
    # 退市日之前查询 → 仍纳入(避免幸存者偏差)
    res2 = pit.point_in_time("2024-04-01", source="tushare", max_stale_days=None, list_gate=True)
    assert "000003" in set(res2["symbol"])
