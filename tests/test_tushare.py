"""Tushare 适配器单测（代码映射离线；真实拉取走 network 标记）。"""
import pandas as pd
import pytest

from quantlab.data import tushare_adapter as tsa


def test_to_ts_code():
    assert tsa.to_ts_code("600519") == "600519.SH"   # 沪
    assert tsa.to_ts_code("000001") == "000001.SZ"   # 深
    assert tsa.to_ts_code("300750") == "300750.SZ"   # 创业板
    assert tsa.to_ts_code("688981") == "688981.SH"   # 科创板
    assert tsa.to_ts_code("830799") == "830799.BJ"   # 北交所
    assert tsa.to_ts_code("600519.SH") == "600519.SH"  # 已带后缀


@pytest.mark.network
def test_fundamentals_pit_live():
    df = tsa.fundamentals_pit(["600519"], "20230101", "20240601", sleep=0.2, verbose=False)
    assert len(df) >= 4
    assert {"symbol", "report_period", "announce_date", "net_profit", "profit_dedt"} <= set(df.columns)
    # 防前视核心字段非空
    assert df["announce_date"].notna().all()
    # 茅台 2023 年报实际公告日 2024-04-03
    annual = df[df["report_period"] == pd.Timestamp("2023-12-31")]
    assert annual["announce_date"].iloc[0] == pd.Timestamp("2024-04-03")
