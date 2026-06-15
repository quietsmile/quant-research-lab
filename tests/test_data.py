"""数据加载与清洗单测。"""
import numpy as np
import pandas as pd
import pytest

from quantlab.data import clean_prices
from quantlab.data.loader import _synthetic_prices, load_prices, _to_yahoo_symbol, _yahoo_prices


def test_yahoo_symbol_mapping():
    assert _to_yahoo_symbol("600519") == "600519.SS"   # 沪市主板
    assert _to_yahoo_symbol("688981") == "688981.SS"   # 科创板
    assert _to_yahoo_symbol("900957") == "900957.SS"   # 沪市 B 股
    assert _to_yahoo_symbol("000001") == "000001.SZ"   # 深市主板
    assert _to_yahoo_symbol("300750") == "300750.SZ"   # 创业板
    assert _to_yahoo_symbol("830799") == "830799.BJ"   # 北交所 8 开头
    assert _to_yahoo_symbol("920002") == "920002.BJ"   # 北交所 920 开头
    assert _to_yahoo_symbol("AAPL") == "AAPL"          # 美股原样
    assert _to_yahoo_symbol("600519.SS") == "600519.SS"  # 已含后缀


@pytest.mark.network
def test_yahoo_real_fetch():
    """联网时验证真实数据流（无网络则该用例可被 -m 'not network' 跳过）。"""
    df = _yahoo_prices("600519", "2023-01-01", "2023-03-31")
    assert len(df) > 30
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert df["close"].between(800, 3000).all()  # 茅台合理价区间


def test_synthetic_is_reproducible():
    a = _synthetic_prices("600519", "2020-01-01", "2020-12-31")
    b = _synthetic_prices("600519", "2020-01-01", "2020-12-31")
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_differs_by_symbol():
    a = _synthetic_prices("600519", "2020-01-01", "2020-12-31")
    b = _synthetic_prices("000001", "2020-01-01", "2020-12-31")
    assert not a["close"].equals(b["close"])


def test_synthetic_prices_positive():
    df = _synthetic_prices("600519", "2020-01-01", "2020-12-31")
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["high"] >= df["low"]).all()


def test_load_prices_synthetic_clean():
    df = load_prices("600519", "2021-01-01", "2021-06-30",
                     source="synthetic", use_cache=False)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.index.has_duplicates


def test_clean_drops_nonpositive_and_dupes():
    idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02", "2020-01-03"])
    df = pd.DataFrame(
        {
            "open": [10, 10, -1, 12],
            "high": [11, 11, 11, 13],
            "low": [9, 9, 9, 11],
            "close": [10.5, 10.5, 10.0, 12.5],
            "volume": [100, 100, 100, 100],
        },
        index=idx,
    )
    out = clean_prices(df)
    assert not out.index.has_duplicates
    assert (out[["open", "high", "low", "close"]] > 0).all().all()


def test_clean_requires_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError):
        clean_prices(df)
