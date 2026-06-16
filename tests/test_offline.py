"""离线数据仓单测（用临时目录，不碰真实仓）。"""
import pandas as pd
import pytest

import quantlab.data.offline as offline
from quantlab.data.loader import _synthetic_prices


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(offline, "OFFLINE_DIR", tmp_path)
    return tmp_path


def test_save_load_roundtrip(tmp_store):
    df = _synthetic_prices("600519", "2020-01-01", "2020-12-31")
    offline.save_offline("600519", df)
    assert offline.has_offline("600519")
    back = offline.load_offline("600519")
    assert len(back) == len(df)
    assert list(back.columns) == ["open", "high", "low", "close", "volume"]


def test_load_slice(tmp_store):
    df = _synthetic_prices("600519", "2020-01-01", "2021-12-31")
    offline.save_offline("600519", df)
    sliced = offline.load_offline("600519", "2021-01-01", "2021-06-30")
    assert sliced.index.min() >= pd.Timestamp("2021-01-01")
    assert sliced.index.max() <= pd.Timestamp("2021-06-30")


def test_incremental_update_dedupes(tmp_store):
    a = _synthetic_prices("600519", "2020-01-01", "2020-06-30")
    b = _synthetic_prices("600519", "2020-04-01", "2020-12-31")  # 与 a 有重叠
    offline.update_offline("600519", a)
    merged = offline.update_offline("600519", b)
    assert not merged.index.has_duplicates
    assert merged.index.is_monotonic_increasing
    assert merged.index.max() >= pd.Timestamp("2020-12-01")


def test_missing_raises(tmp_store):
    with pytest.raises(FileNotFoundError):
        offline.load_offline("999999")


def test_list_offline(tmp_store):
    offline.save_offline("600519", _synthetic_prices("600519", "2020-01-01", "2020-12-31"))
    offline.save_offline("000858", _synthetic_prices("000858", "2020-01-01", "2020-12-31"))
    inv = offline.list_offline()
    assert set(inv["symbol"]) == {"600519", "000858"}
    assert (inv["n_days"] > 0).all()
