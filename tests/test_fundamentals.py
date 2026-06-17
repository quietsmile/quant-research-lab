"""财务库 point-in-time 逻辑单测（构造数据，不依赖网络）。

重点守住"防前视"：point_in_time 绝不返回 as_of 之后才公告的财报；
真实公告日缺失时用法定披露截止日兜底（保守，不前视）。
"""
import pandas as pd
import pytest

import quantlab.data.fundamentals as fund


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fund, "FUND_DIR", tmp_path)
    monkeypatch.setattr(fund, "PERFORMANCE_FILE", tmp_path / "performance.parquet")
    monkeypatch.setattr(fund, "ANNOUNCE_FILE", tmp_path / "announce.parquet")
    # 财务数字（无公告日）
    perf = pd.DataFrame({
        "symbol": ["600519", "600519", "000001", "000001"],
        "name": ["茅台", "茅台", "平安", "平安"],
        "net_profit": [100, 120, 50, 55],
        "report_period": pd.to_datetime(["2023-12-31", "2024-03-31",
                                         "2023-12-31", "2024-03-31"]),
    })
    perf.to_parquet(tmp_path / "performance.parquet", index=False)
    return tmp_path


def _set_announce(tmp_path, rows):
    pd.DataFrame(rows).to_parquet(tmp_path / "announce.parquet", index=False)


def test_pit_uses_real_announce_dates(store):
    # 茅台真实公告日：年报 2024-04-02，一季报 2024-04-28
    _set_announce(store, {
        "symbol": ["600519", "600519"],
        "report_period": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        "announce_date": pd.to_datetime(["2024-04-02", "2024-04-28"]),
    })
    pit = fund.point_in_time("2024-04-10").set_index("symbol")
    # 4/10：年报已出、一季报未出 → 拿到的是年报，且 announce_is_real
    assert pit.loc["600519", "report_period"] == pd.Timestamp("2023-12-31")
    assert bool(pit.loc["600519", "announce_is_real"]) is True


def test_pit_picks_latest_announced(store):
    _set_announce(store, {
        "symbol": ["600519", "600519"],
        "report_period": pd.to_datetime(["2023-12-31", "2024-03-31"]),
        "announce_date": pd.to_datetime(["2024-04-02", "2024-04-28"]),
    })
    pit = fund.point_in_time("2024-05-01").set_index("symbol")
    assert pit.loc["600519", "report_period"] == pd.Timestamp("2024-03-31")
    assert pit.loc["600519", "net_profit"] == 120


def test_statutory_fallback_when_no_real_date(store):
    # 平安无真实公告日 → 用法定截止日：年报次年4-30，一季报当年4-30
    _set_announce(store, {"symbol": [], "report_period": pd.to_datetime([]),
                          "announce_date": pd.to_datetime([])})
    # 2024-04-15：年报兜底日=2024-04-30(未到)，所以连年报都还不算可用
    pit = fund.point_in_time("2024-04-15")
    assert pit.empty or "000001" not in set(pit["symbol"])
    # 2024-05-01：年报(兜底2024-04-30)与一季报(兜底2024-04-30)都可用 → 取最新一季报
    pit2 = fund.point_in_time("2024-05-01").set_index("symbol")
    assert pit2.loc["000001", "report_period"] == pd.Timestamp("2024-03-31")
    assert bool(pit2.loc["000001", "announce_is_real"]) is False


def test_statutory_deadline_values():
    assert fund.statutory_deadline(pd.Timestamp("2023-03-31")) == pd.Timestamp("2023-04-30")
    assert fund.statutory_deadline(pd.Timestamp("2023-06-30")) == pd.Timestamp("2023-08-31")
    assert fund.statutory_deadline(pd.Timestamp("2023-09-30")) == pd.Timestamp("2023-10-31")
    assert fund.statutory_deadline(pd.Timestamp("2023-12-31")) == pd.Timestamp("2024-04-30")


def test_parse_period():
    assert fund._parse_period("2023年年度报告") == pd.Timestamp("2023-12-31")
    assert fund._parse_period("2023年半年度报告") == pd.Timestamp("2023-06-30")
    assert fund._parse_period("2023年第一季度报告") == pd.Timestamp("2023-03-31")
    assert fund._parse_period("2023年第三季度报告") == pd.Timestamp("2023-09-30")
    assert fund._parse_period("关于召开股东大会的通知") is None
    assert fund._parse_period("0201年年度报告") is None   # 离谱年份护栏，防越界日期


def test_quarter_ends():
    qs = fund.quarter_ends(2023, 2024)
    assert qs[0] == "20230331" and qs[-1] == "20241231" and len(qs) == 8
