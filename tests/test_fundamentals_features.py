"""财务特征工程单测：重点验证"累计→单季"的正确性（错了所有因子都错）。"""
import numpy as np
import pandas as pd
import pytest

from quantlab.data import fundamentals_features as ff


def _cum_df():
    # 茅台风格累计净利润 + 两年数据，外加一只缺 Q2 的股票
    rows = [
        ("600519", "茅台", "2022-12-31", 627.0),
        ("600519", "茅台", "2023-03-31", 207.9),
        ("600519", "茅台", "2023-06-30", 359.8),
        ("600519", "茅台", "2023-09-30", 528.8),
        ("600519", "茅台", "2023-12-31", 747.3),
        ("000001", "平安", "2023-03-31", 100.0),
        ("000001", "平安", "2023-09-30", 300.0),  # 缺 Q2 → Q3 单季应 NaN
    ]
    df = pd.DataFrame(rows, columns=["symbol", "name", "report_period", "net_profit"])
    df["report_period"] = pd.to_datetime(df["report_period"])
    # 其余流量字段补齐避免 KeyError
    for c in ["revenue", "eps", "ocfps"]:
        df[c] = df["net_profit"]
    return df


def test_single_quarter_decumulation():
    sq = ff.to_single_quarter(_cum_df()).set_index(["symbol", "report_period"])
    np = sq["net_profit_q"]
    assert np.loc[("600519", "2023-03-31")] == pytest.approx(207.9)       # Q1=累计
    assert np.loc[("600519", "2023-06-30")] == pytest.approx(151.9)       # 359.8-207.9
    assert np.loc[("600519", "2023-09-30")] == pytest.approx(169.0)       # 528.8-359.8
    assert np.loc[("600519", "2023-12-31")] == pytest.approx(218.5)       # 747.3-528.8


def test_missing_quarter_gives_nan():
    sq = ff.to_single_quarter(_cum_df()).set_index(["symbol", "report_period"])
    # 平安缺 Q2，Q3 无法可靠拆季 → NaN（不能用 300-100）
    assert pd.isna(sq["net_profit_q"].loc[("000001", "2023-09-30")])


def test_ttm_equals_annual():
    feat = ff.add_ttm(ff.to_single_quarter(_cum_df())).set_index(["symbol", "report_period"])
    # 四个单季之和应等于年报累计 747.3
    assert feat["net_profit_ttm"].loc[("600519", "2023-12-31")] == pytest.approx(747.3)


def test_single_quarter_yoy():
    df = ff.to_single_quarter(_cum_df())
    df = ff.add_single_quarter_yoy(df).set_index(["symbol", "report_period"])
    # 茅台 2023Q1 单季 207.9 对 2022Q1——本例无 2022Q1 → NaN
    assert pd.isna(df["net_profit_q_yoy"].loc[("600519", "2023-03-31")])


def test_universe_flags():
    df = pd.DataFrame({
        "symbol": ["600519", "000004", "830799"],
        "name": ["茅台", "*ST国华", "某新三板"],
        "report_period": pd.to_datetime(["2023-03-31"] * 3),
        "net_profit": [1.0, 1.0, 1.0], "revenue": [1.0, 1.0, 1.0],
        "eps": [1.0, 1.0, 1.0], "ocfps": [1.0, 1.0, 1.0],
    })
    flagged = ff.add_universe_flags(df).set_index("symbol")
    assert bool(flagged.loc["600519", "is_a_share"]) is True
    assert bool(flagged.loc["000004", "is_st"]) is True       # *ST 识别
    assert bool(flagged.loc["830799", "is_a_share"]) is False  # 新三板剔除
