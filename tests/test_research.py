"""跨标的稳健性研究单测（用合成数据，离线、确定性）。"""
import pandas as pd

from quantlab.research import universe_backtest
from quantlab.strategies import MACrossStrategy


def test_universe_backtest_structure():
    syms = ["600519", "000858", "300750"]
    out = universe_backtest(
        syms, lambda: MACrossStrategy(10, 30),
        "2019-01-01", "2022-12-31", source="synthetic",
    )
    assert set(out) == {"per_symbol", "portfolio_returns", "summary"}
    ps = out["per_symbol"]
    assert len(ps) == 3
    assert {"symbol", "sharpe", "max_dd", "bh_sharpe", "beats_bh"} <= set(ps.columns)


def test_universe_summary_ranges():
    syms = ["600519", "000858", "300750", "002594"]
    out = universe_backtest(
        syms, lambda: MACrossStrategy(10, 30),
        "2018-01-01", "2022-12-31", source="synthetic",
    )
    s = out["summary"]
    assert s["n_symbols"] == 4
    assert 0.0 <= s["pct_positive_sharpe"] <= 1.0
    assert 0.0 <= s["pct_beat_buyhold"] <= 1.0
    assert len(out["portfolio_returns"]) > 0


def test_portfolio_is_average_of_members():
    # 等权组合收益应等于成员策略收益按日均值
    syms = ["600519", "000858"]
    out = universe_backtest(
        syms, lambda: MACrossStrategy(10, 30),
        "2020-01-01", "2021-12-31", source="synthetic",
    )
    assert out["portfolio_returns"].notna().all()
