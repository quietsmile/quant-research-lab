"""评估面板单测（构造数据）。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import report as rp


def _series(mu, sigma, n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_report_keys_and_signs():
    r = _series(0.01, 0.03)
    m = rp.performance_report(r, periods=12)
    for k in ["cagr", "max_drawdown", "sharpe", "sortino", "calmar", "skew",
              "t_stat", "psr_vs0", "pct_positive_years", "max_dd_periods"]:
        assert k in m
    assert m["max_drawdown"] <= 0
    assert m["sharpe"] > 0                       # 正漂移 → 正夏普
    assert 0 <= m["psr_vs0"] <= 1


def test_calmar_and_dd():
    r = pd.Series([0.1, -0.5, 0.2],
                  index=pd.date_range("2020-01-31", periods=3, freq="ME"))
    m = rp.performance_report(r, periods=12)
    assert m["max_drawdown"] == pytest.approx(-0.5)   # 1.1→0.55 回撤 50%


def test_info_ratio_with_benchmark():
    r = _series(0.012, 0.03, seed=1)
    b = _series(0.004, 0.03, seed=2)
    m = rp.performance_report(r, periods=12, benchmark=b)
    assert "info_ratio" in m and "excess_cagr" in m


def test_trade_stats_expectancy():
    # 胜率40%但盈亏比3:1 → 期望为正
    t = pd.Series([3, 3, -1, -1, -1] * 20, dtype=float)
    s = rp.trade_stats(t)
    assert s["win_rate"] == pytest.approx(0.4)
    assert s["expectancy"] > 0
    assert s["profit_factor"] > 1


def test_format_runs():
    m = rp.performance_report(_series(0.01, 0.03), periods=12)
    assert "风险调整" in rp.format_report(m)
