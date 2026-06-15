"""统计指标单测：用解析可验证的输入锁定实现正确性。"""
import numpy as np
import pandas as pd
import pytest

from quantlab.stats import metrics


def test_cumulative_return_known():
    r = pd.Series([0.1, -0.1])  # 1.1 * 0.9 - 1 = -0.01
    assert metrics.cumulative_return(r) == pytest.approx(-0.01)


def test_annualized_volatility_scaling():
    r = pd.Series(np.r_[0.01, -0.01] * 50)
    vol_daily = r.std(ddof=1)
    assert metrics.annualized_volatility(r) == pytest.approx(vol_daily * np.sqrt(252))


def test_sharpe_zero_when_no_variance():
    r = pd.Series([0.001] * 10)
    assert metrics.sharpe_ratio(r) == 0.0  # std=0 -> 约定返回 0


def test_sharpe_sign_positive():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    assert metrics.sharpe_ratio(r) > 0


def test_max_drawdown_known():
    # 净值 1 -> 1.2 -> 0.6：最大回撤 = 0.6/1.2 - 1 = -0.5
    r = pd.Series([0.2, -0.5])
    assert metrics.max_drawdown(r) == pytest.approx(-0.5)


def test_max_drawdown_non_positive():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0, 0.02, 200))
    assert metrics.max_drawdown(r) <= 0


def test_linear_regression_recovers_params():
    x = pd.Series(np.linspace(-1, 1, 100))
    y = 0.5 + 2.0 * x  # 完美线性
    out = metrics.linear_regression(y, x)
    assert out["alpha"] == pytest.approx(0.5, abs=1e-9)
    assert out["beta"] == pytest.approx(2.0, abs=1e-9)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_correlation_perfect():
    a = pd.Series([1.0, 2, 3, 4])
    b = pd.Series([2.0, 4, 6, 8])
    assert metrics.correlation(a, b) == pytest.approx(1.0)


def test_empty_returns_safe():
    empty = pd.Series(dtype=float)
    assert metrics.cumulative_return(empty) == 0.0
    assert metrics.sharpe_ratio(empty) == 0.0
    assert metrics.max_drawdown(empty) == 0.0
