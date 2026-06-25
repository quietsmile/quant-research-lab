"""Barra 暴露分析单测(构造数据)。"""
import numpy as np, pandas as pd, pytest
from quantlab import barra


def test_ls_factor_direction():
    idx = pd.date_range("2020-01-01", periods=50)
    syms = [f"s{i}" for i in range(60)]
    char = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 60)), index=idx, columns=syms)
    rfwd = char * 0.01 + np.random.default_rng(1).normal(0, 0.001, (50, 60))  # 高特征→高收益
    f = barra.ls_factor(char, rfwd, +1)
    assert f.mean() > 0                       # 高减低应为正


def test_barra_exposure_recovers_beta():
    pytest.importorskip("statsmodels")
    idx = pd.date_range("2020-01-01", periods=200)
    rng = np.random.default_rng(2)
    F1 = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    F2 = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    strat = 0.8 * F1 - 0.3 * F2 + rng.normal(0, 0.001, 200)
    strat = pd.Series(strat, index=idx)
    b = barra.barra_exposure(strat, pd.DataFrame({"F1": F1, "F2": F2}))
    assert b["exposure"]["F1"] == pytest.approx(0.8, abs=0.1)
    assert b["exposure"]["F2"] == pytest.approx(-0.3, abs=0.1)
    assert b["r2"] > 0.9
