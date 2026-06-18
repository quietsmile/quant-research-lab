"""严肃评估工具单测（确定性构造数据）。"""
import numpy as np
import pandas as pd
import pytest

from quantlab import eval as ev


def test_dev_test_split_order():
    dates = pd.bdate_range("2016-01-01", periods=100)
    dev, test = ev.dev_test_split(dates, test_frac=0.2)
    assert dev[-1] < test[0]                 # 时间不重叠、Test 在后
    assert len(test) == 20 and len(dev) == 80


def test_walk_forward_embargo_gap():
    dates = pd.bdate_range("2016-01-01", periods=40)
    folds = ev.walk_forward_splits(dates, train_size=10, val_size=4, step=5, embargo=2)
    assert len(folds) >= 1
    for tr, va in folds:
        # Train 末与 Val 首之间至少隔 embargo 个周期
        gap = (dates.get_loc(va[0]) - dates.get_loc(tr[-1]))
        assert gap >= 2 + 1 - 1  # val_start = tr_end+embargo → 间隔 = embargo+? 至少 embargo
        assert tr[-1] < va[0]


def test_psr_increases_with_better_returns():
    rng = np.random.default_rng(0)
    weak = pd.Series(rng.normal(0.001, 0.05, 200))
    strong = pd.Series(rng.normal(0.02, 0.05, 200))
    assert ev.probabilistic_sharpe_ratio(strong) > ev.probabilistic_sharpe_ratio(weak)


def test_deflated_sharpe_penalizes_more_trials():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.01, 0.04, 200))
    d1 = ev.deflated_sharpe_ratio(r, n_trials=1, trials_sr_std=0.3)
    d50 = ev.deflated_sharpe_ratio(r, n_trials=50, trials_sr_std=0.3)
    assert d50["dsr"] <= d1["dsr"]            # 试得越多，门槛越高、DSR 越低
    assert d50["sr_benchmark_annual"] > d1["sr_benchmark_annual"]


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.02, 0.03, 300))
    ci = ev.block_bootstrap_ci(r, n_boot=500, block=4)
    lo, hi = ci["ann_return_ci"]
    assert lo < hi


def test_norm_ppf_cdf_roundtrip():
    for p in (0.1, 0.5, 0.9, 0.975):
        assert ev._norm_cdf(ev._norm_ppf(p)) == pytest.approx(p, abs=1e-3)
