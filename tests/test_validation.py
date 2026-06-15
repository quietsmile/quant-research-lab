"""验证模块单测。"""
import pytest

from quantlab.strategies import MACrossStrategy
from quantlab.validation import train_test_split, walk_forward, overfitting_report
from quantlab.validation.walk_forward import WalkForwardResult
from quantlab.data.loader import _synthetic_prices


def _prices():
    return _synthetic_prices("600519", "2017-01-01", "2022-12-31")


def test_train_test_split_is_ordered_and_disjoint():
    prices = _prices()
    train, test = train_test_split(prices, 0.7)
    assert train.index[-1] < test.index[0]          # 时间不重叠、不打乱
    assert len(train) + len(test) == len(prices)


def test_train_test_split_validates_ratio():
    with pytest.raises(ValueError):
        train_test_split(_prices(), 1.5)


def test_walk_forward_runs_and_reports():
    prices = _prices()
    wf = walk_forward(prices, MACrossStrategy(10, 30), n_folds=5)
    assert isinstance(wf, WalkForwardResult)
    assert wf.n_folds >= 2
    assert "sharpe" in wf.folds.columns
    assert len(wf.oos_returns) > 0


def test_overfitting_report_is_string():
    prices = _prices()
    report = overfitting_report(prices, MACrossStrategy(10, 30))
    assert "过拟合诊断" in report
    assert "样本外夏普" in report
