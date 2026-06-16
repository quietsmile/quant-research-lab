"""策略信号正确性单测（不依赖网络，用构造行情验证逻辑）。"""
import numpy as np
import pandas as pd
import pytest

from quantlab.strategies import (
    DonchianBreakoutStrategy,
    BollingerReversionStrategy,
    MACrossStrategy,
)


def _ohlc(close):
    idx = pd.bdate_range("2021-01-01", periods=len(close))
    c = pd.Series(close, index=idx, dtype=float)
    return pd.DataFrame(
        {"open": c, "high": c, "low": c, "close": c, "volume": 1e6}, index=idx
    )


def test_signals_in_unit_range():
    prices = _ohlc(np.r_[np.linspace(10, 20, 60), np.linspace(20, 12, 60)])
    for strat in (MACrossStrategy(5, 20), DonchianBreakoutStrategy(20, 10),
                  BollingerReversionStrategy(20, 2.0)):
        sig = strat.generate_signals(prices)
        assert sig.between(0, 1).all()
        assert sig.index.equals(prices.index)


def test_donchian_enters_on_breakout():
    # 前 30 天在 10 附近窄幅，然后持续走高 → 应在突破后进场(=1)
    close = np.r_[np.full(30, 10.0), np.linspace(10.5, 25, 40)]
    prices = _ohlc(close)
    sig = DonchianBreakoutStrategy(entry=20, exit=10).generate_signals(prices)
    assert sig.iloc[-1] == 1.0           # 上行末段应持仓
    assert sig.iloc[:25].sum() == 0.0    # 窄幅期未突破，不应持仓


def test_donchian_exits_on_breakdown():
    close = np.r_[np.linspace(10, 25, 40), np.linspace(24, 8, 40)]
    prices = _ohlc(close)
    sig = DonchianBreakoutStrategy(entry=20, exit=10).generate_signals(prices)
    assert sig.iloc[-1] == 0.0           # 持续下跌末段应离场


def test_bollinger_buys_oversold():
    # 平稳后突然急跌 → 跌破下轨应买入(=1)
    close = np.r_[np.full(40, 20.0), np.array([15.0, 14.0, 13.0])]
    prices = _ohlc(close)
    sig = BollingerReversionStrategy(window=20, k=2.0).generate_signals(prices)
    assert sig.iloc[-1] == 1.0


def test_strategy_param_validation():
    with pytest.raises(ValueError):
        DonchianBreakoutStrategy(entry=1)
    with pytest.raises(ValueError):
        BollingerReversionStrategy(window=1)
    with pytest.raises(ValueError):
        BollingerReversionStrategy(window=20, k=0)
