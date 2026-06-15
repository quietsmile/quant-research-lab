"""回测引擎单测：重点验证"不骗自己"的几条规则。"""
import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import Backtester
from quantlab.costs import AShareCostModel
from quantlab.strategies import MACrossStrategy
from quantlab.strategies.base import Strategy
from quantlab.data.loader import _synthetic_prices


class AlwaysFlat(Strategy):
    name = "always_flat"

    def generate_signals(self, prices):
        return pd.Series(0.0, index=prices.index)


class AlwaysFull(Strategy):
    name = "always_full"

    def generate_signals(self, prices):
        return pd.Series(1.0, index=prices.index)


def _flat_market(n=50, price=10.0):
    idx = pd.bdate_range("2021-01-01", periods=n)
    df = pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price,
         "volume": 1_000_000.0},
        index=idx,
    )
    return df


def test_flat_strategy_preserves_cash():
    prices = _flat_market()
    bt = Backtester(initial_cash=1_000_000)
    res = bt.run(prices, AlwaysFlat())
    assert res.equity.iloc[-1] == pytest.approx(1_000_000)
    assert res.total_cost == 0.0
    assert len(res.trades) == 0


def test_buy_hold_stays_invested():
    from quantlab.strategies import BuyHoldStrategy
    prices = _synthetic_prices("600519", "2020-01-01", "2021-12-31")
    res = Backtester().run(prices, BuyHoldStrategy())
    # 首日建仓后基本满仓持有：实际仓位应长期接近 1
    assert res.positions.iloc[-1] > 0.9
    assert res.trades.iloc[0]["side"] == "buy"


def test_full_strategy_only_trades_once_in_flat_market():
    # 价格不动：买入一次后就一直满仓，不应反复交易
    prices = _flat_market()
    bt = Backtester(initial_cash=1_000_000)
    res = bt.run(prices, AlwaysFull())
    assert len(res.trades) == 1
    assert res.trades.iloc[0]["side"] == "buy"


def test_cost_reduces_equity_vs_zero_cost():
    prices = _synthetic_prices("600519", "2019-01-01", "2021-12-31")
    strat = MACrossStrategy(5, 20)
    free = AShareCostModel(stamp_tax_rate=0, commission_rate=0, min_commission=0,
                           transfer_fee_rate=0, slippage_rate=0)
    res_free = Backtester(cost_model=free).run(prices, strat)
    res_cost = Backtester(cost_model=AShareCostModel()).run(prices, strat)
    if len(res_cost.trades) > 0:
        assert res_cost.equity.iloc[-1] < res_free.equity.iloc[-1]
        assert res_cost.total_cost > 0


def test_no_lookahead_signal_is_shifted():
    # 构造价格在某日突然跳涨：若有前视，会在跳涨当日就满仓吃到涨幅。
    idx = pd.bdate_range("2021-01-01", periods=40)
    close = np.r_[np.full(20, 10.0), np.full(20, 20.0)]
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close,
         "volume": 1_000_000.0},
        index=idx,
    )

    class BuyOnJump(Strategy):
        name = "buy_on_jump"
        def generate_signals(self, prices):
            # t 日看到收盘=20 才发出满仓信号
            return (prices["close"] >= 20).astype(float)

    bt = Backtester(cost_model=AShareCostModel(stamp_tax_rate=0, commission_rate=0,
                    min_commission=0, transfer_fee_rate=0, slippage_rate=0))
    res = bt.run(df, BuyOnJump())
    # 信号被 shift 一日，最早在跳涨"次日"开盘(=20)买入，买不到 10->20 的涨幅
    first_trade = res.trades.iloc[0]
    assert first_trade["price"] == pytest.approx(20.0)


def test_price_limit_blocks_buy():
    # 开盘价相对昨收涨停 -> 买单被拒
    idx = pd.bdate_range("2021-01-01", periods=5)
    df = pd.DataFrame(
        {
            "open": [10.0, 11.0, 11.0, 11.0, 11.0],   # 第2日开盘=11 = 昨收10*1.1 涨停
            "high": [10.0, 11.0, 11.0, 11.0, 11.0],
            "low":  [10.0, 11.0, 11.0, 11.0, 11.0],
            "close":[10.0, 11.0, 11.0, 11.0, 11.0],
            "volume": 1_000_000.0,
        },
        index=idx,
    )

    class BuyFromStart(Strategy):
        name = "buy_from_start"
        def generate_signals(self, prices):
            return pd.Series(1.0, index=prices.index)

    bt = Backtester()
    res = bt.run(df, BuyFromStart())
    # 信号 shift 后第2日开盘尝试买入，但当日涨停 -> 被拒至少一次
    assert res.n_rejected >= 1


def test_short_series_raises():
    df = _flat_market(n=1)
    with pytest.raises(ValueError):
        Backtester().run(df, AlwaysFull())
