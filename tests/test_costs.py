"""A 股成本模型单测。"""
import pytest

from quantlab.costs import AShareCostModel


def test_stamp_tax_only_on_sell():
    m = AShareCostModel(min_commission=0.0)
    buy = m.cost(amount=100_000, side="buy")
    sell = m.cost(amount=100_000, side="sell")
    assert buy.stamp_tax == 0.0
    assert sell.stamp_tax == pytest.approx(100_000 * 0.0005)


def test_min_commission_floor():
    m = AShareCostModel()
    # 小额成交，佣金应被最低 5 元托底
    c = m.cost(amount=1000, side="buy")  # 0.025% * 1000 = 0.25 < 5
    assert c.commission == pytest.approx(5.0)


def test_commission_above_floor():
    m = AShareCostModel()
    c = m.cost(amount=1_000_000, side="buy")  # 0.025% * 1e6 = 250 > 5
    assert c.commission == pytest.approx(250.0)


def test_total_is_sum_of_parts():
    m = AShareCostModel()
    c = m.cost(amount=500_000, side="sell")
    assert c.total == pytest.approx(
        c.commission + c.stamp_tax + c.transfer_fee + c.slippage
    )


def test_round_trip_includes_stamp_once():
    m = AShareCostModel(min_commission=0.0)
    rt = m.round_trip_cost_rate()
    expected = 2 * (0.00025 + 0.00001 + 0.0005) + 0.0005  # 双边费用 + 单边印花税
    assert rt == pytest.approx(expected)


def test_price_limits():
    m = AShareCostModel(price_limit_pct=0.10)
    assert m.hit_upper_limit(prev_close=10.0, price=11.0) is True
    assert m.hit_upper_limit(prev_close=10.0, price=10.5) is False
    assert m.hit_lower_limit(prev_close=10.0, price=9.0) is True
    assert m.hit_lower_limit(prev_close=10.0, price=9.5) is False


def test_invalid_inputs():
    m = AShareCostModel()
    with pytest.raises(ValueError):
        m.cost(amount=-1, side="buy")
    with pytest.raises(ValueError):
        m.cost(amount=100, side="hold")
    with pytest.raises(ValueError):
        AShareCostModel(commission_rate=-0.1)
