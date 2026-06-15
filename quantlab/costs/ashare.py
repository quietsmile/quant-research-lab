"""A 股交易成本模型。

默认参数（2026 年常见水平，可被覆盖）：
- 印花税：**仅卖出**收取，成交额 0.05%（2023-08 起由 0.1% 下调至 0.05%）。
- 佣金：双边，成交额 0.025%，单笔最低 5 元。
- 过户费：双边，成交额 0.001%（沪深均已统一按成交额计）。
- 滑点：双边，成交额 0.05%（市价撮合的保守估计）。

制度摩擦：
- **T+1**：当日买入不可当日卖出（在回测引擎里强制）。
- **涨跌停**：触及涨停无法买入、触及跌停无法卖出（撮合可行性检查）。

这些默认值是**可引用、可比较**的基准，不是真理；实盘请按自己券商
费率与标的调整。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TradeCost:
    """单笔交易的成本拆解（金额，单位：元）。"""

    commission: float
    stamp_tax: float
    transfer_fee: float
    slippage: float

    @property
    def total(self) -> float:
        return self.commission + self.stamp_tax + self.transfer_fee + self.slippage


@dataclass
class AShareCostModel:
    """A 股摩擦成本模型。

    费率均以**小数**表示（0.0005 = 0.05% = 5bp）。
    """

    stamp_tax_rate: float = 0.0005       # 印花税，仅卖出
    commission_rate: float = 0.00025     # 佣金，双边
    min_commission: float = 5.0          # 单笔最低佣金（元）
    transfer_fee_rate: float = 0.00001   # 过户费，双边
    slippage_rate: float = 0.0005        # 滑点，双边
    price_limit_pct: float = 0.10        # 涨跌停幅度（主板 10%）

    def __post_init__(self) -> None:
        for name in (
            "stamp_tax_rate", "commission_rate", "transfer_fee_rate",
            "slippage_rate", "price_limit_pct",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负")
        if self.min_commission < 0:
            raise ValueError("min_commission 不能为负")

    def cost(self, *, amount: float, side: str) -> TradeCost:
        """计算单笔成交的各项成本。

        Parameters
        ----------
        amount : float
            成交金额（价格 × 股数），非负。
        side : {"buy", "sell"}
        """
        if amount < 0:
            raise ValueError("成交金额不能为负")
        if side not in {"buy", "sell"}:
            raise ValueError("side 必须是 'buy' 或 'sell'")

        commission = max(amount * self.commission_rate, self.min_commission) if amount > 0 else 0.0
        stamp = amount * self.stamp_tax_rate if side == "sell" else 0.0
        transfer = amount * self.transfer_fee_rate
        slip = amount * self.slippage_rate
        return TradeCost(commission=commission, stamp_tax=stamp, transfer_fee=transfer, slippage=slip)

    def round_trip_cost_rate(self) -> float:
        """一次完整买卖（不含最低佣金、不含滑点起跳）的成本率估计。

        用于快速判断策略换手能不能扛得住成本：买入成本率 + 卖出成本率。
        """
        buy = self.commission_rate + self.transfer_fee_rate + self.slippage_rate
        sell = self.commission_rate + self.transfer_fee_rate + self.slippage_rate + self.stamp_tax_rate
        return buy + sell

    def hit_upper_limit(self, prev_close: float, price: float) -> bool:
        """price 是否触及涨停（无法买入）。"""
        if prev_close <= 0:
            return False
        return price >= prev_close * (1 + self.price_limit_pct) - 1e-9

    def hit_lower_limit(self, prev_close: float, price: float) -> bool:
        """price 是否触及跌停（无法卖出）。"""
        if prev_close <= 0:
            return False
        return price <= prev_close * (1 - self.price_limit_pct) + 1e-9
