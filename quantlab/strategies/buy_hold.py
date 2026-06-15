"""买入持有基准策略。

它本身不产生择时 alpha，但它是**最重要的对照组**：任何择时策略都必须
先回答"我凭什么比一直拿着更好？"。买入持有的经济学依据是**股权风险
溢价**——长期承担权益风险理应获得补偿。
"""
from __future__ import annotations

import pandas as pd

from quantlab.strategies.base import Strategy


class BuyHoldStrategy(Strategy):
    """第一天满仓，之后一直持有。"""

    name = "BuyHold"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=prices.index)
