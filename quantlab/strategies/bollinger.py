"""布林带均值回归策略（与趋势跟踪相反的一类）。

逻辑：收盘价跌破**下轨**（均线 - k×标准差）→ 视为超跌、买入；价格回到
**中轨**（均线）→ 离场。其间保持持仓。

为什么有用（经济学依据）：短期价格常因情绪/流动性冲击**过度反应**，偏离
基本面后倾向于回归均值。均值回归在**震荡/区间**行情里有效，在**强趋势**单
边行情里会吃亏（过早抄底、反复止损）——这恰好与趋势跟踪互补，也是教学上
最重要的对照：没有"万能策略"，只有"匹配市场状态的策略"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


class BollingerReversionStrategy(Strategy):
    """布林带均值回归（多头/空仓）。"""

    def __init__(self, window: int = 20, k: float = 2.0):
        if window < 2:
            raise ValueError("window 需 >=2")
        if k <= 0:
            raise ValueError("k 需 >0")
        self.window = window
        self.k = k
        self.name = f"Bollinger({window},{k})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        mid = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std(ddof=1)
        lower = mid - self.k * std

        raw = pd.Series(np.nan, index=close.index)
        raw[close < lower] = 1.0   # 跌破下轨 → 超跌买入
        raw[close > mid] = 0.0      # 回到中轨上方 → 离场
        signal = raw.ffill().fillna(0.0)
        return self._validate_signals(signal, prices)
