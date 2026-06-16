"""唐奇安通道突破策略（趋势/动量）。

逻辑：收盘价突破**过去 N 日最高价** → 满仓；跌破**过去 M 日最低价** → 空仓；
其间保持原仓位。这是海龟交易法的核心，属趋势跟踪家族。

为什么有用（经济学依据）：价格创出阶段新高，往往意味着新信息被市场逐步
消化、趋势确立；行为上的**反应不足**让趋势在突破后延续一段。与均线交叉
相比，它对"震荡市反复穿越"更钝感（要真突破才动作），换手通常更低。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.strategies.base import Strategy


class DonchianBreakoutStrategy(Strategy):
    """唐奇安通道突破（多头/空仓）。"""

    def __init__(self, entry: int = 20, exit: int = 10):
        if entry < 2 or exit < 1:
            raise ValueError("entry 需 >=2，exit 需 >=1")
        self.entry = entry
        self.exit = exit
        self.name = f"Donchian({entry},{exit})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        # 过去 N/M 日的高低点（shift(1) 排除当日，避免用到当日信息）
        upper = prices["high"].rolling(self.entry, min_periods=self.entry).max().shift(1)
        lower = prices["low"].rolling(self.exit, min_periods=self.exit).min().shift(1)

        raw = pd.Series(np.nan, index=close.index)
        raw[close > upper] = 1.0   # 向上突破 → 入场
        raw[close < lower] = 0.0   # 向下跌破 → 离场
        signal = raw.ffill().fillna(0.0)   # 其余日保持上一状态
        return self._validate_signals(signal, prices)
