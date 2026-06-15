"""均线交叉玩具策略。

逻辑：快线（fast 日均线）上穿慢线（slow 日均线）→ 满仓；下穿 → 空仓。

这是**经典且极易过拟合**的策略，正好用来演示 todo.txt 的 Gate 1：
能跑通，也能说清它为什么可能骗你（见 docs/strategy-assessment.md 与
validation 模块）。
"""
from __future__ import annotations

import pandas as pd

from quantlab.strategies.base import Strategy


class MACrossStrategy(Strategy):
    """双均线交叉（多头/空仓）。"""

    def __init__(self, fast: int = 10, slow: int = 30):
        if fast < 1 or slow < 1:
            raise ValueError("均线窗口必须 >= 1")
        if fast >= slow:
            raise ValueError("fast 必须小于 slow")
        self.fast = fast
        self.slow = slow
        self.name = f"MA({fast},{slow})"

    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        close = prices["close"]
        fast_ma = close.rolling(self.fast, min_periods=self.fast).mean()
        slow_ma = close.rolling(self.slow, min_periods=self.slow).mean()

        # 快线在慢线之上则持仓。注意：信号基于 t 日收盘信息，
        # 由回测引擎错位到 t+1 执行，这里不做 shift。
        signal = (fast_ma > slow_ma).astype(float)
        return self._validate_signals(signal, prices)
