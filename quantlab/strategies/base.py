"""策略基类。

一个策略要做的事很简单：给定行情，产出**目标持仓**信号序列。
为避免前视偏差，约定信号在 t 日收盘后生成，t+1 日执行（引擎负责错位）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    """策略抽象基类。

    子类实现 :meth:`generate_signals`，返回一条与行情索引对齐的
    **目标仓位**序列，取值 [0, 1]：0 = 空仓，1 = 满仓。
    （本框架阶段 1 只做单标的多头，不做做空/杠杆。）
    """

    name: str = "strategy"

    @abstractmethod
    def generate_signals(self, prices: pd.DataFrame) -> pd.Series:
        """根据行情生成目标仓位序列（索引与 prices 对齐，值 ∈ [0,1]）。"""

    def _validate_signals(self, signals: pd.Series, prices: pd.DataFrame) -> pd.Series:
        """校验并规整信号：对齐索引、裁剪到 [0,1]、补 0。"""
        signals = signals.reindex(prices.index)
        signals = signals.fillna(0.0).clip(0.0, 1.0)
        return signals
