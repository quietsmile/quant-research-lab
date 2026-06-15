"""策略层：可继承的策略基类 + 玩具策略。

对应 todo.txt 能力点：端到端跑通一个玩具策略。
"""
from quantlab.strategies.base import Strategy
from quantlab.strategies.ma_cross import MACrossStrategy

__all__ = ["Strategy", "MACrossStrategy"]
