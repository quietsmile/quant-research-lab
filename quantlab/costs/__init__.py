"""交易成本与 A 股摩擦建模。

对应 todo.txt 能力点（核心）：把印花税、佣金、滑点、T+1、涨跌停撮合
假设写进回测。
"""
from quantlab.costs.ashare import AShareCostModel, TradeCost

__all__ = ["AShareCostModel", "TradeCost"]
