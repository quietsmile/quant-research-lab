"""回测层：强制扣成本、强制 T+1、考虑涨跌停撮合的日频回测引擎。

对应 todo.txt 能力点：回测。
"""
from quantlab.backtest.engine import Backtester, BacktestResult

__all__ = ["Backtester", "BacktestResult"]
