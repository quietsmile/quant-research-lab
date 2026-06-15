"""Quant Research Lab — discipline-first quant research framework for A-shares.

设计哲学：先学会不骗自己，再谈收益。所有"赚钱"的功能都被
"防过拟合 / 真实成本 / 样本外验证"包着。
"""
from quantlab.data import load_prices
from quantlab.stats import metrics
from quantlab.costs import AShareCostModel
from quantlab.backtest import Backtester
from quantlab.strategies import MACrossStrategy, Strategy

__version__ = "0.1.0"

__all__ = [
    "load_prices",
    "metrics",
    "AShareCostModel",
    "Backtester",
    "MACrossStrategy",
    "Strategy",
    "__version__",
]
