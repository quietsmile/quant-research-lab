"""数据层：加载、清洗、缓存。

对应 todo.txt 能力点：Python 数据栈 + 数据源。

核心理念：数据源只是适配器。框架开箱即用 —— 没网/没 token 时自动
回落到**可复现的合成行情**，保证学习不被环境卡住。
"""
from quantlab.data.loader import load_prices, clear_cache
from quantlab.data.clean import clean_prices
from quantlab.data.offline import (
    load_offline, update_offline, list_offline, has_offline, OFFLINE_DIR,
)

__all__ = [
    "load_prices", "clear_cache", "clean_prices",
    "load_offline", "update_offline", "list_offline", "has_offline", "OFFLINE_DIR",
]
