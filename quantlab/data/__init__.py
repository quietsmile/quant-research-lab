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
from quantlab.data.fundamentals import (
    build_pit_table, load_performance, statutory_deadline,
)
# 默认 point_in_time 走统一调度器（优先 Tushare 库，缺则回落旧源）
from quantlab.data.pit import point_in_time, active_source

__all__ = [
    "load_prices", "clear_cache", "clean_prices",
    "load_offline", "update_offline", "list_offline", "has_offline", "OFFLINE_DIR",
    # 财务数据（point-in-time / 按公告日，防前视）；point_in_time 默认优先 Tushare
    "point_in_time", "active_source", "build_pit_table", "load_performance", "statutory_deadline",
]
