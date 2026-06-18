"""统一的 point-in-time 财务查询入口（默认优先 Tushare 库）。

数据源优先级（source="auto"）：
1. Tushare PIT 库（tushare_pit.parquet）——权威 f_ann_date、含归母/扣非，**首选**；
2. 回落到旧的 yjbb+cninfo 库（build_pit_table）——Tushare 库尚未生成时兜底。

显式指定 source="tushare" / "legacy" 可强制选源。
"""
from __future__ import annotations

import pandas as pd


def point_in_time(as_of, symbols=None, *, source: str = "auto") -> pd.DataFrame:
    """返回截至 as_of 已公告的、每只股票最新一期财报（防前视）。

    Parameters
    ----------
    as_of : 日期
    symbols : 代码列表，可选
    source : {"auto", "tushare", "legacy"}
    """
    from quantlab.data import tushare_adapter, fundamentals

    if source == "tushare":
        return tushare_adapter.point_in_time(as_of, symbols)
    if source == "legacy":
        return fundamentals.point_in_time(as_of, symbols)
    # auto：优先 Tushare 库，缺失/异常则回落旧源
    try:
        if tushare_adapter.TS_FUND_FILE.exists():
            return tushare_adapter.point_in_time(as_of, symbols)
    except Exception:  # noqa: BLE001
        pass
    return fundamentals.point_in_time(as_of, symbols)


def active_source() -> str:
    """当前 auto 模式实际会用的源（便于确认是否已切到 Tushare）。"""
    from quantlab.data import tushare_adapter
    return "tushare" if tushare_adapter.TS_FUND_FILE.exists() else "legacy"
