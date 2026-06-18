"""统一的 point-in-time 财务查询入口（默认优先 Tushare 库）。

数据源优先级（source="auto"）：
1. Tushare PIT 库（tushare_pit.parquet）——权威 f_ann_date、含归母/扣非，**首选**；
2. 回落到旧的 yjbb+cninfo 库（build_pit_table）——Tushare 库尚未生成时兜底。

显式指定 source="tushare" / "legacy" 可强制选源。
"""
from __future__ import annotations

import pandas as pd


def point_in_time(as_of, symbols=None, *, source: str = "auto",
                  max_stale_days: int | None = 365, list_gate: bool = True) -> pd.DataFrame:
    """返回截至 as_of 已公告的、每只股票最新一期财报（防前视）。

    Parameters
    ----------
    as_of : 日期
    symbols : 代码列表，可选
    source : {"auto", "tushare", "legacy"}
    max_stale_days : int | None
        陈旧度护栏：丢弃报告期距 as_of 超过该天数的记录（默认 365，杀掉停报/退市的
        死数据；正常"Q1 时用上年三季报"的~210 天滞后不受影响）。None 关闭。
    list_gate : bool
        list_date 门控：as_of 早于该股 A 股上市日则剔除（需 listing 缓存，缺则跳过）。
    """
    import pandas as pd
    from quantlab.data import tushare_adapter, fundamentals

    as_of = pd.Timestamp(as_of)
    if source == "tushare":
        snap = tushare_adapter.point_in_time(as_of, symbols)
    elif source == "legacy":
        snap = fundamentals.point_in_time(as_of, symbols)
    else:  # auto：优先 Tushare 库，缺失/异常则回落旧源
        snap = None
        try:
            if tushare_adapter.TS_FUND_FILE.exists():
                snap = tushare_adapter.point_in_time(as_of, symbols)
        except Exception:  # noqa: BLE001
            snap = None
        if snap is None:
            snap = fundamentals.point_in_time(as_of, symbols)

    if snap is None or snap.empty:
        return snap

    # ② 陈旧度护栏
    if max_stale_days is not None and "report_period" in snap.columns:
        stale = (as_of - snap["report_period"]).dt.days > max_stale_days
        snap = snap[~stale]

    # ③ list_date 门控（仅当有 listing 缓存）
    if list_gate:
        try:
            listing = tushare_adapter.load_listing()[["symbol", "list_date"]]
            snap = snap.merge(listing, on="symbol", how="left")
            keep = snap["list_date"].isna() | (as_of >= snap["list_date"])
            snap = snap[keep].drop(columns=["list_date"])
        except FileNotFoundError:
            pass  # 无上市日缓存则跳过门控

    return snap.reset_index(drop=True)


def active_source() -> str:
    """当前 auto 模式实际会用的源（便于确认是否已切到 Tushare）。"""
    from quantlab.data import tushare_adapter
    return "tushare" if tushare_adapter.TS_FUND_FILE.exists() else "legacy"
