"""按交易日拉全市场**日频**后复权收盘价，支持日频回测（含退市、无幸存者偏差）。

按日(daily + adj_factor)拉取，一次调用返回当日所有股票；存长表 parquet。

跑法：python examples/pull_daily_prices.py [起始YYYYMMDD] [结束YYYYMMDD]
"""
from __future__ import annotations

import socket
import sys
import time

socket.setdefaulttimeout(30)

import pandas as pd

from quantlab.data.tushare_adapter import get_pro, _FUND_DIR

DAILY_FILE = _FUND_DIR / "daily_prices.parquet"


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "20160101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20251231"
    pro = get_pro()
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    days = sorted(cal["cal_date"].tolist())
    print(f"日频价格：{len(days)} 个交易日 {start}~{end}", flush=True)

    frames = []
    for i, d in enumerate(days, 1):
        for attempt in range(3):
            try:
                px = pro.daily(trade_date=d, fields="ts_code,close")
                af = pro.adj_factor(trade_date=d, fields="ts_code,adj_factor")
                m = px.merge(af, on="ts_code", how="left")
                m["adj_close"] = m["close"] * m["adj_factor"].fillna(1.0)
                m["trade_date"] = d
                m["symbol"] = m["ts_code"].str[:6]
                frames.append(m[["trade_date", "symbol", "adj_close"]])
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        if i % 200 == 0:
            print(f"  {i}/{len(days)} ...", flush=True)
        time.sleep(0.12)

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    DAILY_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(DAILY_FILE, index=False)
    print(f"\n完成：{len(panel)} 行 / {panel.symbol.nunique()} 只 / "
          f"{panel.trade_date.min().date()}~{panel.trade_date.max().date()} → {DAILY_FILE}", flush=True)


if __name__ == "__main__":
    main()
