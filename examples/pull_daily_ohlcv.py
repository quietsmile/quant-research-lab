"""按交易日拉全市场日频 **OHLCV + 成交额 + 换手率 + 量比**（含退市，无幸存者偏差）。

扩展原来"仅 close"的日频库，支持量价因子(换手/量比/Amihud非流动性/振幅/量价背离)。
daily(OHLCV+vol+amount) + adj_factor(复权) + daily_basic(turnover_rate/volume_ratio)，
按交易日各一次调用。存长表 daily_ohlcv.parquet。

跑法：python examples/pull_daily_ohlcv.py [起始YYYYMMDD] [结束]
"""
from __future__ import annotations
import socket, sys, time
socket.setdefaulttimeout(30)
import pandas as pd
from quantlab.data.tushare_adapter import get_pro, _FUND_DIR

OHLCV_FILE = _FUND_DIR / "daily_ohlcv.parquet"


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20160101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20251231"
    pro = get_pro()
    days = sorted(pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")["cal_date"].tolist())
    print(f"日频OHLCV：{len(days)} 个交易日 {start}~{end}", flush=True)
    frames = []
    for i, d in enumerate(days, 1):
        for attempt in range(3):
            try:
                px = pro.daily(trade_date=d, fields="ts_code,open,high,low,close,vol,amount")
                af = pro.adj_factor(trade_date=d, fields="ts_code,adj_factor")
                db = pro.daily_basic(trade_date=d, fields="ts_code,turnover_rate,volume_ratio")
                m = px.merge(af, on="ts_code", how="left").merge(db, on="ts_code", how="left")
                f = m["adj_factor"].fillna(1.0)
                for c in ("open", "high", "low", "close"):
                    m["adj_" + c] = m[c] * f                       # 复权 OHLC
                m["trade_date"] = d; m["symbol"] = m["ts_code"].str[:6]
                frames.append(m[["trade_date", "symbol", "adj_open", "adj_high", "adj_low",
                                 "adj_close", "vol", "amount", "turnover_rate", "volume_ratio"]])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if i % 200 == 0:
            print(f"  {i}/{len(days)} ...", flush=True)
        time.sleep(0.12)
    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    _FUND_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OHLCV_FILE, index=False)
    print(f"\n完成：{len(panel)} 行 / {panel.symbol.nunique()} 只 / "
          f"{panel.trade_date.min().date()}~{panel.trade_date.max().date()} → {OHLCV_FILE}", flush=True)


if __name__ == "__main__":
    main()
