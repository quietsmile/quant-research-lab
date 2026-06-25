"""增量补全日频 OHLCV 库：只拉现有文件之后缺失的交易日，merge 回写（不破坏历史）。

daily_ohlcv.parquet 之前只到 2025-12-31，导致 2026 回测无开/高/低价、一笔都建不了仓、
净值横线。本脚本从现有最后日期 +1 拉到指定结束日（默认今天），dedup 后存回。

跑法：python examples/update_daily_ohlcv.py [结束YYYYMMDD]
"""
from __future__ import annotations
import socket, sys, time
socket.setdefaulttimeout(30)
import pandas as pd
from quantlab.data.tushare_adapter import get_pro, _FUND_DIR

OHLCV_FILE = _FUND_DIR / "daily_ohlcv.parquet"


def main():
    old = pd.read_parquet(OHLCV_FILE)
    last = pd.to_datetime(old["trade_date"]).max()
    start = (last + pd.Timedelta(days=1)).strftime("%Y%m%d")
    end = sys.argv[1] if len(sys.argv) > 1 else pd.Timestamp.now().strftime("%Y%m%d")
    pro = get_pro()
    days = sorted(pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")["cal_date"].tolist())
    if not days:
        print(f"已是最新（最后 {last.date()}），无需补全。"); return
    print(f"增量补全：{len(days)} 个交易日 {start}~{end}（现有止于 {last.date()}）", flush=True)
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
                    m["adj_" + c] = m[c] * f
                m["trade_date"] = d; m["symbol"] = m["ts_code"].str[:6]
                frames.append(m[["trade_date", "symbol", "adj_open", "adj_high", "adj_low",
                                 "adj_close", "vol", "amount", "turnover_rate", "volume_ratio"]])
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if i % 50 == 0:
            print(f"  {i}/{len(days)} ...", flush=True)
        time.sleep(0.12)
    new = pd.concat(frames, ignore_index=True)
    new["trade_date"] = pd.to_datetime(new["trade_date"], format="%Y%m%d")
    panel = pd.concat([old, new], ignore_index=True).drop_duplicates(["trade_date", "symbol"], keep="last")
    panel = panel.sort_values(["trade_date", "symbol"]).reset_index(drop=True)
    panel.to_parquet(OHLCV_FILE, index=False)
    print(f"\n完成：+{len(new)} 行 → 总 {len(panel)} 行 / "
          f"{panel.trade_date.min().date()}~{panel.trade_date.max().date()} → {OHLCV_FILE}", flush=True)


if __name__ == "__main__":
    main()
