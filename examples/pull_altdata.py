"""拉另类因子原始数据：龙虎榜(top_list) + 融资融券明细(margin_detail) 按交易日；
股东户数(stk_holdernumber) 按股票。存长表 parquet，供 alt 因子构造 + 增量 IC 实验。

跑法：python examples/pull_altdata.py [start YYYYMMDD] [end] [which: all|date|holder]
"""
from __future__ import annotations
import socket, sys, time
socket.setdefaulttimeout(30)
import pandas as pd
from quantlab.data.tushare_adapter import get_pro, _FUND_DIR
from examples.strategy_family import mv          # 复用股票池(syms)

LHB_FILE = _FUND_DIR / "lhb_toplist.parquet"
MARGIN_FILE = _FUND_DIR / "margin_detail.parquet"
HOLDER_FILE = _FUND_DIR / "holder_number.parquet"
syms = list(mv.columns)


def pull_by_date(pro, days):
    lhb, mg = [], []
    for i, d in enumerate(days, 1):
        for attempt in range(3):
            try:
                t = pro.top_list(trade_date=d)
                if t is not None and len(t):
                    lhb.append(t)
                m = pro.margin_detail(trade_date=d)
                if m is not None and len(m):
                    mg.append(m)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        if i % 200 == 0:
            print(f"  by-date {i}/{len(days)} ...", flush=True)
        time.sleep(0.10)
    if lhb:
        L = pd.concat(lhb, ignore_index=True); L["symbol"] = L["ts_code"].str[:6]
        L.to_parquet(LHB_FILE, index=False)
        print(f"龙虎榜 {len(L)} 行 / {L.trade_date.min()}~{L.trade_date.max()} → {LHB_FILE}", flush=True)
    if mg:
        M = pd.concat(mg, ignore_index=True); M["symbol"] = M["ts_code"].str[:6]
        M.to_parquet(MARGIN_FILE, index=False)
        print(f"融资融券 {len(M)} 行 / {M.trade_date.min()}~{M.trade_date.max()} → {MARGIN_FILE}", flush=True)


def pull_holder(pro, start, end):
    # stk_holdernumber 按 ts_code 一次返回该股所有期;只拉本项目股票池
    sfx = lambda s: (s + ".SH") if s[0] == "6" else ((s + ".BJ") if s[0] in "489" else (s + ".SZ"))
    rows = []
    for i, s in enumerate(syms, 1):
        for attempt in range(3):
            try:
                h = pro.stk_holdernumber(ts_code=sfx(s), start_date=start, end_date=end)
                if h is not None and len(h):
                    rows.append(h)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.2 * (attempt + 1))
        if i % 300 == 0:
            print(f"  holder {i}/{len(syms)} ...", flush=True)
        time.sleep(0.06)
    if rows:
        H = pd.concat(rows, ignore_index=True); H["symbol"] = H["ts_code"].str[:6]
        H.to_parquet(HOLDER_FILE, index=False)
        print(f"股东户数 {len(H)} 行 / {H.end_date.min()}~{H.end_date.max()} → {HOLDER_FILE}", flush=True)


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20190101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20260618"
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    pro = get_pro()
    days = sorted(pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")["cal_date"].tolist())
    print(f"另类数据：{start}~{end} | {len(days)} 交易日 | {len(syms)} 只 | which={which}", flush=True)
    if which in ("all", "date"):
        pull_by_date(pro, days)
    if which in ("all", "holder"):
        pull_holder(pro, start, end)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
