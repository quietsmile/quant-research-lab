"""拉「真实成交」所需数据：每日涨跌停价(stk_limit)+原始OHLC(daily) → 判定可买/可卖标志；
名称变更史(namechange) → PIT 识别 ST 区间。供 simulate 真实撮合(涨停不买/跌停停牌不卖/排除ST)。

- buyable[d,sym]: 当日有成交 且 开盘未涨停(open < up_limit) → 可在开盘买入
- sellable[d,sym]: 当日有成交 且 收盘未跌停(close > down_limit) → 可卖出(否则顺延)
- st_periods: name 含 'ST' 的区间(含*ST),PIT 用 start_date~end_date

跑法：python examples/pull_execution_data.py [start] [end] [which: all|date|st]
"""
from __future__ import annotations
import socket, sys, time
socket.setdefaulttimeout(30)
import pandas as pd
from quantlab.data.tushare_adapter import get_pro, _FUND_DIR
from examples.strategy_family import mv

EXEC_FILE = _FUND_DIR / "exec_flags.parquet"
ST_FILE = _FUND_DIR / "st_periods.parquet"
syms = list(mv.columns)
TOL = 1e-3


def pull_by_date(pro, days):
    rows = []
    for i, d in enumerate(days, 1):
        for attempt in range(3):
            try:
                px = pro.daily(trade_date=d, fields="ts_code,open,high,low,close,vol")
                lim = pro.stk_limit(trade_date=d, fields="ts_code,up_limit,down_limit")
                m = px.merge(lim, on="ts_code", how="left")
                m = m[m["vol"].fillna(0) > 0]                      # 有成交才可能买卖
                m["buyable"] = m["open"] < m["up_limit"] - TOL     # 开盘未涨停→可买
                m["sellable"] = m["close"] > m["down_limit"] + TOL  # 收盘未跌停→可卖
                m["trade_date"] = d; m["symbol"] = m["ts_code"].str[:6]
                rows.append(m[["trade_date", "symbol", "buyable", "sellable"]])
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.5 * (attempt + 1))
        if i % 200 == 0:
            print(f"  by-date {i}/{len(days)} ...", flush=True)
        time.sleep(0.10)
    if rows:
        E = pd.concat(rows, ignore_index=True)
        E.to_parquet(EXEC_FILE, index=False)
        print(f"涨跌停标志 {len(E)} 行 / {E.trade_date.min()}~{E.trade_date.max()} → {EXEC_FILE}", flush=True)


def pull_st(pro):
    sfx = lambda s: (s + ".SH") if s[0] == "6" else ((s + ".BJ") if s[0] in "489" else (s + ".SZ"))
    rows = []
    for i, s in enumerate(syms, 1):
        for attempt in range(3):
            try:
                h = pro.namechange(ts_code=sfx(s), fields="ts_code,name,start_date,end_date,change_reason")
                if h is not None and len(h):
                    rows.append(h)
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.2 * (attempt + 1))
        if i % 300 == 0:
            print(f"  st {i}/{len(syms)} ...", flush=True)
        time.sleep(0.06)
    if rows:
        N = pd.concat(rows, ignore_index=True)
        N["symbol"] = N["ts_code"].str[:6]
        st = N[N["name"].str.contains("ST", na=False)].copy()     # 含 ST/*ST
        st.to_parquet(ST_FILE, index=False)
        print(f"ST 区间 {len(st)} 条 / 覆盖 {st.symbol.nunique()} 只 → {ST_FILE}", flush=True)


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20190101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20260618"
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    pro = get_pro()
    days = sorted(pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")["cal_date"].tolist())
    print(f"真实成交数据：{start}~{end} | {len(days)} 交易日 | {len(syms)} 只 | which={which}", flush=True)
    if which in ("all", "date"):
        pull_by_date(pro, days)
    if which in ("all", "st"):
        pull_st(pro)
    print("完成。", flush=True)


if __name__ == "__main__":
    main()
