"""按公告日拉全市场业绩预告(forecast)——前瞻盈利预期,PIT(预告公开日=信号日)。

forecast 接口须按 ann_date 或 ts_code；这里按交易日迭代(含退市，无幸存者偏差)。
存长表 forecast.parquet: ts_code,symbol,ann_date,end_date,type,p_change_min/max,net_profit_min/max。

跑法：python examples/pull_forecast.py [起始YYYYMMDD] [结束]
"""
from __future__ import annotations
import socket, sys, time
socket.setdefaulttimeout(30)
import pandas as pd
from quantlab.data.tushare_adapter import get_pro, _FUND_DIR

FORECAST_FILE = _FUND_DIR / "forecast.parquet"
FIELDS = "ts_code,ann_date,end_date,type,p_change_min,p_change_max,net_profit_min,net_profit_max"


def main():
    start = sys.argv[1] if len(sys.argv) > 1 else "20160101"
    end = sys.argv[2] if len(sys.argv) > 2 else "20251231"
    pro = get_pro()
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, is_open="1")
    days = sorted(cal["cal_date"].tolist())
    print(f"业绩预告：扫 {len(days)} 个交易日 {start}~{end}", flush=True)
    frames = []
    for i, d in enumerate(days, 1):
        for attempt in range(3):
            try:
                df = pro.forecast(ann_date=d, fields=FIELDS)
                if df is not None and len(df):
                    frames.append(df)
                break
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        if i % 300 == 0:
            print(f"  {i}/{len(days)} 累计 {sum(len(x) for x in frames)} 条", flush=True)
        time.sleep(0.1)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(out):
        out["symbol"] = out["ts_code"].str[:6]
        out["ann_date"] = pd.to_datetime(out["ann_date"], format="%Y%m%d", errors="coerce")
        out["end_date"] = pd.to_datetime(out["end_date"], format="%Y%m%d", errors="coerce")
        out = out.drop_duplicates(["ts_code", "ann_date", "end_date"])
        _FUND_DIR.mkdir(parents=True, exist_ok=True)
        out.to_parquet(FORECAST_FILE, index=False)
    print(f"\n完成：{len(out)} 条 / {out['symbol'].nunique() if len(out) else 0} 只 → {FORECAST_FILE}", flush=True)


if __name__ == "__main__":
    main()
