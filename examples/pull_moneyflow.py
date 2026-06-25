"""拉历史个股资金流(按单大小), 计算'主力净流入'(特大+大单净额)及其占成交额比例。
2020-2026, 每交易日一次。存 dashboard_data/moneyflow_main.parquet (date×symbol 主力净流入率)。
"""
import warnings; warnings.filterwarnings("ignore")
import sys, time; sys.path.insert(0, "/home/claudeuser/econ/quant-research-lab")
import pandas as pd, numpy as np, pathlib
from quantlab.data.tushare_adapter import get_pro
pro = get_pro()
OUT = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); OUT.mkdir(exist_ok=True)

cal = pro.trade_cal(exchange="SSE", start_date="20200101", end_date="20260620", is_open="1")
days = sorted(cal["cal_date"].tolist())
print(f"拉 {len(days)} 个交易日主力资金流 {days[0]}~{days[-1]}", flush=True)
rows = []
for i, d in enumerate(days):
    for att in range(3):
        try:
            df = pro.moneyflow(trade_date=d)
            zl = (df["buy_elg_amount"] + df["buy_lg_amount"] - df["sell_elg_amount"] - df["sell_lg_amount"])
            tot = (df["buy_elg_amount"] + df["buy_lg_amount"] + df["buy_md_amount"] + df["buy_sm_amount"]
                   + df["sell_elg_amount"] + df["sell_lg_amount"] + df["sell_md_amount"] + df["sell_sm_amount"])
            sub = pd.DataFrame({"symbol": df["ts_code"].str[:6], "zl_net": zl, "tot": tot})
            sub["trade_date"] = pd.to_datetime(d); rows.append(sub); break
        except Exception as e:
            if att == 2: print(f"  {d} fail {str(e)[:40]}", flush=True)
            time.sleep(1.5)
    if (i + 1) % 200 == 0: print(f"  ...{i+1}/{len(days)}", flush=True)
mf = pd.concat(rows, ignore_index=True)
mf["zl_rate"] = mf["zl_net"] / mf["tot"].replace(0, np.nan)     # 主力净流入占成交额比例(scale-free)
zl_rate = mf.pivot_table(index="trade_date", columns="symbol", values="zl_rate")
zl_net = mf.pivot_table(index="trade_date", columns="symbol", values="zl_net")
zl_rate.to_parquet(OUT / "moneyflow_zlrate.parquet")
zl_net.to_parquet(OUT / "moneyflow_zlnet.parquet")
print(f"保存: 主力净流入率 {zl_rate.shape} 范围 {zl_rate.index.min().date()}~{zl_rate.index.max().date()}", flush=True)
