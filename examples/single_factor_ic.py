"""单因子 IC 拆解：看价值/质量/成长各自的预测力（而非只看合成）。

对每个因子单独算每期 IC（截面因子值 vs 下季收益的 spearman 相关），汇总
均值 IC、IC_IR、IC>0 占比、t 统计量。|IC 均值|>0.03 且 IC_IR>0.3 通常算有效。

跑法：python examples/single_factor_ic.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data import point_in_time
from quantlab.data.tushare_adapter import load_market_panel

FACTORS = {
    "价值-盈利收益率(1/PE)": ("ey", lambda cs, sn: (1.0 / cs["pe_ttm"]).where(cs["pe_ttm"] > 0)),
    "价值-账面市值比(1/PB)": ("bm", lambda cs, sn: (1.0 / cs["pb"]).where(cs["pb"] > 0)),
    "质量-ROE": ("roe", lambda cs, sn: sn["roe"]),
    "成长-归母净利单季同比": ("growth", lambda cs, sn: sn["net_profit_q_yoy"]),
}


def main() -> None:
    panel = load_market_panel().sort_values("trade_date")
    dates = sorted(panel["trade_date"].unique())
    aclose = panel.pivot_table(index="trade_date", columns="symbol", values="adj_close")

    ic_series = {k: [] for k in FACTORS}
    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        cs = panel[panel["trade_date"] == t].set_index("symbol")
        keep = cs.index[cs["total_mv"] >= cs["total_mv"].quantile(0.2)]
        fwd = (aclose.loc[t1] / aclose.loc[t] - 1.0).reindex(keep)
        sn = point_in_time(pd.Timestamp(t).strftime("%Y-%m-%d")).set_index("symbol")
        cs = cs.reindex(keep); sn = sn.reindex(keep)
        for name, (_, fn) in FACTORS.items():
            f = fn(cs, sn).reindex(keep)
            df = pd.concat([f.rename("f"), fwd.rename("r")], axis=1).dropna()
            if len(df) >= 50:
                ic_series[name].append(df["f"].corr(df["r"], method="spearman"))

    print("===== 单因子 IC 拆解（2016–2025，季度，全市场）=====")
    print(f"{'因子':<26}{'IC均值':>8}{'IC_IR':>8}{'IC>0':>7}{'t值':>7}{'  判定'}")
    print("-" * 64)
    for name in FACTORS:
        ic = pd.Series(ic_series[name]).dropna()
        m, s, n = ic.mean(), ic.std(), len(ic)
        ir = m / s if s else 0
        t = m / s * np.sqrt(n) if s else 0
        good = "✅有效" if abs(m) > 0.03 and abs(ir) > 0.3 else "弱"
        print(f"{name:<26}{m:>+8.3f}{ir:>8.2f}{(ic>0).mean():>7.0%}{t:>7.1f}  {good}")
    print("\n注：IC 为正=因子值越大下季收益越高；负=反向(如高PE→低收益则 1/PE 为正IC)。")


if __name__ == "__main__":
    main()
