"""可投纯多头组合：合成因子做**行业+市值中性化**后选高分股，等权持有。

只做多(可投，无做空)；中性化后看的是剔除行业/规模暴露的**纯 alpha**。
对比沪深300 与等权全市场，给出年化、超额、信息比 IR、最大回撤、换手。

跑法：python examples/long_only_portfolio.py [top比例]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantlab.data import point_in_time
from quantlab.data.tushare_adapter import load_market_panel, load_listing, get_pro
from quantlab.factors import winsorize, zscore, neutralize_industry_size
from quantlab.stats import metrics


def main() -> None:
    top = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2
    panel = load_market_panel().sort_values("trade_date")
    dates = sorted(panel["trade_date"].unique())
    aclose = panel.pivot_table(index="trade_date", columns="symbol", values="adj_close")
    ind_map = load_listing().set_index("symbol")["industry"]

    port_rets, bench_ew, prev = [], [], set()
    cost = 0.0015   # 单边换手成本

    for i in range(len(dates) - 1):
        t, t1 = dates[i], dates[i + 1]
        cs = panel[panel["trade_date"] == t].set_index("symbol")
        keep = cs.index[cs["total_mv"] >= cs["total_mv"].quantile(0.2)]
        fwd = (aclose.loc[t1] / aclose.loc[t] - 1.0).reindex(keep)
        sn = point_in_time(pd.Timestamp(t).strftime("%Y-%m-%d")).set_index("symbol")

        df = pd.DataFrame(index=keep)
        df["ey"] = (1.0 / cs["pe_ttm"]).where(cs["pe_ttm"] > 0)
        df["bm"] = (1.0 / cs["pb"]).where(cs["pb"] > 0)
        df["roe"] = sn["roe"].reindex(keep)
        df["growth"] = sn["net_profit_q_yoy"].reindex(keep)
        df["fwd"] = fwd
        df["ind"] = ind_map.reindex(keep)
        df["lsz"] = np.log(cs["total_mv"].reindex(keep))
        df = df.dropna(subset=["fwd", "ind", "lsz"])
        if len(df) < 80:
            continue

        comp = pd.concat([zscore(winsorize(df[c])) for c in ["ey", "bm", "roe", "growth"]],
                         axis=1).mean(axis=1, skipna=True)
        df["score"] = neutralize_industry_size(comp, df["ind"], df["lsz"])
        df = df.dropna(subset=["score"])
        if len(df) < 80:
            continue

        k = max(10, int(len(df) * top))
        sel = set(df["score"].sort_values(ascending=False).head(k).index)
        ret = df.loc[list(sel), "fwd"].mean()
        turn = len(sel ^ prev) / max(len(sel) + len(prev), 1)
        port_rets.append(ret - turn * cost)
        bench_ew.append(df["fwd"].mean())
        prev = sel

    # 沪深300 对齐到季度
    pro = get_pro()
    idx = pro.index_daily(ts_code="000300.SH", start_date="20160101", end_date="20251231",
                          fields="trade_date,close")
    idx["trade_date"] = pd.to_datetime(idx["trade_date"], format="%Y%m%d")
    idx = idx.set_index("trade_date").sort_index()["close"]
    hs = []
    for i in range(len(dates) - 1):
        a = idx.asof(dates[i]); b = idx.asof(dates[i + 1])
        hs.append(b / a - 1 if a and b else np.nan)
    hs = pd.Series(hs[:len(port_rets)])

    pr, ew = pd.Series(port_rets), pd.Series(bench_ew)
    eq = (1 + pr).cumprod()
    ann = lambda r: (1 + pd.Series(r).dropna()).prod() ** (4 / max(len(pd.Series(r).dropna()), 1)) - 1
    excess = pr - hs
    print("===== 纯多头中性化组合（行业+市值中性，季度调仓，扣成本）=====")
    print(f"  期数 {len(pr)} | 每期持有 ~{k} 只 | {pd.Timestamp(dates[0]).date()}~{pd.Timestamp(dates[len(pr)]).date()}")
    print(f"  组合      年化 {ann(pr):+.1%} | 累计 {eq.iloc[-1]-1:+.1%} | 最大回撤 {metrics.max_drawdown(pr):.1%} | 夏普 {metrics.sharpe_ratio(pr, periods=4):.2f}")
    print(f"  沪深300   年化 {ann(hs):+.1%}")
    print(f"  等权全市场 年化 {ann(ew):+.1%}")
    print(f"  超额(vs沪深300) 年化 {ann(pr)-ann(hs):+.1%} | 信息比 IR {excess.mean()/excess.std()*np.sqrt(4):.2f} | 季度胜率 {(excess>0).mean():.0%}")
    print("\n注：纯多头可投；行业+市值已中性化，超额更接近纯选股 alpha。")


if __name__ == "__main__":
    main()
