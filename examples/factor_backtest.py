"""基本面因子回测（兼数据质量探针）。

用 point_in_time 在每个调仓日取**截至当日已公告**的财报（防前视），按因子排序
选股，等权持有到下次调仓，与等权全集对比。同时打印一批**数据质量诊断**，主动
暴露处理 bug（前视、单位、NaN、陈旧数据）。

跑法：python examples/factor_backtest.py [因子] [topK比例]
  因子 ∈ {roe, dedt_yield, np_growth}（默认 roe）
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from quantlab.data import load_prices, point_in_time, active_source
from quantlab.data.universe import default_universe
from quantlab.factors import winsorize, rank_pct
from quantlab.costs import AShareCostModel
from quantlab.stats import metrics


def _price_panel(symbols, start, end):
    closes = {}
    for s in symbols:
        try:
            df = load_prices(s, start, end, source="offline")
            closes[s] = df["close"]
        except Exception:
            pass
    return pd.DataFrame(closes).sort_index()


def _factor_value(snap: pd.DataFrame, name: str) -> pd.Series:
    snap = snap.set_index("symbol")
    if name == "roe":
        return snap["roe"]
    if name == "dedt_yield":          # 扣非净利 / 营收（粗略盈利质量）
        return snap["profit_dedt"] / snap["revenue"].replace(0, np.nan)
    if name == "np_growth":           # 归母净利累计同比（用现成字段近似）
        return snap.get("net_profit")  # 占位：真增长需单季，见下游 features
    raise ValueError(name)


def main() -> None:
    factor = sys.argv[1] if len(sys.argv) > 1 else "roe"
    topq = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
    start, end = "2019-01-01", "2024-12-31"
    syms = default_universe()

    print(f"[数据源] point_in_time active = {active_source()}")
    prices = _price_panel(syms, start, end)
    print(f"[价格] {prices.shape[1]} 只 / {len(prices)} 交易日 "
          f"({prices.index[0].date()}~{prices.index[-1].date()})\n")

    # 季度末调仓日（用价格日历里每季最后一个交易日）
    rebal = prices.groupby(prices.index.to_period("Q")).tail(1).index
    rebal = [d for d in rebal if d >= prices.index[20]]

    diag = []           # 数据质量诊断
    port_rets, bench_rets = [], []
    prev_sel = set()
    cost = AShareCostModel().round_trip_cost_rate()

    for i in range(len(rebal) - 1):
        t, t1 = rebal[i], rebal[i + 1]
        snap = point_in_time(t.strftime("%Y-%m-%d"), symbols=syms)
        if snap.empty:
            continue

        # ---- 前视硬检查：任何 announce_date > t 即为 PIT bug ----
        future = (snap["announce_date"] > t).sum()
        stale_days = (t - snap["announce_date"]).dt.days
        fv = _factor_value(snap, factor)
        fv = winsorize(fv.dropna())
        score = rank_pct(fv)
        k = max(1, int(len(score) * topq))
        sel = list(score.sort_values(ascending=False).head(k).index)

        # 持有期收益
        held = prices.loc[t:t1, [s for s in sel if s in prices.columns]]
        if len(held) < 2:
            continue
        pr = held.iloc[-1] / held.iloc[0] - 1.0
        port_r = pr.mean()
        turn = len(set(sel) ^ prev_sel) / max(len(sel) + len(prev_sel), 1)
        port_r -= turn * cost            # 简单换手成本
        prev_sel = set(sel)
        bench_r = (prices.loc[t:t1, syms].iloc[-1] / prices.loc[t:t1, syms].iloc[0] - 1.0).mean()
        port_rets.append(port_r); bench_rets.append(bench_r)

        diag.append({"date": t.date(), "n_factor": int(fv.shape[0]),
                     "future_leak": int(future), "stale_median_d": int(stale_days.median()),
                     "factor_min": round(float(fv.min()), 2), "factor_med": round(float(fv.median()), 2),
                     "factor_max": round(float(fv.max()), 2), "n_pick": k})

    dd = pd.DataFrame(diag)
    print("===== 数据质量诊断（每个调仓截面）=====")
    print(dd.to_string(index=False))

    print("\n===== ⚠ Bug 自检 =====")
    leaks = int(dd["future_leak"].sum())
    print(f"  前视泄漏(announce_date>调仓日)总数: {leaks}  -> {'❌ 有 PIT bug!' if leaks else '✅ 0，无前视'}")
    print(f"  因子覆盖: 每期 {dd['n_factor'].min()}~{dd['n_factor'].max()} 只(共 {len(syms)})")
    print(f"  数据陈旧度: 中位 {int(dd['stale_median_d'].median())} 天(应≈30~120;过大=有缺口)")
    fmin, fmax = dd["factor_min"].min(), dd["factor_max"].max()
    print(f"  因子值域: [{fmin}, {fmax}]  ({'roe 看着像百分数✅' if factor=='roe' and 1<fmax<100 else '请核对单位'})")

    pr = pd.Series(port_rets); br = pd.Series(bench_rets)
    ann = lambda r: (1 + r).prod() ** (4 / max(len(r), 1)) - 1   # 季度→年化
    print("\n===== 因子组合 vs 等权全集（季度调仓，含换手成本）=====")
    print(f"  因子({factor}) 年化 {ann(pr):+.1%} | 累计 {(1+pr).prod()-1:+.1%}")
    print(f"  等权基准       年化 {ann(br):+.1%} | 累计 {(1+br).prod()-1:+.1%}")
    print(f"  超额(年化)     {ann(pr)-ann(br):+.1%}")
    print("\n注：20 只样本仅作数据质量探针，统计意义有限；重点看上面的 Bug 自检。")


if __name__ == "__main__":
    main()
