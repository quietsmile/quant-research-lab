"""量价因子研究：用日频 OHLCV + 成交额/换手率，严肃评估(IC + 开发/冻结Test + DSR)。

7 个量价 idea(月频、A 股、含退市)：
- turnover_chg : 换手率变化(近月均换手 / 近季均换手)，过热反向
- illiq        : Amihud 非流动性 = |日收益|/成交额 的均值（流动性溢价）
- amount_trend : 成交额趋势(近月/近季)
- amplitude    : 振幅 = (high-low)/close 均值，波动/投机度(反向)
- vol_price_corr: 量价相关(收益与成交额变化的相关)，背离信号
- close_pos    : 收盘位置 = (close-low)/(high-low)，买盘强弱
- turnover_lvl : 换手率水平(高换手反向，过度关注)

跑法：python examples/volume_price_factors.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_ohlcv, load_market_panel
from quantlab import eval as ev

P = 12
COST = 0.002


def _prep():
    d = load_daily_ohlcv().sort_values(["symbol", "trade_date"])
    close = d.pivot_table(index="trade_date", columns="symbol", values="adj_close")
    amount = d.pivot_table(index="trade_date", columns="symbol", values="amount")
    turn = d.pivot_table(index="trade_date", columns="symbol", values="turnover_rate")
    high = d.pivot_table(index="trade_date", columns="symbol", values="adj_high")
    low = d.pivot_table(index="trade_date", columns="symbol", values="adj_low")
    ret = close.pct_change(fill_method=None)
    me = list(close.groupby(close.index.to_period("M")).tail(1).index)
    mvq = load_market_panel().pivot_table(index="trade_date", columns="symbol", values="total_mv").sort_index()
    return dict(close=close, amount=amount, turn=turn, high=high, low=low, ret=ret, me=me, mvq=mvq)


def factor(name, D, i, t):
    close, amount, turn, high, low, ret = D["close"], D["amount"], D["turn"], D["high"], D["low"], D["ret"]
    W = ret.iloc[i - 21:i]
    if name == "turnover_lvl":   return -turn.iloc[i - 21:i].mean()
    if name == "turnover_chg":   return -(turn.iloc[i - 21:i].mean() / (turn.iloc[i - 63:i].mean() + 1e-9))
    if name == "amount_trend":   return amount.iloc[i - 21:i].mean() / (amount.iloc[i - 63:i].mean() + 1e-9)
    if name == "illiq":          return (W.abs() / (amount.iloc[i - 21:i] + 1e-9)).mean() * 1e6  # Amihud
    if name == "amplitude":      return -((high.iloc[i - 21:i] - low.iloc[i - 21:i]) / close.iloc[i - 21:i]).mean()
    if name == "close_pos":      return ((close.iloc[i - 1] - low.iloc[i - 21:i].min()) /
                                         (high.iloc[i - 21:i].max() - low.iloc[i - 21:i].min() + 1e-9))
    if name == "vol_price_corr":
        da = amount.iloc[i - 21:i].pct_change(fill_method=None)
        return -W.corrwith(da)       # 量价背离：负相关(放量下跌/缩量上涨)倾向反转
    raise ValueError(name)


IDEAS = ["turnover_lvl", "turnover_chg", "amount_trend", "illiq", "amplitude", "close_pos", "vol_price_corr"]


def main():
    D = _prep()
    me, mvq = D["me"], D["mvq"]
    close = D["close"]
    print(f"量价因子评估：{close.shape[1]} 只 / 月频 {me[12].date()}~{me[-1].date()}\n")
    ls = {n: {} for n in IDEAS}; ic = {n: [] for n in IDEAS}
    for j, t in enumerate(me[:-1]):
        i = close.index.get_loc(t)
        if i < 63:
            continue
        mv = mvq.loc[:t].iloc[-1] if len(mvq.loc[:t]) else None
        keep = mv.index[mv >= mv.quantile(0.2)] if mv is not None else close.columns
        fwd = (close.loc[me[j + 1]] / close.loc[t] - 1).reindex(keep)
        for n in IDEAS:
            f = factor(n, D, i, t).reindex(keep)
            d = pd.concat([f.rename("x"), fwd.rename("r")], axis=1).dropna()
            if len(d) < 100:
                continue
            ic[n].append(d["x"].corr(d["r"], method="spearman"))
            d["q"] = pd.qcut(d["x"].rank(method="first"), 5, labels=range(1, 6)).astype(int)
            ls[n][t] = d.loc[d["q"] == 5, "r"].mean() - d.loc[d["q"] == 1, "r"].mean() - COST

    print(f"{'因子':<16}{'IC':>7}{'IC_t':>6}{'开发夏普':>9}{'Test夏普':>9}")
    rows = []
    for n in IDEAS:
        s = pd.Series(ls[n]).sort_index(); icv = pd.Series(ic[n]).dropna()
        dev_d, test_d = ev.dev_test_split(s.index, 0.2)
        dev, test = s.loc[s.index.isin(dev_d)], s.loc[s.index.isin(test_d)]
        t_ic = icv.mean() / icv.std() * np.sqrt(len(icv)) if icv.std() else 0
        rows.append((n, icv.mean(), t_ic, ev.sharpe(dev, P), ev.sharpe(test, P)))
    for n, icm, tic, dsr, tsr in sorted(rows, key=lambda x: -x[4]):
        flag = " ✅" if tsr > 0.5 and abs(tic) > 2 else ""
        print(f"{n:<16}{icm:>+7.3f}{tic:>6.1f}{dsr:>9.2f}{tsr:>9.2f}{flag}")
    print("\n注：Test 夏普为冻结样本外多空；后续可把 OOS 最稳的并入合成、走完整面板评估。")


if __name__ == "__main__":
    main()
