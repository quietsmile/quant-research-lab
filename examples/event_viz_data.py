"""为前瞻事件策略可视化预计算 Test 数据(新逻辑:行业相对便宜 + Top-20等权/单只≤5%)。

产物：viz_trades.parquet(实际持有的逐笔) / viz_equity.parquet(每日净值/持仓/基准)。
跑法：python examples/event_viz_data.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, load_listing, _FUND_DIR
from examples.forecast_event_strategy import build_trades, topn_weights, COST

TEST_START = pd.Timestamp("2024-01-01")
TOP_N, CAP = 20, 0.05


def main():
    px, tr = build_trades(growth_thr=30.0, cheap_factor=1.0)     # 行业相对便宜
    sub = tr[tr.strong & tr.cheap].copy()                        # 完整策略 A
    W = topn_weights(px, sub, TOP_N, CAP)                        # Top-N 权重矩阵
    name = load_listing().drop_duplicates("symbol").set_index("symbol")["name"]
    dret = px.pct_change(fill_method=None)
    mktidx = (1 + dret.mean(axis=1)).cumprod()
    posd = {d: i for i, d in enumerate(px.index)}

    # 逐笔(Test 入场，且当天确实进入了 Top-N 组合 → 真实操作)
    rows = []
    for t in sub.itertuples():
        if t.entry < TEST_START or t.symbol not in px.columns:
            continue
        wgt = W.iloc[posd[t.entry]][t.symbol]
        if wgt <= 0:                                            # 未入选 Top-N，没真买
            continue
        s = px[t.symbol]
        if pd.isna(s.get(t.entry)) or pd.isna(s.get(t.exit)):
            continue
        ret = s[t.exit] / s[t.entry] - 1 - 2 * COST
        m = mktidx[t.exit] / mktidx[t.entry] - 1
        rows.append({"symbol": t.symbol, "name": name.get(t.symbol, t.symbol),
                     "entry": t.entry, "exit": t.exit, "hold_days": (t.exit - t.entry).days,
                     "weight": round(wgt, 4), "p_change": t.p_change, "ret": ret, "excess": ret - m})
    trades = pd.DataFrame(rows).sort_values("entry")
    trades.to_parquet(_FUND_DIR / "viz_trades.parquet", index=False)

    # 每日净值/持仓(Test)
    gross = (W * dret).sum(axis=1)
    turnover = W.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    net = (gross - turnover * (COST * 2)).fillna(0.0)
    npos = (W > 0).sum(axis=1)
    td = px.index[px.index >= TEST_START]
    eq = pd.DataFrame({
        "date": td,
        "strategy": (1 + net.reindex(td).fillna(0)).cumprod().values,
        "market": (mktidx.reindex(td) / mktidx.reindex(td).iloc[0]).values,
        "holdings": npos.reindex(td).fillna(0).astype(int).values,
    })
    eq.to_parquet(_FUND_DIR / "viz_equity.parquet", index=False)
    print(f"Test 实际持有 {len(trades)} 笔(Top{TOP_N}/单只≤{CAP:.0%}) | 净值序列 {len(eq)} 天 | "
          f"平均超额/笔 {trades['excess'].mean():+.2%} | 胜率 {(trades['excess']>0).mean():.0%} | "
          f"策略累计 {eq['strategy'].iloc[-1]-1:+.1%} vs 大盘 {eq['market'].iloc[-1]-1:+.1%}")


if __name__ == "__main__":
    main()
