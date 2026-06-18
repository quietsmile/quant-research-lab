"""为前瞻事件策略的可视化预计算 Test 期数据：逐笔交易 + 每日净值/持仓。

产物（供 dashboard/event_app.py 读取）：
  ~/.local/share/quantlab/fundamentals/viz_trades.parquet  逐笔
  ~/.local/share/quantlab/fundamentals/viz_equity.parquet  每日净值/持仓/基准
跑法：python examples/event_viz_data.py
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, load_listing, _FUND_DIR
from examples.forecast_event_strategy import build_trades, COST

TEST_START = pd.Timestamp("2024-01-01")


def main():
    px, tr = build_trades(growth_thr=30.0, fwd_pe_cap=20.0)
    sub = tr[tr.strong & tr.cheap].copy()                  # 完整策略 A
    name = load_listing().drop_duplicates("symbol").set_index("symbol")["name"]
    dret = px.pct_change(fill_method=None)
    mktidx = (1 + dret.mean(axis=1)).cumprod()

    # 逐笔(Test: 入场在 2024 起)
    rows = []
    for t in sub.itertuples():
        if t.entry < TEST_START:
            continue
        if t.symbol not in px.columns:
            continue
        pe, px_ = None, px[t.symbol]
        if pd.isna(px_.get(t.entry)) or pd.isna(px_.get(t.exit)):
            continue
        ret = px_[t.exit] / px_[t.entry] - 1 - 2 * COST
        m = mktidx[t.exit] / mktidx[t.entry] - 1
        rows.append({"symbol": t.symbol, "name": name.get(t.symbol, t.symbol),
                     "entry": t.entry, "exit": t.exit, "hold_days": (t.exit - t.entry).days,
                     "ret": ret, "excess": ret - m})
    trades = pd.DataFrame(rows).sort_values("entry")
    trades.to_parquet(_FUND_DIR / "viz_trades.parquet", index=False)

    # 每日净值/持仓(Test 期)，仅用 Test 内入场的交易构成组合
    cols = {c: i for i, c in enumerate(px.columns)}
    posd = {d: i for i, d in enumerate(px.index)}
    mask = np.zeros(px.shape, dtype=bool)
    for t in sub.itertuples():
        if t.entry >= TEST_START and t.symbol in cols:
            mask[posd[t.entry]:posd[t.exit], cols[t.symbol]] = True
    held = pd.DataFrame(mask, index=px.index, columns=px.columns)
    n = held.sum(axis=1)
    gross = ((held * dret).sum(axis=1) / n.replace(0, np.nan)).fillna(0.0)
    w = held.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    net = gross - turnover * (COST * 2)
    test_days = px.index[px.index >= TEST_START]
    eq = pd.DataFrame({
        "date": test_days,
        "strategy": (1 + net.reindex(test_days).fillna(0)).cumprod().values,
        "market": (mktidx.reindex(test_days) / mktidx.reindex(test_days).iloc[0]).values,
        "holdings": n.reindex(test_days).fillna(0).astype(int).values,
    })
    eq.to_parquet(_FUND_DIR / "viz_equity.parquet", index=False)
    print(f"Test 逐笔 {len(trades)} 笔 | 净值序列 {len(eq)} 天 | 平均超额/笔 {trades['excess'].mean():+.2%} | "
          f"胜率 {(trades['excess']>0).mean():.0%}")
    print("已存 viz_trades.parquet / viz_equity.parquet")


if __name__ == "__main__":
    main()
