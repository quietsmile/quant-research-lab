"""前瞻事件策略：业绩预告(对未来的预期) + 预期超过当前市值 → 买入，持有到正式财报前。

忠实用户 idea（用真·前瞻数据）：
- 信号日 = 业绩预告公告日(forecast ann_date，前瞻信息公开当天，PIT)
- 对未来的预期：正向预告(预增/略增/扭亏/续盈) 且 预告净利同比 p_change > 阈值
- 预期超过当前市值：按预告增速折算后估值仍便宜 → forward_PE = PE_ttm/(1+p_change) < 上限
- 卖出：该期正式财报公告前最后一交易日（"下次财报前最后一天"）

变体：A 完整(预告强 + 便宜)  B 仅预告强(不看估值)  → 看"超过市值"条件是否加分。
评估：把事件交易转成每日等权持仓组合 → 月度收益 → 完整面板 + 冻结 Test。

跑法：python examples/forecast_event_strategy.py [增速阈值%] [forwardPE上限]
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")
import sys

import numpy as np
import pandas as pd

from quantlab.data.tushare_adapter import load_daily_prices, load_market_panel, _FUND_DIR, load_pit
from quantlab import report as rp
from quantlab import eval as ev

POS_TYPES = {"预增", "略增", "扭亏", "续盈"}
COST = 0.003


def build_trades(growth_thr, fwd_pe_cap):
    px = load_daily_prices().pivot_table(index="trade_date", columns="symbol", values="adj_close").sort_index()
    dates = px.index; pos = {d: i for i, d in enumerate(dates)}
    fc = pd.read_parquet(_FUND_DIR / "forecast.parquet")
    fc = fc[fc["type"].isin(POS_TYPES) & fc["p_change_min"].notna()].copy()
    fc["ann_date"] = pd.to_datetime(fc["ann_date"]); fc["end_date"] = pd.to_datetime(fc["end_date"])
    # 正式财报公告日: (symbol, report_period) -> announce_date
    pit = load_pit()[["symbol", "report_period", "announce_date"]].dropna()
    rep_ann = {(r.symbol, r.report_period): r.announce_date for r in pit.itertuples()}
    # PE_ttm 季度末 asof
    pe = load_market_panel().pivot_table(index="trade_date", columns="symbol", values="pe_ttm").sort_index()

    rows = []
    for r in fc.itertuples():
        sym = r.symbol
        if sym not in px.columns:
            continue
        a_rep = rep_ann.get((sym, r.end_date))
        if a_rep is None or a_rep <= r.ann_date:
            continue
        # entry: 预告次日; exit: 正式财报前最后一交易日
        e_after = dates[dates > r.ann_date]
        x_before = dates[dates < a_rep]
        if len(e_after) == 0 or len(x_before) == 0:
            continue
        entry, exit_d = e_after[0], x_before[-1]
        if pos[entry] >= pos[exit_d]:
            continue
        pe_s = pe[sym].loc[:r.ann_date].dropna() if sym in pe.columns else pd.Series(dtype=float)
        pe_now = pe_s.iloc[-1] if len(pe_s) else np.nan
        fwd_pe = pe_now / (1 + r.p_change_min / 100.0) if (pe_now and pe_now > 0) else np.inf
        strong = r.p_change_min > growth_thr
        cheap = 0 < fwd_pe < fwd_pe_cap
        rows.append({"symbol": sym, "entry": entry, "exit": exit_d,
                     "strong": strong, "cheap": cheap})
    return px, pd.DataFrame(rows)


def daily_portfolio(px, trades):
    """事件交易 → 每日等权持仓组合的月度收益。"""
    dret = px.pct_change(fill_method=None)
    cols = {c: i for i, c in enumerate(px.columns)}
    posd = {d: i for i, d in enumerate(px.index)}
    mask = np.zeros(px.shape, dtype=bool)
    for t in trades.itertuples():
        if t.symbol in cols:
            mask[posd[t.entry]:posd[t.exit], cols[t.symbol]] = True
    held = pd.DataFrame(mask, index=px.index, columns=px.columns)
    n = held.sum(axis=1)
    gross = ((held * dret).sum(axis=1) / n.replace(0, np.nan)).fillna(0.0)
    # 扣换手成本：每日权重变化 × 单边成本(进出场都计)
    w = held.div(n.replace(0, np.nan), axis=0).fillna(0.0)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0) / 2.0
    net = gross - turnover * (COST * 2)            # 买卖双边
    monthly = (1 + net).groupby(net.index.to_period("M")).prod() - 1
    monthly.index = monthly.index.to_timestamp("M")
    return monthly


def main():
    gthr = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    pecap = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    print(f"前瞻事件策略：正向预告 & 预告增速>{gthr:.0f}% & forwardPE<{pecap:.0f} → 持有到正式财报前\n")
    px, tr = build_trades(gthr, pecap)
    print(f"候选预告事件: {len(tr)} | 强预告 {int(tr.strong.sum())} | 强且便宜 {int((tr.strong&tr.cheap).sum())}\n")

    # 等权市场月度(基准)
    mkt = (1 + px.pct_change(fill_method=None).mean(axis=1)).groupby(px.index.to_period("M")).apply(lambda x: x.prod() if False else None)
    mkt_m = px.pct_change(fill_method=None).mean(axis=1)
    mkt_m = (1 + mkt_m).groupby(mkt_m.index.to_period("M")).prod() - 1
    mkt_m.index = mkt_m.index.to_timestamp("M")

    for lab, sub in [("A 完整(预告强+便宜)", tr[tr.strong & tr.cheap]),
                     ("B 仅预告强(不看估值)", tr[tr.strong])]:
        m_ret = daily_portfolio(px, sub)
        full = rp.performance_report(m_ret, periods=12, benchmark=mkt_m, n_trials=2, trials_sr_std=0.3)
        print(rp.format_report(full, lab + " 全样本"))
        # 冻结 Test (最近20%)
        dev_d, test_d = ev.dev_test_split(m_ret.index, 0.2)
        tt = m_ret.loc[m_ret.index.isin(test_d)]
        tr_rep = rp.performance_report(tt, periods=12, benchmark=mkt_m.reindex(tt.index))
        print(f"  └ 冻结Test({tt.index[0].date()}~{tt.index[-1].date()}): CAGR {tr_rep.get('cagr',0):+.1%} | "
              f"Sharpe {tr_rep.get('sharpe',0):.2f} | 最大回撤 {tr_rep.get('max_drawdown',0):.1%} | "
              f"信息比 {tr_rep.get('info_ratio',float('nan')):.2f}\n")


if __name__ == "__main__":
    main()
