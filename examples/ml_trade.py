"""可配置 LightGBM 交易策略：Top-N + N日持有 + 跳开过滤 + 止损 + 基本面池，全可调。

- 按持有期 H 训练(label=未来H日收益)，逐年 walk-forward，预测每个调仓日 → 存预测面板;
- simulate(): 执行规则都可调——Top-N、跳开阈值(开盘相对昨收涨幅>阈值则不买)、止损、
  是否加基本面池(趋势&质量)。次日开盘买入、含成本、止损按当日最低触发;
- 预训练 H∈{3,5,10} 存盘，供 web 快速调参(只重算 simulate，不重训)。

跑法：python examples/ml_trade.py            # 预训练并保存各 H 的预测+元数据
"""
import warnings; warnings.filterwarnings("ignore")
import json, pathlib
import numpy as np, pandas as pd, lightgbm as lgb
from examples.ml_alpha import build_features          # 复用 Alpha158 式日频因子面板
from examples.strategy_family import idx, ANN, trend, q, mv, sector
from quantlab.data.tushare_adapter import load_daily_ohlcv

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
syms = list(mv.columns)
SECTOR = sector.reindex(syms)                         # 行业映射(symbol→industry),Layer3 行业集中度用


_RISK = None


def _risk():
    """Layer3 风险门控面板: (AMT20 20日均成交额[元], VOL20 20日日收益std)。对齐 idx×syms。"""
    global _RISK
    if _RISK is None:
        op, hi, lo, cl = _ohlc()
        o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
        amt = o.pivot_table(index="trade_date", columns="symbol", values="amount").reindex(index=idx, columns=syms)
        amt20 = (amt * 1000).rolling(20).mean()        # tushare amount 单位千元 → 元
        vol20 = cl.pct_change(fill_method=None).rolling(20).std()
        _RISK = (amt20, vol20)
    return _RISK


_OHLC = None


def _ohlc():
    global _OHLC
    if _OHLC is None:
        o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
        pv = lambda c: o.pivot_table(index="trade_date", columns="symbol", values=c).reindex(index=idx, columns=syms)
        _OHLC = (pv("adj_open"), pv("adj_high"), pv("adj_low"), pv("adj_close"))
    return _OHLC


def load_signal():
    return pd.read_parquet(DD / "ml_signal.parquet")


_EXEC = None
from quantlab.data.tushare_adapter import _FUND_DIR as _FD


def _exec():
    """真实成交标志面板: (BUYABLE, SELLABLE, IS_ST)，对齐 idx×syms。数据缺失则退化为全可成交。"""
    global _EXEC
    if _EXEC is None:
        ef, sf = _FD / "exec_flags.parquet", _FD / "st_periods.parquet"
        if ef.exists():
            E = pd.read_parquet(ef); E = E[E.symbol.isin(syms)]
            E["trade_date"] = pd.to_datetime(E["trade_date"], format="%Y%m%d", errors="coerce") \
                if E["trade_date"].dtype == object else pd.to_datetime(E["trade_date"])
            buy = E.pivot_table(index="trade_date", columns="symbol", values="buyable", aggfunc="last") \
                   .reindex(index=idx, columns=syms).fillna(False).astype(bool)
            sell = E.pivot_table(index="trade_date", columns="symbol", values="sellable", aggfunc="last") \
                    .reindex(index=idx, columns=syms).fillna(False).astype(bool)
        else:                                            # 无数据:不约束(全可买卖),并提示
            print("⚠️ 未找到 exec_flags.parquet,真实撮合退化为理想成交。先跑 examples/pull_execution_data.py", flush=True)
            buy = pd.DataFrame(True, index=idx, columns=syms); sell = buy.copy()
        st = pd.DataFrame(False, index=idx, columns=syms)
        if sf.exists():
            N = pd.read_parquet(sf)
            for _, r in N.iterrows():
                if r["symbol"] not in syms:
                    continue
                a = pd.to_datetime(str(r["start_date"]), errors="coerce")
                b = pd.to_datetime(str(r["end_date"]), errors="coerce") if pd.notna(r["end_date"]) else idx[-1]
                if pd.isna(a):
                    continue
                st.loc[(st.index >= a) & (st.index <= b), r["symbol"]] = True
        _EXEC = (buy, sell, st)
    return _EXEC


SIG_H = 10              # IC 评估视野(10日);与持有期解耦——持有期是后处理
HORIZONS = (5, 10, 20)  # 多视野集成:各视野各训一模型,预测平均(实验证明显著提升IC)
LOOKBACK_M = 24         # 训练用滚动窗(月);实验证明滚动24月 ICIR 0.576 > 扩张窗 0.531(+8.5%)


def _xs_rank(panel):    # 当日横截面 rank → [0,1];作为训练标签(实验最大杠杆:学排序而非幅度)
    return panel.rank(axis=1, pct=True)


def train_signal(horizons=HORIZONS, train_step=2, purge=None, lookback_m=LOOKBACK_M):
    """训纯信号模型:**月度滚动窗口** walk-forward + purge,逐(隔)日出分。

    经 IC 实验(examples/ml_ic_experiments*.py / window_compare.py)定下三条增益:
      ① 训练标签用「未来收益的横截面 rank」(学排序、对离群/大盘整体涨跌更稳) —— 单项 IC +50%+;
      ② 多视野集成:对 5/10/20 日收益各训一模型、预测取平均(跨视野去噪) —— 再 +6%;
      ③ 训练用滚动 lookback_m 月窗(默认24)而非扩张窗 —— A股非平稳下近2年更跟得上风格,ICIR +8.5%。
    严格 PIT:每月预测前用「该月之前、且近 lookback_m 月内」的数据重训;训练样本须在预测起点前
    purge 个交易日(默认=最长视野,确保所有视野的未来标签都已实现、不泄漏)。IC 以未来10日收益为基准。
    训练只学收益排序,不含任何交易规则(跳开/止损/持有期/基本面)——这些都后处理。
    lookback_m=None 退回扩张窗。
    """
    if purge is None:
        purge = max(horizons)
    F, _label, close = build_features()
    feats = list(F)
    ic_label = (close.shift(-SIG_H) / close - 1).clip(-0.5, 0.5)        # IC 基准:未来10日收益
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    tr_days = days[::train_step]
    datas = {}                                                          # 各视野各自的训练样本表(标签=横截面rank)
    for h in horizons:
        lab = _xs_rank((close.shift(-h) / close - 1).clip(-0.5, 0.5))
        rows = []
        for d in tr_days:
            df = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
            df["y"] = lab.loc[d]; df["date"] = d
            rows.append(df)
        datas[h] = pd.concat(rows).dropna(subset=["y"])
    pred_panel = pd.DataFrame(np.nan, index=idx, columns=syms)
    meta = {"feats": feats, "label": f"未来{'/'.join(map(str,horizons))}日收益的横截面rank(多视野集成)",
            "n_train_samples": int(len(datas[SIG_H if SIG_H in datas else horizons[0]])),
            "train_step": train_step, "purge": purge, "horizons": list(horizons), "lookback_m": lookback_m,
            "retrain": (f"月度滚动{lookback_m}月窗" if lookback_m else "月度扩张窗口"), "folds": [], "ic": []}
    imp_acc = np.zeros(len(feats))
    ipos = {d: i for i, d in enumerate(idx)}
    months = sorted({(d.year, d.month) for d in idx if d.year >= 2021})
    for (Y, M) in months:
        te_days = [d for d in idx if d.year == Y and d.month == M]
        if not te_days:
            continue
        cut_i = ipos[te_days[0]] - purge                 # purge:标签须已实现(预测起点前 purge 日)
        if cut_i <= 0:
            continue
        cut_date = idx[cut_i]
        lo_date = (cut_date - pd.DateOffset(months=lookback_m)) if lookback_m else pd.Timestamp("2000-01-01")
        models = []                                      # 多视野:每个视野训一模型
        last_rows = 0
        for h in horizons:
            tr = datas[h][(datas[h]["date"] <= cut_date) & (datas[h]["date"] > lo_date)]
            if len(tr) < 4000:
                continue
            med = tr[feats].median()
            m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
            m.fit(tr[feats].fillna(med), tr["y"])
            models.append((m, med)); imp_acc += m.feature_importances_; last_rows = len(tr)
        if not models:
            continue
        for d in te_days:                                # 预测该月所有交易日(各视野平均)
            base = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
            preds = [pd.Series(mm.predict(base.fillna(md)[feats]), index=syms) for mm, md in models]
            pred = pd.concat(preds, axis=1).mean(axis=1)
            pred_panel.loc[d] = pred.values
            ic = pred.corr(ic_label.loc[d], method="spearman")
            if pd.notna(ic): meta["ic"].append({"date": str(d.date()), "ic": round(float(ic), 4)})
        meta["folds"].append({"month": f"{Y}-{M:02d}", "train_rows": int(last_rows),
                              "train_end": str(cut_date.date()), "test_days": len(te_days)})
    meta["importance"] = pd.Series(imp_acc, index=feats).sort_values(ascending=False).round(0).to_dict()
    meta["mean_ic"] = round(float(np.mean([x["ic"] for x in meta["ic"]])), 4) if meta["ic"] else None
    return pred_panel, meta


def simulate(pred, hold=5, top_n=20, gap_thr=0.05, stop_loss=0.08, use_fund=True, cost=0.0015,
             realistic=True, exclude_st=True, risk_gate=False, min_amount=5e7, max_vol=0.06, max_ind_frac=0.30):
    """后处理执行规则：Top-N、持有hold日、次日开盘买入、跳开>gap_thr不买、跌破entry*(1-stop)止损。

    realistic=True 时叠加 A 股真实撮合摩擦(需 examples/pull_execution_data.py 数据)：
      · 涨停/停牌(当日开盘不可买) → 该票当日不买;
      · 跌停/停牌(收盘不可卖) → 止损或到期当天卖不掉,**顺延到下一个可卖日按收盘价成交**;
    exclude_st=True 时把入场日处于 ST/*ST 区间的票剔出股票池。
    risk_gate=True 时叠加 Layer3 风险门控(可调)：
      · min_amount: 20日均成交额下限(元,默认5000万)——流动性;
      · max_vol: 20日日收益std上限(默认6%)——波动;
      · max_ind_frac: 单行业最大占比(默认30%)——行业集中度。
    """
    op, hi, lo, cl = _ohlc()
    BUY, SELL, ST = _exec() if (realistic or exclude_st) else (None, None, None)
    AMT, VOL = _risk() if risk_gate else (None, None)
    ind_cap = max(1, int(np.ceil(top_n * max_ind_frac))) if risk_gate else top_n
    pos = {d: i for i, d in enumerate(idx)}
    rb = [d for d in idx[::hold] if not pred.loc[d].isna().all()]
    trades = []
    for d in rb:
        i = pos[d]
        if i + 1 >= len(idx):
            break
        s = pred.loc[d].dropna()
        if use_fund:
            ok = (trend.loc[d].reindex(s.index).fillna(False)) & (q.loc[d].reindex(s.index).fillna(False))
            s = s[ok]
        cand = list(s.nlargest(top_n * 4).index)        # 多取些,过滤(跳开/涨停/ST/风险门控)后凑够
        entry_i = i + 1                                  # 次日开盘
        picks = []; ind_cnt = {}
        for sym in cand:
            o0, pc = op.iloc[entry_i][sym], cl.iloc[i][sym]
            if pd.isna(o0) or pd.isna(pc) or pc <= 0:
                continue
            if o0 / pc - 1 > gap_thr:                    # 跳开过滤(策略选择)
                continue
            if exclude_st and bool(ST.iloc[entry_i][sym]):           # ST 剔除
                continue
            if realistic and not bool(BUY.iloc[entry_i][sym]):       # 涨停/停牌不可买
                continue
            if risk_gate:                                            # Layer3 风险门控
                a, v = AMT.iloc[i][sym], VOL.iloc[i][sym]            # 用 d 日(已知)的流动性/波动
                if pd.isna(a) or a < min_amount:                     # 流动性下限
                    continue
                if pd.isna(v) or v > max_vol:                        # 波动上限
                    continue
                g = SECTOR.get(sym, "NA")
                if ind_cnt.get(g, 0) >= ind_cap:                     # 行业集中度上限
                    continue
                ind_cnt[g] = ind_cnt.get(g, 0) + 1
            picks.append(sym)
            if len(picks) >= top_n:
                break
        if not picks:
            continue
        end_i = min(entry_i + hold, len(idx) - 1)
        for sym in picks:
            ent = op.iloc[entry_i][sym]
            if pd.isna(ent) or ent <= 0:
                continue
            stop_px = ent * (1 - stop_loss)
            exit_i, exit_px, stopped = end_i, cl.iloc[end_i][sym], False
            for j in range(entry_i, end_i + 1):          # 逐日查止损(按最低)
                if lo.iloc[j][sym] <= stop_px:
                    exit_i, exit_px, stopped = j, stop_px, True
                    break
            if realistic:                                # 真实:意向出场日卖不掉→顺延到下一可卖日按收盘价
                j = exit_i
                while j < len(idx) - 1 and not bool(SELL.iloc[j][sym]):
                    j += 1
                if j != exit_i:
                    exit_i, exit_px, stopped = j, cl.iloc[j][sym], False
            if pd.isna(exit_px):
                continue
            r = exit_px / ent - 1 - 2 * cost
            trades.append({"entry": idx[entry_i], "exit": idx[exit_i], "symbol": sym, "ret": r,
                           "hold": exit_i - entry_i, "stopped": bool(stopped)})
    return _daily_from_trades(trades, top_n, cost), pd.DataFrame(trades)


def _daily_from_trades(trades, top_n, cost=0.0015):
    op, hi, lo, cl = _ohlc()
    ret = cl.pct_change(fill_method=None)
    w = pd.DataFrame(0.0, index=idx, columns=syms)
    pos = {d: i for i, d in enumerate(idx)}
    for t in trades:
        i0, i1 = pos[t["entry"]], pos[t["exit"]]
        w.iloc[i0 + 1:i1 + 1, w.columns.get_loc(t["symbol"])] = 1.0 / top_n
    gross = (w * ret).sum(axis=1).fillna(0.0)
    turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0) / 2
    return (gross - turnover * cost * 2).fillna(0.0)


def metrics(port):
    p = port.fillna(0); nav = (1 + p).cumprod()
    cg = nav.iloc[-1] ** (ANN / len(p)) - 1 if len(p) else 0
    sh = p.mean() / (p.std() + 1e-12) * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    return dict(cagr=cg, sharpe=sh, maxdd=dd, calmar=cg / abs(dd) if dd else 0, nav=nav)


def compare_realism():
    """同一信号下,理想撮合 vs 真实撮合(涨停不买/跌停停牌顺延/排除ST)的指标对比。"""
    pred = load_signal()
    print(f"{'配置':40s} {'年化':>6s} {'夏普':>6s} {'回撤':>7s} {'Calmar':>7s} {'笔数':>6s} {'止损率':>6s}", flush=True)
    for hold in (5, 10):
        for tag, kw in [("理想撮合", dict(realistic=False, exclude_st=False)),
                        ("真实撮合(涨停/跌停/停牌)", dict(realistic=True, exclude_st=False)),
                        ("真实撮合+排除ST", dict(realistic=True, exclude_st=True)),
                        ("真实+ST+Layer3风险门控", dict(realistic=True, exclude_st=True, risk_gate=True))]:
            port, tr = simulate(pred, hold=hold, **kw)
            m = metrics(port)
            print(f"持有{hold}日 {tag:30s} {m['cagr']*100:>+5.0f}% {m['sharpe']:>6.2f} {m['maxdd']*100:>+6.0f}% "
                  f"{m['calmar']:>7.2f} {len(tr):>6d} {tr['stopped'].mean()*100 if len(tr) else 0:>5.0f}%", flush=True)


def main():
    print(f"训练纯信号模型(标签=未来5/10/20日收益的横截面rank, 多视野集成, 月度滚动{LOOKBACK_M}月窗+purge)...", flush=True)
    pred, meta = train_signal()
    pred.to_parquet(DD / "ml_signal.parquet")
    json.dump(meta, open(DD / "ml_signal_meta.json", "w"), ensure_ascii=False, default=float)
    print(f"信号: 训练样本{meta['n_train_samples']} | 均IC {meta['mean_ic']} | 出分日 {pred.notna().any(axis=1).sum()}")
    print("\n真实撮合 vs 理想撮合对比:")
    compare_realism()
    print("预训练完成: ml_signal.parquet + ml_signal_meta.json")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        compare_realism()                                # 仅对比,不重训
    else:
        main()
