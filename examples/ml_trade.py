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
from examples.strategy_family import idx, ANN, trend, q, mv
from quantlab.data.tushare_adapter import load_daily_ohlcv

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
syms = list(mv.columns)


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


SIG_H = 10              # IC 评估视野(10日);与持有期解耦——持有期是后处理
HORIZONS = (5, 10, 20)  # 多视野集成:各视野各训一模型,预测平均(实验证明显著提升IC)


def _xs_rank(panel):    # 当日横截面 rank → [0,1];作为训练标签(实验最大杠杆:学排序而非幅度)
    return panel.rank(axis=1, pct=True)


def train_signal(horizons=HORIZONS, train_step=2, purge=None):
    """训纯信号模型:**月度扩张窗口** walk-forward + purge,逐(隔)日出分。

    经 IC 实验(examples/ml_ic_experiments*.py)定下两条增益最大的技巧:
      ① 训练标签用「未来收益的横截面 rank」(学排序、对离群/大盘整体涨跌更稳) —— 单项 IC +50%+;
      ② 多视野集成:对 5/10/20 日收益各训一模型、预测取平均(跨视野去噪) —— 再 +6%。
    严格 PIT:每月预测前用「该月之前的全部数据」重训;训练样本须在预测起点前 purge 个交易日
    (默认=最长视野,确保所有视野的未来标签都已实现、不泄漏)。IC 仍以未来10日收益为基准。
    训练只学收益排序,不含任何交易规则(跳开/止损/持有期/基本面)——这些都后处理。
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
            "train_step": train_step, "purge": purge, "horizons": list(horizons),
            "retrain": "月度扩张窗口(每月用该月以前全部数据重训)", "folds": [], "ic": []}
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
        models = []                                      # 多视野:每个视野训一模型
        last_rows = 0
        for h in horizons:
            tr = datas[h][datas[h]["date"] <= cut_date]
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


def simulate(pred, hold=5, top_n=20, gap_thr=0.05, stop_loss=0.08, use_fund=True, cost=0.0015):
    """后处理执行规则：Top-N、持有hold日、次日开盘买入、跳开>gap_thr不买、跌破entry*(1-stop)止损。"""
    op, hi, lo, cl = _ohlc()
    pos = {d: i for i, d in enumerate(idx)}
    rb = [d for d in idx[::hold] if not pred.loc[d].isna().all()]
    port = pd.Series(0.0, index=idx); trades = []
    for d in rb:
        i = pos[d]
        if i + 1 >= len(idx):
            break
        s = pred.loc[d].dropna()
        if use_fund:
            ok = (trend.loc[d].reindex(s.index).fillna(False)) & (q.loc[d].reindex(s.index).fillna(False))
            s = s[ok]
        cand = list(s.nlargest(top_n * 2).index)        # 多取些,过滤跳开后凑够
        entry_i = i + 1                                  # 次日开盘
        picks = []
        for sym in cand:
            o0, pc = op.iloc[entry_i][sym], cl.iloc[i][sym]
            if pd.isna(o0) or pd.isna(pc) or pc <= 0:
                continue
            if o0 / pc - 1 > gap_thr:                    # 跳开过滤
                continue
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
            exit_i, exit_px = end_i, cl.iloc[end_i][sym]
            for j in range(entry_i, end_i + 1):          # 逐日查止损(按最低)
                if lo.iloc[j][sym] <= stop_px:
                    exit_i, exit_px = j, stop_px
                    break
            if pd.isna(exit_px):
                continue
            r = exit_px / ent - 1 - 2 * cost
            trades.append({"entry": idx[entry_i], "exit": idx[exit_i], "symbol": sym, "ret": r,
                           "hold": exit_i - entry_i, "stopped": bool(exit_px == stop_px)})
    # 用逐笔等权构日度组合(止损日后转现金)
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


def main():
    print("训练纯信号模型(标签=未来5/10/20日收益的横截面rank, 多视野集成, 月度扩张窗口+purge)...", flush=True)
    pred, meta = train_signal()
    pred.to_parquet(DD / "ml_signal.parquet")
    json.dump(meta, open(DD / "ml_signal_meta.json", "w"), ensure_ascii=False, default=float)
    print(f"信号: 训练样本{meta['n_train_samples']} | 均IC {meta['mean_ic']} | 出分日 {pred.notna().any(axis=1).sum()}")
    print("\n后处理规则示例(同一信号,不同持有期):")
    for hold in (3, 5, 10):
        port, tr = simulate(pred, hold=hold)
        m = metrics(port)
        print(f"  持有{hold}日 Top20 跳开5% 止损8% 基本面: 年化{m['cagr']*100:+.0f}% 夏普{m['sharpe']:.2f} "
              f"回撤{m['maxdd']*100:+.0f}% | 笔数{len(tr)} 止损率{tr['stopped'].mean()*100:.0f}%")
    print("预训练完成: ml_signal.parquet + ml_signal_meta.json")


if __name__ == "__main__":
    main()
