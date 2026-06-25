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


SIG_H = 10   # 信号的预测视野(纯信号,与持有期解耦——持有期是后处理)


def train_signal(label_h=SIG_H, train_step=2, purge=None):
    """训纯信号模型:label=未来 label_h 日收益,**月度扩张窗口** walk-forward,逐(隔)日出分。

    每个月预测前,用「该月之前的全部数据」重训一次(含最近几个月)——严格 PIT:
    训练样本须在预测起点前 purge 个交易日(默认=label_h),确保其未来收益标签已实现、不泄漏。
    这样每个预测点都用满了它之前的所有数据(尤其近期),而非冻结在去年底。
    训练只学收益预测,不含任何交易规则(跳开/止损/持有期/基本面)——这些都后处理。
    train_step: 训练样本采样间隔(2=隔日,降冗余/提速);预测对所有交易日。
    """
    if purge is None:
        purge = label_h
    F, _label, close = build_features()
    feats = list(F)
    label = (close.shift(-label_h) / close - 1).clip(-0.5, 0.5)
    days = [d for d in idx if d >= pd.Timestamp("2018-01-01")]
    tr_days = days[::train_step]
    rows = []
    for d in tr_days:
        df = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
        df["y"] = label.loc[d]; df["date"] = d; df["symbol"] = df.index
        rows.append(df)
    data = pd.concat(rows).dropna(subset=["y"])
    pred_panel = pd.DataFrame(np.nan, index=idx, columns=syms)
    meta = {"feats": feats, "label": f"未来{label_h}日收益", "n_train_samples": int(len(data)),
            "train_step": train_step, "purge": purge, "retrain": "月度扩张窗口(每月用该月以前全部数据重训)",
            "folds": [], "ic": []}
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
        tr = data[data["date"] <= cut_date]
        if len(tr) < 4000:
            continue
        med = tr[feats].median()
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03, subsample=0.8,
                              colsample_bytree=0.7, min_child_samples=100, n_jobs=4, verbosity=-1)
        m.fit(tr[feats].fillna(med), tr["y"])
        for d in te_days:                                # 预测该月所有交易日(逐日出分)
            X = pd.DataFrame({k: v.loc[d] for k, v in F.items()}).fillna(med)
            pred_panel.loc[d] = m.predict(X[feats])
            ic = pd.Series(pred_panel.loc[d], index=syms).corr(label.loc[d], method="spearman")
            if pd.notna(ic): meta["ic"].append({"date": str(d.date()), "ic": round(float(ic), 4)})
        meta["folds"].append({"month": f"{Y}-{M:02d}", "train_rows": int(len(tr)),
                              "train_end": str(cut_date.date()), "test_days": len(te_days)})
        imp_acc += m.feature_importances_
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
    print("训练纯信号模型(label=未来10日收益, 月度扩张窗口walk-forward+purge, 逐日出分)...", flush=True)
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
