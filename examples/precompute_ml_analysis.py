"""预计算 ML 交易策略的各项指标与拆解,存 dashboard_data/ml_trade_analysis.json,供看板展示。

包含:信号质量(IC/ICIR/IC时序) + 分层单调性(10层+头部) + breadth扫描&对冲后alpha +
容量曲线 + 乐观度拆解。跑法：python examples/precompute_ml_analysis.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, pathlib
import numpy as np, pandas as pd
from examples.ml_trade import load_signal, simulate, metrics, syms, _ohlc
from examples.strategy_family import idx
from examples.ml_signal_deciles import layered
from examples.ml_breadth_hedge import BENCH, hedged_sharpe, MVRANK, dd
from quantlab.data.tushare_adapter import load_daily_ohlcv

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
ANN = 242


def main():
    pred = load_signal(); cl = _ohlc()[3]
    meta = json.load(open(DD / "ml_signal_meta.json"))
    ic = pd.Series([x["ic"] for x in meta["ic"]]); icd = pd.to_datetime([x["date"] for x in meta["ic"]])
    icir = float(ic.mean() / (ic.std() + 1e-12))
    out = {"signal": {"label": meta["label"], "retrain": meta.get("retrain"), "feats": len(meta["feats"]),
                      "mean_ic": meta["mean_ic"], "icir": round(icir, 3), "win": round(float((ic > 0).mean()), 3),
                      "t": round(icir * np.sqrt(len(ic)), 1)}}
    # IC 时序(20日平滑,降采样存)
    s = ic.rolling(20).mean(); out["ic_ts"] = [{"date": str(d.date()), "ic": round(float(v), 4)}
                                               for d, v in zip(icd[::3], s.values[::3]) if pd.notna(v)]
    # 分层单调性
    dm, hm, base = layered(pred, cl, None)
    dmp, hmp, basep = layered(pred, cl, True)
    out["deciles"] = {"pure": [round(x, 2) for x in dm], "pool": [round(x, 2) for x in dmp],
                      "head": {k: round(v, 2) for k, v in hm.items()},
                      "head_pool": {k: round(v, 2) for k, v in hmp.items()}}
    # breadth 扫描 + 对冲后 alpha + 容量数据
    o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
    amt = o.pivot_table(index="trade_date", columns="symbol", values="amount").reindex(index=idx, columns=syms) * 1000
    br_ret, bc = BENCH["中证500(IC)"]
    breadth = []; cap_trades = None
    for N in (20, 50, 100, 150, 200, 300):
        port, tr = simulate(pred, hold=10, top_n=N, realistic=True, exclude_st=True, use_fund=True)
        m = metrics(port)
        hs, beta = hedged_sharpe(port, br_ret, bc)
        sz = float(np.nanmean([MVRANK.loc[t["entry"], t["symbol"]] for _, t in tr.iterrows()
                               if t["symbol"] in MVRANK.columns])) if len(tr) else None
        q1 = port[(port.index >= "2024-01-01") & (port.index <= "2024-03-31")]
        q1nav = (1 + q1.fillna(0)).cumprod()
        breadth.append({"N": N, "long_sharpe": round(m["sharpe"], 2), "long_cagr": round(m["cagr"], 3),
                        "maxdd": round(m["maxdd"], 3), "calmar": round(m["calmar"], 2),
                        "hedge_net_ic": round(float(hs), 2), "ic_beta": round(float(beta), 2), "size": round(sz, 2),
                        "q1ret": round(float(q1nav.iloc[-1] - 1), 3), "q1dd": round(float(dd(q1nav)), 3)})
        if N == 20:
            tr["amt"] = [amt.loc[t["entry"], t["symbol"]] if t["symbol"] in amt.columns else np.nan for _, t in tr.iterrows()]
            cap_trades = tr.dropna(subset=["amt"]); cap_base = m["cagr"]
    out["breadth"] = breadth
    # 容量曲线(Top-20)
    cap = []
    for aum in [1e6, 5e6, 1e7, 3e7, 1e8, 3e8, 1e9]:
        ratio = (aum / 20) / cap_trades["amt"]
        impact = 1.0 * 0.025 * np.sqrt((ratio - 0.005).clip(lower=0, upper=5))
        drag = (ANN / 10) * impact.mean() * 2
        cap.append({"aum_yi": round(aum / 1e8, 3), "over10": round(float((ratio > 0.10).mean()), 3),
                    "drag": round(float(drag), 3), "net_cagr": round(float(cap_base - drag), 3)})
    out["capacity"] = cap
    # 乐观度拆解(静态,来自 optimism_budget.py 文档值)
    out["optimism"] = [{"step": "严格基线", "cagr": 0.10}, {"step": "+理想撮合", "cagr": 0.10},
                       {"step": "+去成本", "cagr": 0.14}, {"step": "+日频调仓", "cagr": 0.16},
                       {"step": "+下沉小盘", "cagr": 0.19}, {"step": "+样本内泄漏", "cagr": 0.51}]
    json.dump(out, open(DD / "ml_trade_analysis.json", "w"), ensure_ascii=False, default=float)
    print(f"已存 ml_trade_analysis.json | IC {out['signal']['mean_ic']} ICIR {out['signal']['icir']} | "
          f"breadth {len(breadth)}档 | 容量 {len(cap)}档")


if __name__ == "__main__":
    main()
