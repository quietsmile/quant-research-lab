"""(B-1) 特征扩展:25因子子集 vs 接近完整 Alpha158(~100量价因子),比 IC/ICIR。

我们之前只用 ~25 个 Alpha158 式因子;Alpha158=158个。本脚本构造 ~100 个(KBAR + 多窗口
ROC/MA/STD/MAX/MIN/RSV/QTL/CORR/CNT/SUM/VMA/VSTD/WVMA),用同一多视野rank口径比样本外 IC/ICIR。

跑法：python examples/feature_expansion.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from examples.ml_alpha import build_features as build25
from examples.ml_ic_stability import xs_rank, ic_series, stats
from examples.strategy_family import idx, mv
from quantlab.data.tushare_adapter import load_daily_ohlcv

syms = list(mv.columns)
W = [5, 10, 20, 30, 60]


def build_alpha158():
    o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
    pv = lambda c: o.pivot_table(index="trade_date", columns="symbol", values=c).reindex(index=idx, columns=syms)
    O, H, L, C, V = pv("adj_open"), pv("adj_high"), pv("adj_low"), pv("adj_close"), pv("vol")
    ret = C.pct_change(fill_method=None); lv = np.log(V.clip(lower=1))
    F = {}
    # KBAR(单日蜡烛)
    F["KMID"] = (C - O) / (O + 1e-9); F["KLEN"] = (H - L) / (O + 1e-9)
    F["KUP"] = (H - np.maximum(O, C)) / (O + 1e-9); F["KLOW"] = (np.minimum(O, C) - L) / (O + 1e-9)
    F["KSFT"] = (2 * C - H - L) / (O + 1e-9)

    def rcorr(x, y, w):                              # 向量化滚动相关
        ex, ey = x.rolling(w).mean(), y.rolling(w).mean()
        exy = (x * y).rolling(w).mean()
        return (exy - ex * ey) / (x.rolling(w).std() * y.rolling(w).std() + 1e-9)

    up = (ret > 0).astype(float); dn = (ret < 0).astype(float)
    gain = ret.clip(lower=0); loss = (-ret).clip(lower=0)
    av = ret.abs() * V
    for w in W:
        F[f"ROC{w}"] = C.shift(w) / C - 1
        F[f"MA{w}"] = C.rolling(w).mean() / C - 1
        F[f"STD{w}"] = C.rolling(w).std() / C
        F[f"MAX{w}"] = H.rolling(w).max() / C - 1
        F[f"MIN{w}"] = L.rolling(w).min() / C - 1
        F[f"RSV{w}"] = (C - L.rolling(w).min()) / (H.rolling(w).max() - L.rolling(w).min() + 1e-9)
        F[f"QTLU{w}"] = C.rolling(w).quantile(0.8) / C - 1
        F[f"QTLD{w}"] = C.rolling(w).quantile(0.2) / C - 1
        F[f"CORR{w}"] = rcorr(C, lv, w)
        F[f"CORD{w}"] = rcorr(ret, lv.diff(), w)
        F[f"CNTP{w}"] = up.rolling(w).mean()
        F[f"CNTD{w}"] = up.rolling(w).mean() - dn.rolling(w).mean()
        F[f"SUMP{w}"] = gain.rolling(w).sum() / (gain.rolling(w).sum() + loss.rolling(w).sum() + 1e-9)
        F[f"VMA{w}"] = V.rolling(w).mean() / (V + 1e-9) - 1
        F[f"VSTD{w}"] = V.rolling(w).std() / (V + 1e-9)
        F[f"WVMA{w}"] = av.rolling(w).std() / (av.rolling(w).mean() + 1e-9)
    return {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}


def main():
    print("构建特征(25子集 + Alpha158扩展) ...", flush=True)
    F25, _l, close = build25()
    F25 = {k: v.reindex(index=idx, columns=syms) for k, v in F25.items()}
    F158 = build_alpha158()
    lab = lambda h: (close.shift(-h) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    ic_label = lab(10); LBS = [xs_rank(lab(5)), xs_rank(lab(10)), xs_rank(lab(20))]
    print(f"特征数: 子集 {len(F25)} | Alpha158扩展 {len(F158)}", flush=True)
    print(f"\n{'特征集':18s} {'RankIC':>7s} {'ICstd':>7s} {'ICIR':>6s} {'胜率':>5s} {'t值':>6s} {'#因子':>5s}", flush=True)
    for nm, Fd in [("25因子子集(现状)", F25), ("Alpha158扩展(~100)", F158), ("两者合并", {**F25, **F158})]:
        s = ic_series(Fd, LBS, ic_label, step=3, purge=20)
        st = stats(s)
        print(f"{nm:18s} {st['IC']:>7.4f} {st['ICstd']:>7.4f} {st['ICIR']:>6.3f} "
              f"{st['win']*100:>4.0f}% {st['t']:>6.1f} {len(Fd):>5d}", flush=True)
    print("\n判读: 若Alpha158扩展/合并的 RankIC、ICIR 明显高于25子集 → 特征确实是真实杠杆,值得全口径重训;"
          "若持平 → 我们25个已抓住主要信息,加特征只增过拟合。")


if __name__ == "__main__":
    main()
