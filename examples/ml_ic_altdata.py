"""另类因子(龙虎榜/融资融券/股东户数)的增量 IC 测试。

构造 PIT 安全的日频另类因子,加入多视野rank模型,对比加入前后的 IC均值/ICIR/胜率/t值。
依赖 examples/pull_altdata.py 的产物。跑法：python examples/ml_ic_altdata.py
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib
import numpy as np, pandas as pd
from examples.ml_alpha import build_features
from examples.strategy_family import idx, mv
from examples.ml_ic_stability import xs_rank, ic_series, stats

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
FUND = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
syms = list(mv.columns)


def _panel(df, val, agg="sum"):
    p = df.pivot_table(index="trade_date", columns="symbol", values=val, aggfunc=agg)
    p.index = pd.to_datetime(p.index)
    return p.reindex(index=idx, columns=syms)


def build_alt():
    """返回 {因子名: 面板} 的 dict，按数据可得性。所有因子均为「截至当日已知」,PIT 安全。"""
    alt = {}
    # 龙虎榜：净买入占流通值(当日盘后披露,可用于次日);不在榜=0
    f = FUND / "lhb_toplist.parquet"
    if f.exists():
        L = pd.read_parquet(f)
        L["lhb_net"] = L["net_amount"] / (L["float_values"].replace(0, np.nan))
        p = _panel(L, "lhb_net", "sum").fillna(0.0)
        alt["lhb_net"] = p
        alt["lhb_net5"] = p.rolling(5).sum()
    # 融资融券：融资余额变化(5/20日);融资余额占比
    f = FUND / "margin_detail.parquet"
    if f.exists():
        M = pd.read_parquet(f)
        rzye = _panel(M, "rzye", "sum")
        alt["rz_chg5"] = rzye / rzye.shift(5) - 1
        alt["rz_chg20"] = rzye / rzye.shift(20) - 1
        alt["rz_level"] = (rzye / (mv.reindex(index=idx, columns=syms) * 1e4 + 1e-9))  # 融资余额/流通市值(元)
    # 股东户数：PIT(按 ann_date 可见),环比变化(下降为正信号 → 取负)
    f = FUND / "holder_number.parquet"
    if f.exists():
        H = pd.read_parquet(f).dropna(subset=["holder_num"])
        H["ann_date"] = pd.to_datetime(H["ann_date"]); H = H.sort_values(["symbol", "ann_date"])
        H["prev"] = H.groupby("symbol")["holder_num"].shift(1)
        H["holder_chg"] = -(H["holder_num"] / H["prev"] - 1)        # 户数降→筹码集中→正
        wide = H.pivot_table(index="ann_date", columns="symbol", values="holder_chg", aggfunc="last")
        alt["holder_chg"] = wide.reindex(index=idx, columns=syms).ffill()  # 按公告日前向填充(PIT)
    return alt


def main():
    print("构建价量因子 + 另类因子 ...", flush=True)
    F, _label, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    lab = lambda h: (close.shift(-h) / close - 1).clip(-0.5, 0.5).reindex(index=idx, columns=syms)
    lab10 = lab(10); r5, r10, r20 = xs_rank(lab(5)), xs_rank(lab10), xs_rank(lab(20))
    LABELS = [r5, r10, r20]

    alt = build_alt()
    print(f"另类因子可得：{list(alt)}", flush=True)
    groups = {
        "基线(仅价量,多视野rank)": None,
        "+龙虎榜": {k: v for k, v in alt.items() if k.startswith("lhb")},
        "+融资融券": {k: v for k, v in alt.items() if k.startswith("rz")},
        "+股东户数": {k: v for k, v in alt.items() if k.startswith("holder")},
        "+全部另类": alt,
    }
    print(f"\n{'配置':28s} {'IC均值':>7s} {'IC标准差':>8s} {'ICIR':>6s} {'胜率':>6s} {'t值':>6s} {'#因子':>5s}", flush=True)
    base_ic = None
    for name, extra in groups.items():
        if extra is not None and len(extra) == 0:
            print(f"{name:28s}  (无数据,跳过)"); continue
        s = ic_series(F, LABELS, lab10, step=3, purge=20, extra_feat=extra)
        st = stats(s); n_extra = 0 if extra is None else len(extra)
        if base_ic is None:
            base_ic = st["IC"]
        delta = f"  (Δ{(st['IC']-base_ic)*1000:+.1f}‰)" if extra is not None else ""
        print(f"{name:28s} {st['IC']:>7.4f} {st['ICstd']:>8.4f} {st['ICIR']:>6.3f} "
              f"{st['win']*100:>5.0f}% {st['t']:>6.1f} {len(F)+n_extra:>5d}{delta}", flush=True)


if __name__ == "__main__":
    main()
