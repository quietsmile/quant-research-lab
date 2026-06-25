"""(B') 持仓数扫描 × 市值暴露 × 指数对冲(IF/IC/IM,扣贴水) × 分年度 + MDD + 2024Q1。

回答三件事(而非单点夏普): ① 最优持仓数(√breadth增益 vs edge衰减的平衡点);
② 组合是不是小盘beta马甲(市值暴露 + 用匹配基准对冲后还剩多少); ③ 扛不扛得住2024年初小微盘危机。

跑法：python examples/ml_breadth_hedge.py
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib
import numpy as np, pandas as pd
from examples.ml_trade import load_signal, simulate, metrics, syms
from examples.strategy_family import idx, mv

ANN = 242
F = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
DD = pathlib.Path("dashboard_data")
MVRANK = mv.reindex(index=idx, columns=syms).rank(axis=1, pct=True)   # 市值百分位(越大越大盘)
# 基准日收益 + 股指期货年化贴水成本(对冲方持续付出,粗估)
B = pd.read_parquet(DD / "pullback_bench.parquet"); B.index = pd.to_datetime(B.index)
BENCH = {"沪深300(IF)": (B["沪深300"].reindex(idx), 0.03)}
for f, nm, cost in [("idx500.parquet", "中证500(IC)", 0.07), ("idx1000.parquet", "中证1000(IM)", 0.09)]:
    s = pd.read_parquet(F / f)["close"]; s.index = pd.to_datetime(s.index)
    BENCH[nm] = (s.pct_change().reindex(idx), cost)


def dd(nav):
    return (nav / nav.cummax() - 1).min()


def hedged_sharpe(port, bench, basis, window=60):
    """标准口径:滚动 window 日估 β(再 lag1,PIT 不用未来) → β 中性残差 → 扣贴水。
    注意:绝不可用全样本 np.cov 估 β——那是未来函数。详见 examples/ml_rolling_hedge.py。"""
    a = pd.concat([port, bench], axis=1).dropna(); a.columns = ["p", "b"]
    if len(a) < window + 30:
        return np.nan, np.nan
    beta = (a["p"].rolling(window).cov(a["b"]) / (a["b"].rolling(window).var() + 1e-12)).shift(1)
    res = (a["p"] - beta * a["b"] - basis / ANN).dropna()   # 残差(市场中性) 扣贴水
    return res.mean() / (res.std() + 1e-12) * np.sqrt(ANN), float(beta.dropna().mean())


def main():
    pred = load_signal()
    print(f"{'N':>4s} {'市值%ile':>7s} {'多头夏普':>7s} {'多头MDD':>7s} {'Calmar':>7s} "
          f"{'对冲夏普(IF/IC/IM,扣贴水)':>26s} {'2024Q1收益':>9s} {'2024Q1回撤':>9s}", flush=True)
    rows = []
    for N in (20, 50, 100, 150, 200, 300):
        port, tr = simulate(pred, hold=10, top_n=N, realistic=True, exclude_st=True, use_fund=True)
        m = metrics(port); nav = m["nav"]
        # 市值暴露:持仓在 entry 日的市值百分位均值
        sz = np.nanmean([MVRANK.loc[t["entry"], t["symbol"]] for _, t in tr.iterrows()
                         if t["symbol"] in MVRANK.columns]) if len(tr) else np.nan
        # 2024Q1
        q1 = port[(port.index >= "2024-01-01") & (port.index <= "2024-03-31")]
        q1nav = (1 + q1.fillna(0)).cumprod()
        q1ret = q1nav.iloc[-1] - 1 if len(q1nav) else np.nan
        q1dd = dd(q1nav) if len(q1nav) else np.nan
        hs = {nm: hedged_sharpe(port, br, c) for nm, (br, c) in BENCH.items()}
        hs_str = " ".join(f"{hs[k][0]:.2f}" for k in BENCH)
        print(f"{N:>4d} {sz:>7.2f} {m['sharpe']:>7.2f} {m['maxdd']*100:>+6.0f}% {m['calmar']:>7.2f} "
              f"{hs_str:>26s} {q1ret*100:>+8.1f}% {q1dd*100:>+8.1f}%", flush=True)
        rows.append({"N": N, "sz": sz, "long_sharpe": m["sharpe"], "long_mdd": m["maxdd"],
                     "hedge_IC": hs["中证500(IC)"][0], "beta_IC": hs["中证500(IC)"][1],
                     "q1ret": q1ret, "q1dd": q1dd, "port": port})
    # 分年度(取 N=100 与 N=200 详看)
    print("\n分年度收益(%):")
    for N in (20, 100, 200):
        r = next(x for x in rows if x["N"] == N)["port"]
        ys = {y: round(((1 + r[r.index.year == y].fillna(0)).prod() - 1) * 100) for y in range(2021, 2027)}
        print(f"  N={N}: {ys}")
    best = max(rows, key=lambda x: x["hedge_IC"] if pd.notna(x["hedge_IC"]) else -9)
    print(f"\n对冲夏普(IC)最高点: N={best['N']} 夏普{best['hedge_IC']:.2f} (市值%ile {best['sz']:.2f}, IC-beta {best['beta_IC']:.2f})")
    print("解读: 市值%ile<0.5=偏小盘(对冲该用IC/IM);若对冲后夏普≈0或2024Q1巨亏→小盘beta马甲;"
          "对冲夏普随N先升后降→顶点即最优breadth。")
    pd.DataFrame({f"N{r['N']}": (1 + r["port"].fillna(0)).cumprod() for r in rows}).to_parquet(DD / "ml_breadth_nav.parquet")


if __name__ == "__main__":
    main()
