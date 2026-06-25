"""beta 中性对冲复查(标准做法):滚动窗口估 β(PIT,不用未来) → 残差 alpha 夏普。

纠正 ml_breadth_hedge.py 的全样本 β(未来函数)。对每个 Top-N 组合,用过去 window 日
组合收益对各指数(IF/IC/IM)滚动回归得 β(再 lag 1 日防前视),hedged=组合−β×指数,看残差夏普。
纯 alpha 体检故不扣贴水(贴水只影响落地收益,不影响"alpha在不在");落地另算。

跑法：python examples/ml_rolling_hedge.py
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib
import numpy as np, pandas as pd
from examples.strategy_family import idx

ANN = 242
DD = pathlib.Path("dashboard_data")
Fd = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
B = pd.read_parquet(DD / "pullback_bench.parquet"); B.index = pd.to_datetime(B.index)
BENCH = {"IF沪深300": B["沪深300"].reindex(idx)}
for f, nm in [("idx500.parquet", "IC中证500"), ("idx1000.parquet", "IM中证1000")]:
    s = pd.read_parquet(Fd / f)["close"]; s.index = pd.to_datetime(s.index)
    BENCH[nm] = s.pct_change().reindex(idx)


def sh(p):
    p = p.dropna()
    return p.mean() / (p.std() + 1e-12) * np.sqrt(ANN) if len(p) > 30 else np.nan


def static_beta_hedge(port, bench):                       # 旧法:全样本β(未来函数,仅作对比)
    a = pd.concat([port, bench], axis=1).dropna(); a.columns = ["p", "b"]
    beta = np.cov(a["p"], a["b"])[0, 1] / (a["b"].var() + 1e-12)
    return sh(a["p"] - beta * a["b"]), beta


def rolling_beta_hedge(port, bench, window=60, basis=0.0):  # 标准法:滚动β + lag1(PIT)
    a = pd.concat([port, bench], axis=1).dropna(); a.columns = ["p", "b"]
    beta = (a["p"].rolling(window).cov(a["b"]) / (a["b"].rolling(window).var() + 1e-12)).shift(1)
    res = (a["p"] - beta * a["b"]).dropna()
    return sh(res), float(beta.dropna().mean()), sh(res - basis / ANN)


def main():
    nav = pd.read_parquet(DD / "ml_breadth_nav.parquet")
    nav.index = pd.to_datetime(nav.index)
    BASIS = {"IF沪深300": 0.03, "IC中证500": 0.07, "IM中证1000": 0.09}    # 年化贴水成本(粗估)
    print("滚动60日β中性对冲(PIT) —— 残差alpha夏普: 毛(未扣贴水) / 净(扣贴水)\n")
    print(f"{'组合':6s} {'多头夏普':>7s} | " + " | ".join(f"{k}:毛/净(β̄)" for k in BENCH))
    for col in nav.columns:
        ret = nav[col].pct_change()
        cells = []
        for k, br in BENCH.items():
            gross, bbar, net = rolling_beta_hedge(ret, br, basis=BASIS[k])
            cells.append(f"{gross:+.2f}/{net:+.2f}(β{bbar:.2f})")
        print(f"{col:6s} {sh(ret):>7.2f} | " + " | ".join(cells))
    print("\n判读: 看中证500(IC)那列。毛夏普=alpha在不在(滚动β中性,PIT干净); 净夏普=扣IC贴水7%/年后的可落地值。"
          "\n      alpha随N衰减(集中在头部); 落地还受整数张/容量约束(小资金对冲不干净)。")


if __name__ == "__main__":
    main()
