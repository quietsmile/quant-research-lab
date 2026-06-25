"""按 stable_largecap.json 的最优参数，重算各大盘策略净值曲线(+沪深300)存盘，供看板画图。

只算最优配置(5个)，快。跑法：python examples/stable_largecap_nav.py
"""
import warnings; warnings.filterwarnings("ignore")
import json, pathlib
import numpy as np, pandas as pd
from examples.strategy_family import run, size_neutral, trend, q, idx
from examples.stable_largecap import STRATS, largecap, HS300

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")


def main():
    best = json.load(open(DD / "stable_largecap.json"))
    navs = {}
    for name, (score_fn, _space) in STRATS.items():
        P = best[name]["P"]
        sel = trend & q & largecap(P["mv_top"])
        sc = size_neutral(score_fn(P))
        p, nh = run(sel, sc, int(P["topn"]), int(P["rebal"]))
        navs[name] = (1 + p.fillna(0)).cumprod()
        print(f"{name}: 净值末值 {navs[name].iloc[-1]:.2f}")
    navs["沪深300"] = (1 + HS300.fillna(0)).cumprod()
    out = pd.DataFrame(navs).reindex(idx)
    out.to_parquet(DD / "stable_largecap_nav.parquet")
    print(f"已存 {len(out)} 天净值 → stable_largecap_nav.parquet")


if __name__ == "__main__":
    main()
