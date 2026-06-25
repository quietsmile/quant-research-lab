"""测试'主力资金转流出就第一时间逃'是不是更好的卖出策略。
同一买入逻辑(热门板块+上升趋势+主力流入), 对比每日退出方式:
  A 不退出(持到调仓)  B 跌破X日均线退出  C 主力资金转流出退出(跟着钱逃)  D 两者任一
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pathlib
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data"); ANN = 242
g = lambda n: pd.read_parquet(DD / f"pullback_{n}.parquet")
close = g("close"); ret = g("ret"); sret = g("sret"); mv = g("mv"); bench = g("bench"); sector = g("sector")["industry"]
for x in [close, ret, sret, mv, bench]: x.index = pd.to_datetime(x.index)
zl = pd.read_parquet(DD / "moneyflow_zlrate.parquet"); zl.index = pd.to_datetime(zl.index)
syms = list(close.columns); zl = zl.reindex(columns=syms)
idx = close.index[close.index >= "2020-01-01"]
close, ret, sret, mv = [x.reindex(idx) for x in (close, ret, sret, mv)]; zl = zl.reindex(idx)

# 买入候选(热门板块top5 + 价>20日均线 + 主力近5日净流入>0)
snav = (1 + sret.fillna(0)).cumprod()
hot = (snav / snav.shift(30) - 1).rank(axis=1, ascending=False) <= 5
smap = sector.reindex(syms).values; sh = hot.reindex(columns=smap); sh.columns = syms; sh = sh.reindex(idx).fillna(False)
ma = close.rolling(20, min_periods=10).mean()
entry = sh & (close >= ma) & (zl.rolling(5, min_periods=1).mean() > 0) & (mv > 500000) & ret.notna()
# 退出条件(每日)
exit_ma = close < ma * (1 - 0.0)                          # 跌破均线
exit_mf = zl.rolling(3, min_periods=1).mean() < 0         # 主力近3日转净流出
rfwd = ret.shift(-1).clip(-0.21, 0.21); rebal = 5

def run(exit_mask, cost=0.001):
    rb = np.zeros(len(idx), bool); rb[::rebal] = True
    held = pd.DataFrame(False, index=idx, columns=syms)
    cur = pd.Series(False, index=syms)
    for i, d in enumerate(idx):
        if rb[i]:
            cur = entry.loc[d].fillna(False)              # 调仓日按入场条件重选
        else:
            if exit_mask is not None:
                cur = cur & ~exit_mask.loc[d].fillna(False)  # 每日退出: 触发就剔除(到下次调仓才再进)
        held.loc[d] = cur
    w = held.astype(float).div(held.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    port = (w * rfwd).sum(axis=1) - (w - w.shift(1)).abs().sum(axis=1) * cost
    return port

def stat(p):
    p = p.fillna(0); nav = (1 + p).cumprod(); yrs = len(p) / ANN
    cagr = nav.iloc[-1] ** (1 / yrs) - 1; sh = p.mean() / (p.std() + 1e-12) * np.sqrt(ANN)
    dd = (nav / nav.cummax() - 1).min()
    by = {d.year: round(((1 + p[p.index.year == d.year]).prod() - 1) * 100) for d in p.index[::240]}
    return cagr, sh, dd, by

print("买入=热门板块top5+价>20MA+主力流入; 5日调仓。对比退出方式:", flush=True)
for tag, em in [("A 不退出", None), ("B 跌破均线退出", exit_ma), ("C 主力转流出退出(跟着钱逃)", exit_mf), ("D 任一触发退出", exit_ma | exit_mf)]:
    c, s, dd, by = stat(run(em))
    print(f"[{tag:22s}] 年化{c*100:+5.0f}% 夏普{s:+.2f} 最大回撤{dd*100:4.0f}% 逐年{by}", flush=True)
b = bench["等权全市场"].reindex(idx).fillna(0)
print(f"[等权基准]              年化{((1+b).prod())**(ANN/len(b))*100-100:+.0f}% 回撤{((1+b).cumprod()/(1+b).cumprod().cummax()-1).min()*100:.0f}%", flush=True)
