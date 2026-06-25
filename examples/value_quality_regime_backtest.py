"""价值-质量多头 + Markov 状态闸门：WF 历史 + 2026 真·样本外回测（自包含）。

直接从规范数据源构建因子，复现 docs/value-quality-regime.md 的核心数字。
数据（均在 ~/.local/share/quantlab/fundamentals/）：
  daily_ohlcv.parquet   行情 OHLCV+成交额 2016-2025（含退市，无幸存者偏差）
  market_panel.parquet  季末 pe_ttm/pb/total_mv 2016-2025
  tushare_features.parquet  PIT 财务（扣非ROE/毛利，按公告日 as-of）
  daily_2026.parquet    2026 真·样本外行情+估值（research/pull_2026.py 拉取）

运行：python3 examples/value_quality_regime_backtest.py
"""
import pathlib
import numpy as np
import pandas as pd

from quantlab.strategies.value_quality_regime import (
    value_quality_score, long_only_returns, RegimeGate, gated_returns,
)
from quantlab.eval import deflated_sharpe_ratio

D = pathlib.Path.home() / ".local/share/quantlab/fundamentals"
ANN = 242
CLIP = 0.21          # A 股涨跌停区间，剔停牌复牌跳空


def _universe(oh, n=500, asof="2023-01-01"):
    """用 asof 之前的平均成交额取流动性 Top-N（避免选池前视）。"""
    hist = oh[oh.trade_date < asof]
    return set(hist.groupby("symbol")["amount"].mean().nlargest(n).index)


def _pit(tf, col, dates, syms):
    """按公告日 as-of 的 PIT 因子 → 宽表横截面 rank。"""
    cal = pd.DataFrame({"announce_date": pd.DatetimeIndex(dates)})
    out = {}
    for s, g in tf.groupby("symbol"):
        g = g.dropna(subset=[col]).sort_values("announce_date")
        if len(g):
            out[s] = pd.merge_asof(cal, g[["announce_date", col]], on="announce_date").set_index("announce_date")[col]
    return pd.DataFrame(out).reindex(index=pd.DatetimeIndex(dates), columns=syms)


def _segment_returns(prices_close, topn=50, k=20):
    """给定一段 adj_close 宽表，返回 (fwd_returns, 等权基准日收益)。"""
    rf = prices_close.pct_change(fill_method=None).shift(-1).clip(-CLIP, CLIP)
    return rf, rf.mean(axis=1)


# --------------------------------------------------------------------------- #
# 1) 历史段 2016-2025：行情/估值/质量 → 价值质量分 → 多头收益
# --------------------------------------------------------------------------- #
oh = pd.read_parquet(D / "daily_ohlcv.parquet")
oh["symbol"] = oh["symbol"].astype(str).str.zfill(6)
liq = _universe(oh)
oh = oh[oh.symbol.isin(liq)]
dates_h = pd.DatetimeIndex(sorted(oh.trade_date.unique()))
syms = sorted(oh.symbol.unique())
close_h = oh.pivot(index="trade_date", columns="symbol", values="adj_close").reindex(index=dates_h, columns=syms)

# 估值：季末快照按日价重建（防前视）
mp = pd.read_parquet(D / "market_panel.parquet"); mp["symbol"] = mp["symbol"].astype(str).str.zfill(6)
mp = mp[mp.symbol.isin(liq)]
def q2d(col):
    q = mp.pivot(index="trade_date", columns="symbol", values=col).reindex(columns=syms)
    return q.reindex(index=dates_h.union(q.index)).sort_index().ffill().reindex(dates_h)
ratio = close_h / q2d("adj_close")
inv_pb_h = (1.0 / q2d("pb")) / ratio
inv_pe_h = (1.0 / q2d("pe_ttm")) / ratio

tf = pd.read_parquet(D / "tushare_features.parquet")[["symbol", "announce_date", "roe_dedt", "gross_margin"]].copy()
tf["symbol"] = tf["symbol"].astype(str).str.zfill(6); tf = tf[tf.symbol.isin(liq)]
tf = tf.dropna(subset=["announce_date"]).sort_values("announce_date")
roe_h = _pit(tf, "roe_dedt", dates_h, syms); gm_h = _pit(tf, "gross_margin", dates_h, syms)

score_h = value_quality_score(inv_pb_h, inv_pe_h, roe_h, gm_h)
rf_h, mkt_h = _segment_returns(close_h)
strat_h = long_only_returns(score_h, rf_h)

# --------------------------------------------------------------------------- #
# 2) 2026 真·样本外段
# --------------------------------------------------------------------------- #
p26 = pd.read_parquet(D / "daily_2026.parquet"); p26["symbol"] = p26["symbol"].astype(str).str.zfill(6)
p26 = p26[p26.symbol.isin(liq)]
dates_26 = pd.DatetimeIndex(sorted(p26.trade_date.unique())); s26 = sorted(p26.symbol.unique())
def w26(c): return p26.pivot(index="trade_date", columns="symbol", values=c).reindex(index=dates_26, columns=s26)
inv_pb_26 = w26("pb").rdiv(1.0); inv_pe_26 = w26("pe_ttm").rdiv(1.0)
roe_26 = _pit(tf, "roe_dedt", dates_26, s26); gm_26 = _pit(tf, "gross_margin", dates_26, s26)
score_26 = value_quality_score(inv_pb_26, inv_pe_26, roe_26, gm_26)
rf_26, mkt_26 = _segment_returns(w26("adj_close"))
strat_26 = long_only_returns(score_26, rf_26)

# --------------------------------------------------------------------------- #
# 3) 拼接 + Markov 状态闸门（只用 ≤2025 拟合，2026 纯 OOS）
# --------------------------------------------------------------------------- #
strat = pd.concat([strat_h[strat_h.index >= "2020-01-01"], strat_26]).sort_index()
strat = strat[~strat.index.duplicated()].dropna()
is25 = strat.index <= "2025-12-31"; is26 = strat.index >= "2026-01-01"

gate = RegimeGate(hard=True).fit(strat[is25])
strat_gated = gated_returns(strat, gate)


def stats(r, mask):
    x = r[mask].dropna()
    cum = (1 + x).prod() - 1
    ann = x.mean() * ANN
    sh = x.mean() / (x.std() + 1e-12) * np.sqrt(ANN)
    return cum, ann, sh


print("=" * 78)
print("价值质量多头 + Markov 状态闸门  (Markov 仅 ≤2025 拟合, 2026 纯样本外)")
print("=" * 78)
print(f"股票池 liq500 | 历史 {dates_h.min().date()}~{dates_h.max().date()} | 2026 {dates_26.min().date()}~{dates_26.max().date()}")
print(f"闸门可用(statsmodels): {gate.fitted}")
for tag, r in [("不闸门 (原价值质量)", strat), ("Markov 状态闸门", strat_gated)]:
    c25, a25, s25 = stats(r, is25); c26, a26, s26 = stats(r, is26)
    print(f"\n[{tag}]")
    print(f"  ≤2025 :  年化 {a25*100:+6.1f}%   夏普 {s25:+.2f}")
    print(f"  2026  :  累计 {c26*100:+6.1f}%   夏普 {s26:+.2f}   (基准 {stats(mkt_26, mkt_26.index>='2000')[0]*100:+.1f}%)")
ds = deflated_sharpe_ratio(strat[is25].dropna(), n_trials=10, trials_sr_std=float(strat[is25].std()*0.5), periods=ANN)
print(f"\n≤2025 不闸门 DSR(校正10次)={ds['dsr']:.3f}")
print(f"2026 在场比例(闸门): {(gate.exposure(strat)[is26]).mean():.2f}  (低=多在风险态空仓避险)")
print("\n结论：闸门是纪律化风控(逆风期避险)，非新 alpha；详见 docs/value-quality-regime.md")
