"""Barra 式风格因子暴露分析。

把策略收益回归到一组风格因子(市场/规模/价值/动量/波动/成长/流动性)上，得到：
- 各风格的**暴露(beta)** 与 t 值 —— 策略的钱来自哪些风格;
- **alpha(年化)** 与 t 值 —— 剔除风格后真正独立的超额;
- R² —— 风格能解释多少。

风格因子用截面 top-bottom 三分位多空收益构造，方向已规范为"溢价为正"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:  # noqa: BLE001
    _HAS_SM = False


def ls_factor(char: pd.DataFrame, rfwd: pd.DataFrame, direction: int = 1, frac: float = 1 / 3) -> pd.Series:
    """按特征截面分三分位，下一期收益的 (高 − 低)×direction 多空序列。

    direction=+1：高特征−低特征；-1：低特征−高特征(如低波动/低换手溢价)。
    """
    out = {}
    for d in char.index:
        c = char.loc[d].dropna()
        r = rfwd.loc[d]
        if len(c) < 30:
            continue
        k = max(1, int(len(c) * frac))
        hi = c.nlargest(k).index
        lo = c.nsmallest(k).index
        out[d] = direction * (r.reindex(hi).mean() - r.reindex(lo).mean())
    return pd.Series(out).sort_index()


def build_style_factors(rfwd, *, logmv=None, ep=None, mom=None, vol=None,
                        growth=None, turnover=None, market=None) -> pd.DataFrame:
    """构造标准风格因子收益表(列=因子)。传入哪个特征就建哪个。

    规范方向（正=该风格溢价）：规模=小减大，价值=高EP减低，动量=高减低，
    波动=低减高，成长=高减低，流动性=低换手减高。
    """
    cols = {}
    if market is not None:
        cols["MKT"] = market
    if logmv is not None:
        cols["SIZE(小)"] = ls_factor(logmv, rfwd, -1)
    if ep is not None:
        cols["VALUE(高EP)"] = ls_factor(ep, rfwd, +1)
    if mom is not None:
        cols["MOM(高动量)"] = ls_factor(mom, rfwd, +1)
    if vol is not None:
        cols["VOL(低波)"] = ls_factor(vol, rfwd, -1)
    if growth is not None:
        cols["GROWTH(高增长)"] = ls_factor(growth, rfwd, +1)
    if turnover is not None:
        cols["LIQ(低换手)"] = ls_factor(turnover, rfwd, -1)
    return pd.DataFrame(cols).dropna(how="all")


def barra_exposure(strategy_ret: pd.Series, style: pd.DataFrame, ann: int = 242) -> dict:
    """策略收益 ~ 风格因子 OLS：返回各 beta/t、年化 alpha/t、R²。"""
    dat = pd.concat([strategy_ret.rename("y"), style], axis=1).dropna()
    if len(dat) < 20 or not _HAS_SM:
        return {}
    X = sm.add_constant(dat[style.columns])
    res = sm.OLS(dat["y"], X).fit()
    exp = {c: float(res.params[c]) for c in style.columns}
    tval = {c: float(res.tvalues[c]) for c in style.columns}
    return {
        "exposure": exp, "tstat": tval,
        "alpha_ann": float(res.params["const"] * ann),
        "alpha_t": float(res.tvalues["const"]),
        "r2": float(res.rsquared),
        "n": int(len(dat)),
    }


def format_exposure(b: dict, name: str = "") -> str:
    if not b:
        return f"{name}: 无法回归(缺 statsmodels 或样本不足)"
    parts = [f"{k} {v:+.2f}(t{b['tstat'][k]:+.1f})" for k, v in b["exposure"].items()]
    return (f"{name} | α年化 {b['alpha_ann']:+.1%}(t{b['alpha_t']:+.1f}) | R² {b['r2']:.0%} | "
            + " ".join(parts))
