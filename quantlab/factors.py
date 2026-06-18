"""横截面因子工具（**按调仓截面现算，绝不预存**）。

防泄漏纪律：去极值/标准化/中性化都只能用"当期截面"的统计量。若在全样本上
预先算好存起来，就用到了未来分布信息 = 前视。故这些函数都作用于**单期截面
Series**，由回测在每个调仓日临时调用。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """分位数去极值（仅用本截面分位）。"""
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def winsorize_mad(s: pd.Series, n: float = 3.0) -> pd.Series:
    """MAD 去极值：中位数 ± n×(1.4826·MAD)。"""
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    scale = 1.4826 * mad
    return s.clip(med - n * scale, med + n * scale)


def zscore(s: pd.Series) -> pd.Series:
    """截面 z-score 标准化。"""
    std = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - s.mean()) / std


def rank_pct(s: pd.Series) -> pd.Series:
    """截面百分位排名 [0,1]（对单调变换稳健，常用于打分）。"""
    return s.rank(pct=True)


def neutralize(s: pd.Series, groups: pd.Series) -> pd.Series:
    """行业/分组中性化：减去每个分组的均值（去除分组系统性差异）。"""
    df = pd.DataFrame({"v": s, "g": groups})
    return df["v"] - df.groupby("g")["v"].transform("mean")
