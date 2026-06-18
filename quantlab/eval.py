"""严肃回测评估：时间切分 + Walk-Forward(带 purge/embargo) + Deflated Sharpe + Bootstrap。

设计原则（针对日频及更低频）：
- 严格按时间顺序；保留一段**完全冻结的 Test**（不参与任何决策）。
- 开发区间内做 Walk-Forward Train/Validation，Train 与 Validation 之间留 **embargo gap
  ≥ 标签持有周期**（防标签重叠泄漏）。
- 用 **Deflated Sharpe Ratio** 校正"试了很多次"带来的选择偏差。
- 用 **block bootstrap** 给收益/夏普置信区间（时间序列有自相关，用块）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from math import erf, sqrt, log, exp

_EULER = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam 近似逆正态
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p = min(max(p, 1e-10), 1 - 1e-10)
    pl = 0.02425
    if p < pl:
        q = sqrt(-2 * log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - pl:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = sqrt(-2 * log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


# ---------------- 时间切分 ----------------

def dev_test_split(dates, test_frac: float = 0.2):
    """按时间顺序切出"开发区间"与"冻结 Test"（Test 为最后一段连续时间）。"""
    dates = sorted(pd.to_datetime(pd.Index(dates)))
    n = len(dates)
    cut = int(n * (1 - test_frac))
    return dates[:cut], dates[cut:]


def walk_forward_splits(dates, train_size: int, val_size: int, step: int, embargo: int = 1):
    """生成 (train_idx, val_idx) 折，按时间滚动；Train 与 Val 间留 embargo 个周期。

    单位均为"周期数"（如季度数）。embargo ≥ 标签持有周期，防重叠泄漏。
    返回每折的 (train日期列表, val日期列表)。
    """
    dates = sorted(pd.to_datetime(pd.Index(dates)))
    n = len(dates)
    folds = []
    start = 0
    while True:
        tr_end = start + train_size
        val_start = tr_end + embargo
        val_end = val_start + val_size
        if val_end > n:
            break
        folds.append((dates[start:tr_end], dates[val_start:val_end]))
        start += step
    return folds


# ---------------- 夏普与显著性 ----------------

def sharpe(returns, periods: int = 4) -> float:
    r = pd.Series(returns).dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * sqrt(periods))


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0, periods: int = 4) -> float:
    """PSR：观测夏普显著高于 benchmark 的概率（考虑样本量、偏度、峰度）。"""
    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 4:
        return float("nan")
    sr = sharpe(r, periods) / sqrt(periods)        # 每期 SR
    sk = float(r.skew()); ku = float(r.kurt()) + 3.0
    srb = sr_benchmark / sqrt(periods)
    denom = sqrt(1 - sk*sr + (ku - 1) / 4 * sr**2)
    return _norm_cdf(((sr - srb) * sqrt(n - 1)) / denom)


def deflated_sharpe_ratio(returns, n_trials: int, trials_sr_std: float, periods: int = 4) -> dict:
    """Deflated Sharpe Ratio：用"试了 n_trials 次"的选择偏差抬高 benchmark 后的 PSR。

    trials_sr_std：各次尝试的(每期)夏普的标准差。返回 dict(dsr, sr_benchmark_annual, psr_vs_0)。
    """
    n = max(int(n_trials), 1)
    # 期望最大夏普(null)：sr0 = E[max] ≈ std * [(1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(N·e))]
    z1 = _norm_ppf(1 - 1.0 / n)
    z2 = _norm_ppf(1 - 1.0 / (n * exp(1)))
    sr0_per = trials_sr_std * ((1 - _EULER) * z1 + _EULER * z2)
    dsr = probabilistic_sharpe_ratio(returns, sr_benchmark=sr0_per * sqrt(periods), periods=periods)
    return {"dsr": dsr,
            "sr_benchmark_annual": sr0_per * sqrt(periods),
            "psr_vs_0": probabilistic_sharpe_ratio(returns, 0.0, periods),
            "n_trials": n}


# ---------------- Block bootstrap ----------------

def block_bootstrap_ci(returns, n_boot: int = 2000, block: int = 4,
                       periods: int = 4, seed: int = 0):
    """移动块自助：给"年化收益"和"年化夏普"的 95% 置信区间（保留自相关）。"""
    r = pd.Series(returns).dropna().to_numpy()
    n = len(r)
    if n < block + 1:
        return {}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    ann_r, ann_s = [], []
    starts_max = n - block
    for _ in range(n_boot):
        idx = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([r[s:s + block] for s in idx])[:n]
        s = pd.Series(sample)
        ann_r.append((1 + s).prod() ** (periods / len(s)) - 1)
        ann_s.append(sharpe(s, periods))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"ann_return_ci": q(ann_r), "sharpe_ci": q(ann_s)}
