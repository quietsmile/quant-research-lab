"""价值-质量多头 + Markov 状态闸门策略（横截面多因子）。

这是 2026-06 一轮系统化研究（16 次迭代、含 GTJA191/WQ101/Alpha158 共 ~450 个量价因子
的证伪、转向基本面、再做 Walk-Forward + Deflated Sharpe + 2026 真·样本外）的**蒸馏产物**。

核心结论（详见 docs/value-quality-regime.md）：
  1. 纯量价因子的多空在 Walk-Forward 下不稳健（2022 崩盘），A 股稳健的边际在「便宜 + 高质量」。
  2. 价值-质量多头（1/PB, 1/PE, 扣非ROE, 毛利率）WF 2020-2025 IR≈0.94，但 2025-26 深度逆风。
  3. 给策略自身净值加 **Markov 2 状态闸门**（只用训练期拟合、因果应用），能在价值逆风期自动避险：
     2026 真·OOS 从 -7.8% 改善到约 +1.6%，且不损害历史夏普——**这是纪律化风控，不是新 alpha**。

设计原则：
  - 全部为**横截面 rank**因子，等权合成，避免量纲/异常值问题；
  - **只做多**（A 股做空难，且研究发现 2025 的回撤主要来自做空腿）；
  - 闸门用状态持续模型（非滞后的收益追涨），抗噪声——这是「test-time 适应」在量化里能成立的形态。

依赖：numpy / pandas；闸门可选依赖 statsmodels（缺失时优雅退化为不闸门）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "value_quality_score",
    "long_only_returns",
    "RegimeGate",
    "gated_returns",
]


# --------------------------------------------------------------------------- #
# 1) 因子合成
# --------------------------------------------------------------------------- #
def _csrank(wide: pd.DataFrame) -> pd.DataFrame:
    """逐日横截面百分位 rank（0~1，越大越好）。"""
    return wide.rank(axis=1, pct=True)


def value_quality_score(
    inv_pb: pd.DataFrame,
    inv_pe: pd.DataFrame,
    roe: pd.DataFrame,
    gross_margin: pd.DataFrame,
) -> pd.DataFrame:
    """价值-质量综合分 = 4 个因子横截面 rank 的等权平均（越高=越便宜且越优质）。

    参数均为宽表 (index=date, columns=symbol)，数值越大代表该维度越好：
      inv_pb       : 1 / 市净率（账面价值溢价，BP）
      inv_pe       : 1 / 市盈率TTM（盈利收益率，EP）
      roe          : 扣非 ROE（质量）
      gross_margin : 毛利率（质量；研究中 2024-25 最稳健的单因子）

    注意：4 个因子在历史训练期 IC 同向为正，故直接等权 rank 平均；不含小市值（研究发现
    小市值是 2021 后的陷阱）、不含成长（拥挤）。
    """
    ranks = [_csrank(x) for x in (inv_pb, inv_pe, roe, gross_margin)]
    return pd.concat(ranks).groupby(level=0).mean()


# --------------------------------------------------------------------------- #
# 2) 只做多 TopN 回测
# --------------------------------------------------------------------------- #
def long_only_returns(
    score: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    topn: int = 50,
    rebalance: int = 20,
    open_cost: float = 0.0005,
    close_cost: float = 0.0015,
) -> pd.Series:
    """按综合分选 TopN 等权多头，每 ``rebalance`` 个交易日调仓，扣换手成本。

    score        : 综合分宽表 (date×symbol)
    fwd_returns   : 次日收益宽表 (date×symbol)，fwd_returns.loc[d] = d→d+1 实现收益
                    （建议预先 clip 到 ±0.21 即 A 股涨跌停区间，剔除停牌复牌跳空）
    返回：每日组合净收益序列（已扣成本）。
    """
    held: set[str] = set()
    port = pd.Series(0.0, index=score.index)
    cost = pd.Series(0.0, index=score.index)
    for i, d in enumerate(score.index):
        if i % rebalance == 0:
            row = score.loc[d].dropna()
            if len(row) >= topn:
                sel = set(row.nlargest(topn).index)
                cost.loc[d] = len(sel ^ held) / max(len(sel), 1) * (open_cost + close_cost)
                held = sel
        if held:
            cols = [s for s in held if s in fwd_returns.columns]
            port.loc[d] = fwd_returns.loc[d, cols].mean()
    return port - cost


# --------------------------------------------------------------------------- #
# 3) Markov 状态闸门（风控）
# --------------------------------------------------------------------------- #
class RegimeGate:
    """对一条收益序列拟合 2 状态 Markov-switching（高斯，状态间方差可变），
    把均值更低的那个状态判为「风险态」，在该状态下降低敞口。

    关键纪律（防前视）：
      - ``fit`` 只用训练期收益估计参数；
      - ``exposure`` 用 **固定训练参数** 在全样本上跑 filtered 概率（每个 t 只用 ≤t 的数据），
        并对敞口再 shift 1 天，确保 t 日的仓位只依赖 t-1 及之前的信息。

    缺少 statsmodels 时 ``fitted=False``，exposure 恒为 1（等价不闸门）。
    """

    def __init__(self, hard: bool = True):
        self.hard = hard          # True: 风险态空仓(0/1)；False: 敞口=P(好态) 连续缩放
        self.fitted = False
        self._params = None
        self._bad = None

    def fit(self, train_returns: pd.Series) -> "RegimeGate":
        try:
            from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        except Exception:
            self.fitted = False
            return self
        y = pd.Series(train_returns).dropna()
        mod = MarkovRegression(y.values, k_regimes=2, trend="c", switching_variance=True)
        res = mod.fit(disp=False)
        self._params = res.params
        # 训练期内，哪个状态的收益均值更低 = 风险态
        probs = res.filtered_marginal_probabilities
        means = [y.values[probs[:, g] > 0.5].mean() if (probs[:, g] > 0.5).any() else np.inf
                 for g in (0, 1)]
        self._bad = int(np.argmin(means))
        self.fitted = True
        return self

    def prob_bad(self, returns: pd.Series) -> pd.Series:
        """全样本风险态 filtered 概率（固定训练参数，每个 t 因果）。"""
        if not self.fitted:
            return pd.Series(0.0, index=returns.index)
        from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
        r = pd.Series(returns)
        mod = MarkovRegression(r.values, k_regimes=2, trend="c", switching_variance=True)
        filt = mod.filter(self._params)
        return pd.Series(filt.filtered_marginal_probabilities[:, self._bad], index=r.index)

    def exposure(self, returns: pd.Series) -> pd.Series:
        """目标敞口序列 ∈ [0,1]，已 shift 1 天保证因果。"""
        if not self.fitted:
            return pd.Series(1.0, index=returns.index)
        pbad = self.prob_bad(returns).shift(1)
        exp = (pbad < 0.5).astype(float) if self.hard else (1.0 - pbad).clip(0.0, 1.0)
        return exp.fillna(1.0)


def gated_returns(strategy_returns: pd.Series, gate: RegimeGate) -> pd.Series:
    """把状态闸门敞口乘到策略收益上（风险态减仓/空仓）。"""
    return strategy_returns * gate.exposure(strategy_returns)
