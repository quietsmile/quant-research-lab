"""财务特征工程（第一类：一次性清洗，算好存储）。

把 PIT 业绩表加工成可直接用于因子的特征：
1. 累计→单季：A 股利润表/现金流为年初至今累计，单季 = 本期累计 − 上期累计
   （Q1 单季=Q1 累计）。缺中间季则该单季置 NaN（不瞎减）。
2. TTM：滚动四个单季之和，去季节性。
3. 单季同比：本期单季 / 去年同期单季 − 1。
4. 股票池标记：是否沪深京 A 股、是否 ST、已披露期数（上市/数据年限代理）。

注意（防泄漏）：winsorize / 横截面标准化 / 行业中性化 **不在这里做**，
它们必须在每个调仓截面现算（见 quantlab/factors.py），否则用到未来分布=泄漏。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.fundamentals import FUND_DIR, build_pit_table, is_a_share

FEATURES_FILE = FUND_DIR / "features.parquet"

# 累计口径的流量字段（需要拆单季）
FLOW_FIELDS = ["net_profit", "revenue", "eps", "ocfps"]


def _quarter_no(periods: pd.Series) -> pd.Series:
    return periods.dt.month.map({3: 1, 6: 2, 9: 3, 12: 4})


def to_single_quarter(df: pd.DataFrame, flow_fields: list[str] | None = None) -> pd.DataFrame:
    """累计 → 单季。为每个流量字段新增 <field>_q 列。

    flow_fields 默认 yjbb 的 FLOW_FIELDS；其他源（如 Tushare）可传入自己的累计字段。
    """
    fields = flow_fields if flow_fields is not None else FLOW_FIELDS
    out = df.sort_values(["symbol", "report_period"]).copy()
    out["fy"] = out["report_period"].dt.year
    out["q"] = _quarter_no(out["report_period"])
    for f in fields:
        if f not in out.columns:
            continue
        # 取同一公司、同一财年、上一个季度(q-1)的累计值
        prev = out[["symbol", "fy", "q", f]].copy()
        prev["q"] = prev["q"] + 1                       # 该行作为"下一季"的上期
        prev = prev.rename(columns={f: f + "_prevcum"})
        out = out.merge(prev, on=["symbol", "fy", "q"], how="left")
        out[f + "_q"] = np.where(out["q"] == 1, out[f], out[f] - out[f + "_prevcum"])
        out = out.drop(columns=[f + "_prevcum"])
    return out


def add_ttm(df: pd.DataFrame, flow_fields: list[str] | None = None) -> pd.DataFrame:
    """TTM：对单季字段做滚动 4 季求和（要求连续四季，否则 NaN）。"""
    fields = flow_fields if flow_fields is not None else FLOW_FIELDS
    out = df.sort_values(["symbol", "report_period"]).copy()
    for f in fields:
        qcol = f + "_q"
        if qcol not in out.columns:
            continue
        ttm = out.groupby("symbol")[qcol].rolling(4, min_periods=4).sum() \
            .reset_index(level=0, drop=True)
        out[f + "_ttm"] = ttm
    return out


def add_single_quarter_yoy(df: pd.DataFrame, flow_fields: list[str] | None = None) -> pd.DataFrame:
    """单季同比：<field>_q 对去年同季。新增 <field>_q_yoy。"""
    fields = flow_fields if flow_fields is not None else FLOW_FIELDS
    out = df.copy()
    for f in fields:
        qcol = f + "_q"
        if qcol not in out.columns:
            continue
        prev = out[["symbol", "fy", "q", qcol]].copy()
        prev["fy"] = prev["fy"] + 1                      # 去年同季
        prev = prev.rename(columns={qcol: qcol + "_ly"})
        out = out.merge(prev, on=["symbol", "fy", "q"], how="left")
        denom = out[qcol + "_ly"].abs()
        out[qcol + "_yoy"] = np.where(denom > 0, (out[qcol] - out[qcol + "_ly"]) / denom, np.nan)
        out = out.drop(columns=[qcol + "_ly"])
    return out


def add_universe_flags(df: pd.DataFrame) -> pd.DataFrame:
    """股票池标记：A 股、ST、累计已披露期数。"""
    out = df.copy()
    out["is_a_share"] = out["symbol"].map(is_a_share)
    out["is_st"] = out["name"].astype(str).str.contains("ST", case=False, na=False)
    out = out.sort_values(["symbol", "report_period"])
    out["n_reports"] = out.groupby("symbol").cumcount() + 1   # 上市/数据年限代理
    return out


def build_features(df: pd.DataFrame | None = None, *, save: bool = True) -> pd.DataFrame:
    """从 PIT 表构建完整特征表并（可选）落 parquet。"""
    base = df if df is not None else build_pit_table()
    feat = to_single_quarter(base)
    feat = add_ttm(feat)
    feat = add_single_quarter_yoy(feat)
    feat = add_universe_flags(feat)
    if save:
        FUND_DIR.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(FEATURES_FILE, index=False)
    return feat


def load_features() -> pd.DataFrame:
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"特征表不存在：{FEATURES_FILE}（先运行 build_features）")
    return pd.read_parquet(FEATURES_FILE)
