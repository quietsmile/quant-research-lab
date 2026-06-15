"""行情清洗。

把"脏数据"变成回测可信的输入。常见的脏：缺失、零/负价、停牌日、
重复日期、未排序。清洗动作要**可解释、不引入前视偏差**。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def clean_prices(df: pd.DataFrame, *, drop_suspended: bool = True) -> pd.DataFrame:
    """清洗 OHLCV 行情。

    Parameters
    ----------
    df : DataFrame
        以 DatetimeIndex 为索引，至少包含 open/high/low/close/volume。
    drop_suspended : bool
        是否丢弃疑似停牌日（成交量为 0 且价格不动）。

    Returns
    -------
    DataFrame
        清洗后的行情，索引唯一、升序，价格为正。
    """
    if df.empty:
        raise ValueError("传入的行情为空")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"行情缺少必要列: {missing}")

    out = df.copy()

    # 1) 索引必须是时间且唯一、升序
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()

    # 2) 价格必须为正；非正价视为无效，置 NaN
    price_cols = ["open", "high", "low", "close"]
    out[price_cols] = out[price_cols].where(out[price_cols] > 0, np.nan)

    # 3) 成交量缺失补 0；负量视为无效置 0
    out["volume"] = out["volume"].fillna(0).clip(lower=0)

    # 4) 停牌日（量=0 且收盘不变）
    if drop_suspended:
        suspended = (out["volume"] == 0) & (out["close"].diff().fillna(0) == 0)
        out = out[~suspended]

    # 5) 价格缺失：仅用前值填充（绝不用后值，否则前视偏差）
    out[price_cols] = out[price_cols].ffill()

    # 6) 丢弃仍含 NaN 的行（通常是序列开头）
    out = out.dropna(subset=price_cols)

    if out.empty:
        raise ValueError("清洗后行情为空，请检查输入数据质量")

    return out
