"""离线数据仓：把真实行情持久化为 parquet，支持增量累积、断网可用。

与 loader 的 CSV 缓存的区别：
- 缓存（~/.cache/quantlab）：按 (symbol,区间,source) 分文件，是"这次请求"的快照。
- 离线仓（本模块）：按 symbol 一个 parquet，**跨多次下载累积**成一段尽量长的
  历史，是项目长期的"高质量数据总量"底座（回应数据基建目标）。

目录由环境变量 QUANTLAB_OFFLINE 指定，默认 ~/.local/share/quantlab/offline。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

OFFLINE_DIR = Path(os.environ.get(
    "QUANTLAB_OFFLINE", Path.home() / ".local" / "share" / "quantlab" / "offline"
))

_COLUMNS = ["open", "high", "low", "close", "volume"]


def _path(symbol: str) -> Path:
    return OFFLINE_DIR / f"{symbol}.parquet"


def has_offline(symbol: str) -> bool:
    return _path(symbol).exists()


def save_offline(symbol: str, df: pd.DataFrame) -> Path:
    """覆盖写入某标的的离线 parquet（df 需为 OHLCV、DatetimeIndex）。"""
    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    out = df[_COLUMNS].copy()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out.to_parquet(_path(symbol))
    return _path(symbol)


def load_offline(symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """从离线仓读取某标的，可选按日期切片。缺失则抛 FileNotFoundError。"""
    p = _path(symbol)
    if not p.exists():
        raise FileNotFoundError(f"离线仓无此标的: {symbol}（先运行 sync_offline 下载）")
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    return df


def update_offline(symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """把新抓到的行情**增量合并**进离线仓（按日期去重、保留最新、升序）。返回合并后全量。"""
    new_df = new_df[_COLUMNS].copy()
    new_df.index = pd.to_datetime(new_df.index)
    if has_offline(symbol):
        old = load_offline(symbol)
        merged = pd.concat([old, new_df])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    else:
        merged = new_df.sort_index()
    save_offline(symbol, merged)
    return merged


def list_offline() -> pd.DataFrame:
    """列出离线仓里所有标的及其覆盖区间/天数。"""
    if not OFFLINE_DIR.exists():
        return pd.DataFrame(columns=["symbol", "n_days", "start", "end"])
    rows = []
    for p in sorted(OFFLINE_DIR.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
            rows.append({"symbol": p.stem, "n_days": len(df),
                         "start": df.index.min().date(), "end": df.index.max().date()})
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)
