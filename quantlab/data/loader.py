"""行情加载器。

加载优先级：本地缓存 → AKShare（若安装且联网）→ 可复现合成行情。

合成行情不是"假装有数据"，而是一个**几何布朗运动 + 缓动趋势**的
可复现序列，专门用来在没有数据源时验证策略与回测代码的正确性。
用同一个 symbol+区间永远得到同一段行情（种子由 symbol 决定）。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

from quantlab.data.clean import clean_prices

_CACHE_DIR = Path(os.environ.get("QUANTLAB_CACHE", Path.home() / ".cache" / "quantlab"))


def _cache_path(symbol: str, start: str, end: str, source: str) -> Path:
    key = f"{source}_{symbol}_{start}_{end}".replace(":", "")
    return _CACHE_DIR / f"{key}.csv"


def clear_cache() -> int:
    """清空本地行情缓存，返回删除的文件数。"""
    if not _CACHE_DIR.exists():
        return 0
    n = 0
    for f in _CACHE_DIR.glob("*.csv"):
        f.unlink()
        n += 1
    return n


def _seed_from_symbol(symbol: str) -> int:
    """由 symbol 生成稳定种子，保证同标的的合成行情可复现。"""
    h = hashlib.md5(symbol.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _synthetic_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """生成可复现的合成 A 股日线行情（仅交易日，近似周一~周五）。"""
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        raise ValueError(f"区间内无交易日: {start} ~ {end}")

    rng = np.random.default_rng(_seed_from_symbol(symbol))
    n = len(dates)

    # 年化漂移 ~8%，年化波动 ~25%，叠加一个缓慢正弦趋势制造"可被均线捕捉"的结构
    mu, sigma = 0.08 / 252, 0.25 / np.sqrt(252)
    trend = 0.0006 * np.sin(np.linspace(0, 6 * np.pi, n))
    rets = rng.normal(mu, sigma, n) + trend

    close = 30.0 * np.exp(np.cumsum(rets))
    # 由收盘价反推一个合理的 OHLC
    daily_range = np.abs(rng.normal(0, sigma, n)) * close
    open_ = close * (1 + rng.normal(0, sigma / 2, n))
    high = np.maximum(open_, close) + daily_range / 2
    low = np.minimum(open_, close) - daily_range / 2
    low = np.clip(low, 0.01, None)
    volume = rng.integers(1_000_000, 20_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    df.index.name = "date"
    return df


def _akshare_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """通过 AKShare 拉取 A 股前复权日线（需 pip install akshare 且联网）。"""
    import akshare as ak  # 局部导入：未安装时不影响核心功能

    start_ak = start.replace("-", "")
    end_ak = end.replace("-", "")
    raw = ak.stock_zh_a_hist(
        symbol=symbol, period="daily", start_date=start_ak, end_date=end_ak, adjust="qfq"
    )
    if raw is None or raw.empty:
        raise ValueError(f"AKShare 未返回数据: {symbol}")

    rename = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    }
    df = raw.rename(columns=rename)[["date", "open", "high", "low", "close", "volume"]]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").astype(float)
    return df


def load_prices(
    symbol: str,
    start: str = "2018-01-01",
    end: str = "2023-12-31",
    *,
    source: str = "auto",
    use_cache: bool = True,
    clean: bool = True,
) -> pd.DataFrame:
    """加载日线行情（OHLCV）。

    Parameters
    ----------
    symbol : str
        标的代码，如 "600519"。
    start, end : str
        起止日期 "YYYY-MM-DD"。
    source : {"auto", "akshare", "synthetic"}
        "auto"：优先 AKShare，失败回落合成行情。
    use_cache : bool
        是否读写本地 CSV 缓存。
    clean : bool
        是否运行 clean_prices。

    Returns
    -------
    DataFrame
        DatetimeIndex 索引的 OHLCV，已清洗（除非 clean=False）。
    """
    if source not in {"auto", "akshare", "synthetic"}:
        raise ValueError(f"未知 source: {source}")

    cache_file = _cache_path(symbol, start, end, source)
    if use_cache and cache_file.exists():
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        return clean_prices(df) if clean else df

    df: pd.DataFrame
    if source == "synthetic":
        df = _synthetic_prices(symbol, start, end)
    elif source == "akshare":
        df = _akshare_prices(symbol, start, end)
    else:  # auto
        try:
            df = _akshare_prices(symbol, start, end)
        except Exception:
            # 未装 akshare / 无网络 / 接口变动 —— 一律回落合成行情，保证可学习
            df = _synthetic_prices(symbol, start, end)

    if clean:
        df = clean_prices(df)

    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_file)

    return df
