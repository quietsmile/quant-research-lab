"""A 股财务数据（point-in-time / 按公告日组织）。

为什么按公告日：一季报报告期是 3/31，但往往 4 月底才公告。回测里若在 4 月
初就用到一季报 = 前视偏差，结果作废。本模块对每条财报都用**真实公告日**对齐，
查询时只返回"截至某日已公告"的数据。

数据源分工（均已实测可用）：
- **财务数字**：东财 `stock_yjbb_em(date=季度末)`，一次覆盖全市场，但它自带的
  「最新公告日期」是该公司最近一次披露日、**不是这期报告的首发日**，故不用于 PIT。
- **公告日**：巨潮 cninfo `stock_zh_a_disclosure_report_cninfo`，给每期定期报告的
  **权威首发公告时间**（按 symbol 抓取）。
- **兜底**：cninfo 没覆盖到时，用**法定披露截止日**（年报/Q1≤次/当年4-30、
  半年报≤8-31、Q3≤10-31）——保守，绝不会比真实公告日更早，因此不前视。

存储：parquet，目录由环境变量 QUANTLAB_FUND 指定，默认
~/.local/share/quantlab/fundamentals。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

FUND_DIR = Path(os.environ.get(
    "QUANTLAB_FUND", Path.home() / ".local" / "share" / "quantlab" / "fundamentals"
))
PERFORMANCE_FILE = FUND_DIR / "performance.parquet"   # 财务数字（yjbb）
ANNOUNCE_FILE = FUND_DIR / "announce_dates.parquet"   # 真实公告日（cninfo）

# 东财业绩报表中文列 → 规范英文列（最新公告日期改名为 latest_disclosure，不用于 PIT）
_COLMAP = {
    "股票代码": "symbol", "股票简称": "name", "每股收益": "eps",
    "营业总收入-营业总收入": "revenue", "营业总收入-同比增长": "revenue_yoy",
    "净利润-净利润": "net_profit", "净利润-同比增长": "net_profit_yoy",
    "每股净资产": "bps", "净资产收益率": "roe", "每股经营现金流量": "ocfps",
    "销售毛利率": "gross_margin", "所属行业": "industry",
    "最新公告日期": "latest_disclosure",
}

# cninfo 定期报告类别 → 报告期月日
_CNINFO_CATEGORIES = ["年报", "半年报", "一季报", "三季报"]


# A 股代码前缀（沪深主板/科创/创业 + 北交所），用于从 yjbb 全集中剔除新三板等
_A_PREFIX2 = {"60", "68", "00", "30", "83", "87", "88", "92"}


def is_a_share(symbol: str) -> bool:
    """是否为沪深京 A 股代码（粗筛，排除新三板等）。"""
    s = str(symbol).zfill(6)
    return s[:2] in _A_PREFIX2 or s[:3] in {"920", "430"}


def a_share_symbols_from_store() -> list[str]:
    """从已下载的财务库取全部 A 股代码（供公告日批量抓取用）。"""
    perf = load_performance()
    return sorted(s for s in perf["symbol"].unique() if is_a_share(s))


def quarter_ends(start_year: int, end_year: int) -> list[str]:
    """生成 [start_year, end_year] 的季度末日期 YYYYMMDD。"""
    return [f"{y}{mmdd}" for y in range(start_year, end_year + 1)
            for mmdd in ("0331", "0630", "0930", "1231")]


def statutory_deadline(report_period: pd.Timestamp) -> pd.Timestamp:
    """该报告期的法定披露截止日（保守 PIT 兜底，绝不早于真实公告日）。"""
    p = pd.Timestamp(report_period)
    m = p.month
    if m == 3:    # 一季报：当年 4-30
        return pd.Timestamp(p.year, 4, 30)
    if m == 6:    # 半年报：当年 8-31
        return pd.Timestamp(p.year, 8, 31)
    if m == 9:    # 三季报：当年 10-31
        return pd.Timestamp(p.year, 10, 31)
    return pd.Timestamp(p.year + 1, 4, 30)   # 年报：次年 4-30


# ---------------- 财务数字（yjbb） ----------------

def _normalize_perf(df: pd.DataFrame, period: str) -> pd.DataFrame:
    cols = {k: v for k, v in _COLMAP.items() if k in df.columns}
    out = df.rename(columns=cols)[list(cols.values())].copy()
    out["report_period"] = pd.to_datetime(period, format="%Y%m%d")
    if "latest_disclosure" in out.columns:
        out["latest_disclosure"] = pd.to_datetime(out["latest_disclosure"], errors="coerce")
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    return out


def download_performance(periods, *, save=True, sleep=0.5, verbose=True) -> pd.DataFrame:
    """按季度下载全市场业绩报表（财务数字），合并落 parquet。"""
    import time
    import akshare as ak

    frames = []
    for i, p in enumerate(periods, 1):
        try:
            raw = ak.stock_yjbb_em(date=p)
            if raw is not None and len(raw):
                frames.append(_normalize_perf(raw, p))
                if verbose:
                    print(f"[数字 {i}/{len(periods)}] {p}: {len(raw)} 行 ✅", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[数字 {i}/{len(periods)}] {p}: ❌ {type(e).__name__}: {str(e)[:50]}", flush=True)
        if i < len(periods):
            time.sleep(sleep)

    if not frames:
        return pd.DataFrame()
    new = pd.concat(frames, ignore_index=True)
    if save:
        new = _merge(new, PERFORMANCE_FILE, ["symbol", "report_period"])
    return new


# ---------------- 公告日（cninfo） ----------------

def _parse_period(title: str) -> pd.Timestamp | None:
    """从公告标题解析报告期末日。"""
    m = re.search(r"(\d{4})\s*年", title)
    if not m:
        return None
    y = int(m.group(1))
    # 注意顺序：「半年度报告」含「年度报告」子串，必须先判半年报
    if "半年度报告" in title or "中期报告" in title:
        return pd.Timestamp(y, 6, 30)
    if "第一季度" in title or "一季度报告" in title:
        return pd.Timestamp(y, 3, 31)
    if "第三季度" in title or "三季度报告" in title:
        return pd.Timestamp(y, 9, 30)
    if "年度报告" in title:
        return pd.Timestamp(y, 12, 31)
    return None


def download_announce_dates(symbols, start_date="20100101", end_date="20251231",
                            *, save=True, sleep=0.3, verbose=True) -> pd.DataFrame:
    """逐只从 cninfo 抓定期报告的真实首发公告日，落 parquet。

    对每只股票、每类定期报告，取每个报告期的**最早**公告时间作为 PIT 日期。
    """
    import time
    import akshare as ak

    rows = []
    syms = [str(s).zfill(6) for s in symbols]
    for i, sym in enumerate(syms, 1):
        got = 0
        for cat in _CNINFO_CATEGORIES:
            try:
                df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=sym, market="沪深京", category=cat,
                    start_date=start_date, end_date=end_date)
            except Exception:  # noqa: BLE001
                continue
            if df is None or not len(df):
                continue
            for _, r in df.iterrows():
                period = _parse_period(str(r.get("公告标题", "")))
                if period is None:
                    continue
                rows.append({"symbol": sym, "report_period": period,
                             "announce_date": pd.to_datetime(r["公告时间"], errors="coerce")})
                got += 1
        if verbose:
            print(f"[公告 {i}/{len(syms)}] {sym}: {got} 条", flush=True)
        if i < len(syms):
            time.sleep(sleep)

    if not rows:
        return pd.DataFrame()
    new = pd.DataFrame(rows).dropna(subset=["announce_date"])
    # 同一 (symbol, period) 取最早公告时间
    new = new.sort_values("announce_date").drop_duplicates(
        subset=["symbol", "report_period"], keep="first")
    if save:
        new = _merge(new, ANNOUNCE_FILE, ["symbol", "report_period"], keep="first",
                     sort_col="announce_date")
    return new


# ---------------- 通用合并 / 读取 ----------------

def _merge(new, path, keys, *, keep="last", sort_col=None):
    if path.exists():
        old = pd.read_parquet(path)
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    merged = merged.sort_values(sort_col if sort_col else keys)
    merged = merged.drop_duplicates(subset=keys, keep=keep).reset_index(drop=True)
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def load_performance() -> pd.DataFrame:
    if not PERFORMANCE_FILE.exists():
        raise FileNotFoundError(f"财务数字库不存在：{PERFORMANCE_FILE}")
    df = pd.read_parquet(PERFORMANCE_FILE)
    for c in ("report_period", "latest_disclosure"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_announce_dates() -> pd.DataFrame:
    if not ANNOUNCE_FILE.exists():
        return pd.DataFrame(columns=["symbol", "report_period", "announce_date"])
    df = pd.read_parquet(ANNOUNCE_FILE)
    for c in ("report_period", "announce_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def build_pit_table() -> pd.DataFrame:
    """合成 PIT 表：财务数字 + 公告日（真实优先、缺失用法定截止日兜底）。

    新增列：
    - announce_date    : 用于 PIT 的可用日期（真实公告日 或 法定截止日）
    - announce_is_real : True=cninfo 真实公告日；False=法定截止日兜底
    """
    perf = load_performance()
    ann = load_announce_dates()
    if not ann.empty:
        perf = perf.merge(ann, on=["symbol", "report_period"], how="left")
    else:
        perf["announce_date"] = pd.NaT

    real = perf["announce_date"].copy()
    fallback = perf["report_period"].map(statutory_deadline)
    perf["announce_is_real"] = real.notna()
    perf["announce_date"] = real.fillna(fallback)
    return perf


def point_in_time(as_of, symbols=None) -> pd.DataFrame:
    """Point-in-time 查询：返回截至 as_of **已公告**的、每只股票最新的一期财报。"""
    as_of = pd.Timestamp(as_of)
    df = build_pit_table()
    df = df[df["announce_date"].notna() & (df["announce_date"] <= as_of)]
    if symbols is not None:
        syms = [str(s).zfill(6) for s in symbols]
        df = df[df["symbol"].isin(syms)]
    if df.empty:
        return df
    df = df.sort_values(["symbol", "report_period"])
    return df.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
