"""Tushare 适配器——结构化财务数据的主源（PIT 自带、含扣非、全口径）。

为什么用它做主源：
- 利润表/资产负债表/现金流量表自带 **f_ann_date（实际首发公告日）**，PIT 开箱即用，
  无需再爬 cninfo；`update_flag` 可区分原始 vs 重述，支持真 vintage。
- `fina_indicator` 提供 **扣非净利润 profit_dedt**、扣非 ROE、毛利率等几十个比率。
- 覆盖北交所、退市股、指数成分、复权因子。

token 来源：环境变量 TUSHARE_TOKEN，或文件 ~/.tushare_token。
注意：Tushare 为积分制且有每分钟频率限制，逐只拉取请带 sleep。
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TS_FUND_FILE = Path(os.environ.get(
    "QUANTLAB_FUND", Path.home() / ".local" / "share" / "quantlab" / "fundamentals"
)) / "tushare_pit.parquet"

_TOKEN_FILE = Path.home() / ".tushare_token"
_pro_cached = None


def _token() -> str:
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text().strip()
    raise RuntimeError("未配置 Tushare token（设 TUSHARE_TOKEN 或写入 ~/.tushare_token）")


def get_pro():
    """惰性初始化 tushare pro 客户端（带 token）。"""
    global _pro_cached
    if _pro_cached is None:
        import tushare as ts
        ts.set_token(_token())
        _pro_cached = ts.pro_api()
    return _pro_cached


def to_ts_code(symbol: str) -> str:
    """6 位代码 → Tushare ts_code（带交易所后缀）。"""
    s = str(symbol).zfill(6)
    if "." in str(symbol):
        return str(symbol).upper()
    if s[0] in ("6", "9"):
        return f"{s}.SH"
    if s.startswith("920") or s[0] in ("4", "8"):
        return f"{s}.BJ"
    return f"{s}.SZ"          # 0/2/3 开头

# 报告期末 → 标准化为 datetime
def _to_period(end_date: pd.Series) -> pd.Series:
    return pd.to_datetime(end_date, format="%Y%m%d", errors="coerce")


# n_income_attr_p = 归母净利润（做因子通常用归母，而非含少数股东的 n_income）
_INCOME_FIELDS = "ts_code,f_ann_date,end_date,total_revenue,n_income,n_income_attr_p,update_flag"
_INDI_FIELDS = "ts_code,end_date,roe,roe_dt,profit_dedt,grossprofit_margin,eps"


def fundamentals_pit(symbols, start_date="20150101", end_date="20251231",
                     *, sleep: float = 0.35, verbose: bool = True) -> pd.DataFrame:
    """逐只拉取利润表 + 财务指标，合并成 PIT 财务表（announce_date = f_ann_date）。

    返回列：symbol, report_period, announce_date, revenue, net_profit, profit_dedt,
            roe, roe_dedt, gross_margin, eps, update_flag
    """
    import time
    pro = get_pro()
    frames = []
    syms = [str(s).zfill(6) for s in symbols]
    for i, sym in enumerate(syms, 1):
        ts_code = to_ts_code(sym)
        try:
            inc = pro.income(ts_code=ts_code, start_date=start_date, end_date=end_date,
                             fields=_INCOME_FIELDS)
            time.sleep(sleep)
            ind = pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date,
                                     fields=_INDI_FIELDS)
            # 同一报告期可能有原始+重述多行，保留最新（update_flag 大者）
            inc = inc.sort_values("update_flag").drop_duplicates("end_date", keep="last")
            ind = ind.drop_duplicates("end_date", keep="last")
            m = inc.merge(ind, on=["ts_code", "end_date"], how="left")
            m["symbol"] = sym
            frames.append(m)
            if verbose:
                print(f"[{i}/{len(syms)}] {ts_code}: {len(m)} 期 ✅", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[{i}/{len(syms)}] {ts_code}: ❌ {type(e).__name__}: {str(e)[:50]}", flush=True)
        if i < len(syms):
            time.sleep(sleep)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({
        "symbol": df["symbol"],
        "report_period": _to_period(df["end_date"]),
        "announce_date": pd.to_datetime(df["f_ann_date"], format="%Y%m%d", errors="coerce"),
        "revenue": df["total_revenue"],
        # 归母净利润优先（缺则回落含少数股东的 n_income）
        "net_profit": df["n_income_attr_p"].fillna(df["n_income"]) if "n_income_attr_p" in df else df["n_income"],
        "net_profit_total": df.get("n_income"),     # 含少数股东
        "profit_dedt": df.get("profit_dedt"),       # 扣非归母净利润
        "roe": df.get("roe"),
        "roe_dedt": df.get("roe_dt"),               # 扣非 ROE
        "gross_margin": df.get("grossprofit_margin"),
        "eps": df.get("eps"),
        "update_flag": df.get("update_flag"),
    })
    out = out.dropna(subset=["report_period", "announce_date"])
    out = out.sort_values(["symbol", "report_period"]).reset_index(drop=True)
    return out


def save_pit(df: pd.DataFrame) -> Path:
    TS_FUND_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(TS_FUND_FILE, index=False)
    return TS_FUND_FILE


def load_pit() -> pd.DataFrame:
    if not TS_FUND_FILE.exists():
        raise FileNotFoundError(f"Tushare PIT 库不存在：{TS_FUND_FILE}")
    df = pd.read_parquet(TS_FUND_FILE)
    for c in ("report_period", "announce_date"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def point_in_time(as_of, symbols=None) -> pd.DataFrame:
    """PIT 查询（基于 Tushare 真实 f_ann_date）：截至 as_of 已公告的每只最新财报。"""
    as_of = pd.Timestamp(as_of)
    df = load_pit()
    df = df[df["announce_date"].notna() & (df["announce_date"] <= as_of)]
    if symbols is not None:
        syms = [str(s).zfill(6) for s in symbols]
        df = df[df["symbol"].isin(syms)]
    if df.empty:
        return df
    df = df.sort_values(["symbol", "report_period"])
    return df.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
