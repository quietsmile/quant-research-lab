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

_FUND_DIR = Path(os.environ.get(
    "QUANTLAB_FUND", Path.home() / ".local" / "share" / "quantlab" / "fundamentals"
))
TS_FUND_FILE = _FUND_DIR / "tushare_pit.parquet"
TS_FEATURES_FILE = _FUND_DIR / "tushare_features.parquet"  # 含单季/TTM/同比
LISTING_FILE = _FUND_DIR / "listing.parquet"   # 上市日（list_date 门控用）

# Tushare PIT 表里的累计口径流量字段（需拆单季/TTM）
TS_FLOW_FIELDS = ["net_profit", "net_profit_total", "revenue", "profit_dedt", "eps"]

_TOKEN_FILE = Path.home() / ".tushare_token"
_pro_cached = None


def _token() -> str:
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    if _TOKEN_FILE.exists():
        return _TOKEN_FILE.read_text().strip()
    raise RuntimeError("未配置 Tushare token（设 TUSHARE_TOKEN 或写入 ~/.tushare_token）")


def _install_keepalive_session():
    """把 tushare 客户端的 requests 换成**单连接 keep-alive Session**。

    本环境是按连接分配出口 IP 的多出口 NAT；tushare 默认每次调用新建连接 →
    出口 IP 轮换 → 触发"每 token 最多 N 个 IP"。改用复用单条长连接的 Session 后，
    所有调用走同一出口 IP，只占用 1 个 IP 名额。
    """
    import requests
    from requests.adapters import HTTPAdapter
    import tushare.pro.client as cli

    sess = requests.Session()
    # 强制单连接池，最大化复用同一条 TCP 连接（即同一出口 IP）
    adapter = HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=2)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    cli.requests = sess          # query() 内部 requests.post → sess.post
    return sess


def get_pro():
    """惰性初始化 tushare pro 客户端（带 token + 单连接 Session 固定出口 IP）。"""
    global _pro_cached
    if _pro_cached is None:
        _install_keepalive_session()
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


def download_listing(*, save: bool = True) -> pd.DataFrame:
    """拉全 A 股名录（含**退市/暂停**），缓存 listing.parquet。

    关键：list_status 取 L(在市)+D(退市)+P(暂停)三类，**纳入已退市公司以避免
    幸存者偏差**；带 list_date(门控)与 delist_date(退市日)。
    """
    pro = get_pro()
    frames = []
    for st in ("L", "D", "P"):
        try:
            d = pro.stock_basic(exchange="", list_status=st,
                                fields="symbol,name,industry,market,list_date,delist_date")
            d["list_status"] = st
            frames.append(d)
        except Exception:  # noqa: BLE001
            continue
    df = pd.concat(frames, ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["list_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce")
    df["delist_date"] = pd.to_datetime(df.get("delist_date"), format="%Y%m%d", errors="coerce")
    df = df.drop_duplicates("symbol")
    if save:
        LISTING_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(LISTING_FILE, index=False)
    return df


def all_symbols(include_delisted: bool = True) -> list[str]:
    """全 A 股代码列表（默认含退市，避免幸存者偏差）。需先 download_listing。"""
    df = load_listing()
    if not include_delisted:
        df = df[df["list_status"] == "L"]
    return sorted(df["symbol"].tolist())


def load_listing() -> pd.DataFrame:
    if not LISTING_FILE.exists():
        raise FileNotFoundError(f"上市日库不存在：{LISTING_FILE}（先运行 download_listing）")
    df = pd.read_parquet(LISTING_FILE)
    df["list_date"] = pd.to_datetime(df["list_date"], errors="coerce")
    return df


MARKET_PANEL_FILE = _FUND_DIR / "market_panel.parquet"


def quarter_end_panel(start_year: int, end_year: int, *, save: bool = True,
                      sleep: float = 0.3, verbose: bool = True) -> pd.DataFrame:
    """按季度末交易日拉全市场截面（一次调用一天，含退市股，无幸存者偏差）。

    返回长表：trade_date, symbol, adj_close(后复权收盘), pe_ttm, pb, total_mv。
    用于多因子分层回测的价格/估值面板。
    """
    import time
    pro = get_pro()
    # 各季度末交易日
    cal = pro.trade_cal(exchange="SSE", start_date=f"{start_year}0101",
                        end_date=f"{end_year}1231", is_open="1")
    cal["d"] = pd.to_datetime(cal["cal_date"], format="%Y%m%d")
    q_last = cal.groupby(cal["d"].dt.to_period("Q"))["cal_date"].max().tolist()

    frames = []
    for i, d in enumerate(q_last, 1):
        try:
            db = pro.daily_basic(trade_date=d, fields="ts_code,close,pe_ttm,pb,total_mv")
            af = pro.adj_factor(trade_date=d, fields="ts_code,adj_factor")
            m = db.merge(af, on="ts_code", how="left")
            m["adj_close"] = m["close"] * m["adj_factor"].fillna(1.0)
            m["trade_date"] = pd.to_datetime(d, format="%Y%m%d")
            m["symbol"] = m["ts_code"].str[:6]
            frames.append(m[["trade_date", "symbol", "adj_close", "pe_ttm", "pb", "total_mv"]])
            if verbose:
                print(f"[{i}/{len(q_last)}] {d}: {len(m)} 只 ✅", flush=True)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"[{i}/{len(q_last)}] {d}: ❌ {str(e)[:40]}", flush=True)
        time.sleep(sleep)

    panel = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if save and not panel.empty:
        MARKET_PANEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(MARKET_PANEL_FILE, index=False)
    return panel


def load_market_panel() -> pd.DataFrame:
    if not MARKET_PANEL_FILE.exists():
        raise FileNotFoundError(f"市场面板不存在：{MARKET_PANEL_FILE}")
    df = pd.read_parquet(MARKET_PANEL_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


DAILY_FILE = _FUND_DIR / "daily_prices.parquet"


def load_daily_prices() -> pd.DataFrame:
    """全市场**日频**后复权收盘价长表（trade_date, symbol, adj_close）。

    由 examples/pull_daily_prices.py 构建；支持日频/任意频率回测。
    """
    if not DAILY_FILE.exists():
        raise FileNotFoundError(f"日频价格库不存在：{DAILY_FILE}（先运行 examples/pull_daily_prices.py）")
    df = pd.read_parquet(DAILY_FILE)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return df


def build_features(df: pd.DataFrame | None = None, *, save: bool = True) -> pd.DataFrame:
    """对 Tushare PIT 表做累计→单季 + TTM + 单季同比（口径一致、跨期可比）。

    解决"截面混报告期、累计口径不可比"：单季/TTM 字段与报告期长度无关。
    新增列：<f>_q / <f>_ttm / <f>_q_yoy（f ∈ TS_FLOW_FIELDS）。
    """
    from quantlab.data import fundamentals_features as ff
    base = df if df is not None else load_pit()
    feat = ff.to_single_quarter(base, flow_fields=TS_FLOW_FIELDS)
    feat = ff.add_ttm(feat, flow_fields=TS_FLOW_FIELDS)
    feat = ff.add_single_quarter_yoy(feat, flow_fields=TS_FLOW_FIELDS)
    if save:
        TS_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
        feat.to_parquet(TS_FEATURES_FILE, index=False)
    return feat


def point_in_time(as_of, symbols=None) -> pd.DataFrame:
    """PIT 查询（基于 Tushare 真实 f_ann_date）：截至 as_of 已公告的每只最新财报。

    若已构建特征表（tushare_features.parquet）则优先用它（含单季/TTM/同比）。
    """
    as_of = pd.Timestamp(as_of)
    if TS_FEATURES_FILE.exists():
        df = pd.read_parquet(TS_FEATURES_FILE)
        for c in ("report_period", "announce_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    else:
        df = load_pit()
    df = df[df["announce_date"].notna() & (df["announce_date"] <= as_of)]
    if symbols is not None:
        syms = [str(s).zfill(6) for s in symbols]
        df = df[df["symbol"].isin(syms)]
    if df.empty:
        return df
    df = df.sort_values(["symbol", "report_period"])
    return df.groupby("symbol", as_index=False).tail(1).reset_index(drop=True)
