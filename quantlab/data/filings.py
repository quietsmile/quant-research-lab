"""财报原文（PDF）获取与关键段抽取——文本层"理解式处理"的前置。

设计核心（省 token）：**规则定位 → 大模型只读相关段**。
一份年报 ≈ 11 万字符 ≈ 112k tokens，全文喂大模型不可承受；而审计意见 /
关键审计事项 / 商誉减值等可用关键词在文本里精准定位，只把那几千字交给
大模型理解，token 降到约 1/40。

本模块只做**规则部分**（下载 + 抽文本 + 定位段落）；大模型抽取在
examples/extract_filings_pilot.py 里用子 agent 完成，便于单独计量成本。
"""
from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path

FILINGS_DIR = Path(os.environ.get(
    "QUANTLAB_FILINGS", Path.home() / ".local" / "share" / "quantlab" / "filings"
))
_UA = {"User-Agent": "Mozilla/5.0"}


def _disclosure_link(symbol: str, category: str, start: str, end: str, keyword: str):
    """用 akshare cninfo 接口找到目标公告的详情链接（含 orgId）。"""
    import akshare as ak
    df = ak.stock_zh_a_disclosure_report_cninfo(
        symbol=symbol, market="沪深京", category=category,
        start_date=start, end_date=end)
    if df is None or not len(df):
        return None
    hit = df[df["公告标题"].str.contains(keyword) & ~df["公告标题"].str.contains("摘要|英文")]
    if not len(hit):
        return None
    return hit.iloc[0].to_dict()


def _resolve_adjunct(stock_code: str, org_id: str, title_kw: str, start: str, end: str):
    """查 cninfo hisAnnouncement 拿 adjunctUrl（真实 PDF 相对路径）。"""
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    body = urllib.parse.urlencode({
        "stock": f"{stock_code},{org_id}", "tabName": "fulltext",
        "pageSize": "30", "pageNum": "1",
        "seDate": f"{start[:4]}-{start[4:6]}-{start[6:]}~{end[:4]}-{end[4:6]}-{end[6:]}",
        "column": "szse"}).encode()
    req = urllib.request.Request(url, data=body, headers={
        **_UA, "Content-Type": "application/x-www-form-urlencoded",
        "X-Requested-With": "XMLHttpRequest"})
    j = json.loads(urllib.request.urlopen(req, timeout=20).read())
    for a in (j.get("announcements") or []):
        t = a.get("announcementTitle", "")
        if title_kw in t and "摘要" not in t and "英文" not in t:
            return a.get("adjunctUrl")
    return None


def fetch_annual_report_pdf(symbol: str, year: int) -> Path:
    """下载某公司某年度的年报 PDF，返回本地路径。"""
    start, end = f"{year + 1}0101", f"{year + 1}0630"
    kw = f"{year}年年度报告"
    link = _disclosure_link(symbol, "年报", start, end, kw)
    if not link:
        raise ValueError(f"未找到 {symbol} {year} 年报公告")
    url = link["公告链接"]
    org = re.search(r"orgId=([^&]+)", url)
    code = re.search(r"stockCode=(\d+)", url)
    org_id = org.group(1) if org else ""
    stock_code = code.group(1) if code else symbol
    adjunct = _resolve_adjunct(stock_code, org_id, kw, start, end)
    if not adjunct:
        raise ValueError(f"未解析到 {symbol} {year} 年报 PDF 地址")
    pdf_url = f"http://static.cninfo.com.cn/{adjunct}"
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)
    out = FILINGS_DIR / f"{symbol}_{year}_annual.pdf"
    req = urllib.request.Request(pdf_url, headers=_UA)
    out.write_bytes(urllib.request.urlopen(req, timeout=60).read())
    return out


def pdf_text(path: str | Path) -> str:
    """抽取 PDF 全文文本。"""
    import fitz
    doc = fitz.open(str(path))
    return "".join(p.get_text() for p in doc)


# 关键段落锚点：用于规则定位，只把相关段交给大模型
SECTION_ANCHORS = {
    "audit": ["审计意见", "审计报告"],
    "goodwill": ["商誉", "商誉减值"],
    "non_recurring": ["非经常性损益"],
}


def locate_section(text: str, anchors: list[str], window: int = 3500) -> str:
    """按锚点定位关键段，返回锚点附近 window 字符（找不到返回空串）。"""
    for a in anchors:
        i = text.find(a)
        if i >= 0:
            return text[i: i + window]
    return ""


def extract_key_sections(text: str) -> dict:
    """规则抽取各关键段，供大模型理解（大幅压缩输入）。"""
    return {k: locate_section(text, anchors) for k, anchors in SECTION_ANCHORS.items()}
