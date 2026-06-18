"""为持仓池抽取最近一年年报的"审计意见"段（规则层：下载+定位，存紧凑文件）。

下游由大模型读这些段做结构化抽取（见 README 文本层说明）。
跑法：python examples/fetch_universe_audit.py [输出目录]
"""
from __future__ import annotations

import json
import os
import socket
import sys

socket.setdefaulttimeout(40)

from quantlab.data.filings import fetch_annual_report_pdf, pdf_text, locate_audit, locate_section, SECTION_ANCHORS
from quantlab.data.universe import LIQUID_LEADERS


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit_run"
    os.makedirs(outdir, exist_ok=True)
    ok = 0
    for i, (code, name) in enumerate(LIQUID_LEADERS.items(), 1):
        year_used = None
        for year in (2025, 2024):                # 先试最近年报，缺则退一年
            try:
                pdf = fetch_annual_report_pdf(code, year)
                txt = pdf_text(pdf)
                audit = locate_audit(txt, window=3500)
                goodwill = locate_section(txt, SECTION_ANCHORS["goodwill"], window=1500)
                if not audit:
                    continue
                json.dump({"code": code, "name": name, "year": year,
                           "audit": audit, "goodwill": goodwill},
                          open(f"{outdir}/{code}.json", "w"), ensure_ascii=False)
                year_used = year
                ok += 1
                break
            except Exception as e:  # noqa: BLE001
                last = f"{type(e).__name__}: {str(e)[:40]}"
        flag = f"✅ {year_used}年报" if year_used else f"❌ {last}"
        print(f"[{i:>2}/{len(LIQUID_LEADERS)}] {code} {name:<6} {flag}", flush=True)
    print(f"\n完成：{ok}/{len(LIQUID_LEADERS)} 份审计段已存到 {outdir}/")


if __name__ == "__main__":
    main()
