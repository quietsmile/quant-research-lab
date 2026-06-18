"""把持仓池里尚未拿到审计意见的公司，多轮重试抓到收敛。

cninfo PDF 抓取有瞬时失败率，多轮重试可补齐大部分；个别因年报被拆成多文件/
双重上市等边界情况仍可能失败，会如实列出。

跑法：python examples/converge_universe_audit.py [输出目录] [最大轮数]
"""
from __future__ import annotations

import json
import os
import socket
import sys

socket.setdefaulttimeout(40)

from quantlab.data.filings import fetch_annual_report_pdf, pdf_text, locate_audit, locate_section, SECTION_ANCHORS
from quantlab.data.universe import LIQUID_LEADERS

CONFIRM = ("我们认为", "我们审计了")


def _try_one(code: str, name: str, outdir: str) -> int | None:
    for year in (2025, 2024, 2023):
        try:
            pdf = fetch_annual_report_pdf(code, year)
            txt = pdf_text(pdf)
            audit = locate_audit(txt, 3500)
            if not audit or not any(k in audit for k in CONFIRM):
                continue   # 空段/拆分文件 → 换一年再试
            gw = locate_section(txt, SECTION_ANCHORS["goodwill"], 1500)
            json.dump({"code": code, "name": name, "year": year, "audit": audit, "goodwill": gw},
                      open(f"{outdir}/{code}.json", "w"), ensure_ascii=False)
            return year
        except Exception:  # noqa: BLE001
            continue
    return None


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit_run"
    max_rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    os.makedirs(outdir, exist_ok=True)

    done = {f[:-5] for f in os.listdir(outdir) if f.endswith(".json")}
    todo = {c: n for c, n in LIQUID_LEADERS.items() if c not in done}
    print(f"已完成 {len(done)} 家，待补 {len(todo)} 家", flush=True)

    for rnd in range(1, max_rounds + 1):
        if not todo:
            break
        print(f"\n--- 第 {rnd} 轮，剩 {len(todo)} 家 ---", flush=True)
        got = []
        for code, name in list(todo.items()):
            y = _try_one(code, name, outdir)
            if y:
                print(f"  ✅ {code} {name} ({y}年报)", flush=True)
                got.append(code)
            else:
                print(f"  ❌ {code} {name}", flush=True)
        for c in got:
            todo.pop(c, None)

    total = len({f[:-5] for f in os.listdir(outdir) if f.endswith('.json')})
    print(f"\n收敛结束：持仓池 {total}/{len(LIQUID_LEADERS)} 已拿到审计段。", flush=True)
    if todo:
        print("仍失败（边界情况，需人工/换源）：", list(todo.items()), flush=True)


if __name__ == "__main__":
    main()
