"""Quant Research Lab 看板(多页)：策略族+Barra暴露 / 前瞻事件策略·Test操作。

跑法：streamlit run dashboard/main_app.py
"""
from __future__ import annotations
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from quantlab.data.tushare_adapter import _FUND_DIR
import pathlib
DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")

st.set_page_config(page_title="Quant Research Lab", layout="wide")
page = st.sidebar.radio("页面", ["📊 策略族 & Barra 暴露", "📈 前瞻事件策略 · Test 操作"])


def page_strategy_family():
    st.title("📊 策略族 + 参数空间 + Barra 风格暴露")
    st.caption("主题：热门板块+趋势+主力资金+财报质量。每个策略有可调参数空间，网格搜索后按"
               "**最差子区间夏普**挑稳健最优，强制分散(持股≥15)；并对最优配置做 Barra 多因子暴露。"
               "回测 2020–2026、含成本。**判据是稳健**，不是最高年化。")
    try:
        d = json.load(open(DD / "strategy_family.json"))
    except Exception as e:  # noqa: BLE001
        st.error(f"结果未就绪，请先运行 examples/strategy_family.py：{e}"); return
    S = d["strategies"]; B = d["benchmark"]

    rows = []
    for k, v in S.items():
        rows.append({"策略": k, "年化": v["cagr"], "夏普": v["sharpe"], "最大回撤": v["maxdd"],
                     "Calmar": v["calmar"], "夏普20-22": v["sh1"], "夏普23-26": v["sh2"],
                     "最差年%": v["worst"], "持股": round(v["nh"]),
                     "α年化": v["barra"].get("alpha_ann"), "α_t": v["barra"].get("alpha_t"),
                     "网格夏普中位": v["grid_sharpe_median"], "两子区间均正%": v["grid_robust_pct"]})
    df = pd.DataFrame(rows)
    st.subheader("策略对比（最优配置）")
    fmt = df.copy()
    for c in ["年化", "最大回撤", "α年化"]: fmt[c] = (fmt[c] * 100).round(1)
    fmt["两子区间均正%"] = (fmt["两子区间均正%"] * 100).round(0)
    for c in ["夏普", "Calmar", "夏普20-22", "夏普23-26", "α_t", "网格夏普中位"]: fmt[c] = fmt[c].round(2)
    st.dataframe(fmt, use_container_width=True)
    st.caption(f"等权基准：年化 {B['cagr']*100:+.0f}% | 夏普 {B['sharpe']:.2f} | 最大回撤 {B['maxdd']*100:+.0f}%（最难超越的基线）")

    c1, c2 = st.columns(2)
    # 风险调整 vs 等权
    figc = go.Figure()
    figc.add_trace(go.Bar(x=df["策略"], y=df["夏普"], name="夏普"))
    figc.add_hline(y=B["sharpe"], line_dash="dash", line_color="red", annotation_text=f"等权 {B['sharpe']:.2f}")
    figc.update_layout(title="夏普 vs 等权基准", height=340, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(figc, use_container_width=True)
    # alpha
    figa = px.bar(df, x="策略", y="α年化", title="Barra 剔除风格后的年化 α（>0 且 t 显著才算真 alpha）")
    figa.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".0%")
    figa.add_hline(y=0, line_color="gray")
    c2.plotly_chart(figa, use_container_width=True)

    # Barra 暴露热图
    st.subheader("Barra 风格暴露（各策略对各风格因子的 beta）")
    exp_rows = {k: v["barra"].get("exposure", {}) for k, v in S.items()}
    exp_df = pd.DataFrame(exp_rows).T
    if not exp_df.empty:
        fig = px.imshow(exp_df, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                        labels=dict(color="beta"), title="暴露热图（红=正暴露/蓝=负）")
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("SIZE(小)正=押小盘、VALUE(高EP)正=偏便宜、MOM正=追动量、VOL(低波)正=偏低波、"
                   "GROWTH正=偏高增长。多数策略 SIZE 正暴露——超额很大程度是小盘 beta（2026 已逆风）。")

    with st.expander("各策略最优参数 + 逐年收益"):
        for k, v in S.items():
            st.markdown(f"**{k}** — 最优参数 `{v['best_param']}` | 逐年 {v['by']}")


def page_event():
    st.title("📈 前瞻事件策略 · 样本外(2024–2025) 操作")
    st.caption("正向业绩预告 & 预告增长后 forward PE 低于行业中位 → 预告次日买入，持有到正式财报前。"
               "仓位：Top-20 等权、单只≤5%。收益为扣成本净值。")
    try:
        trades = pd.read_parquet(_FUND_DIR / "viz_trades.parquet")
        eq = pd.read_parquet(_FUND_DIR / "viz_equity.parquet")
    except Exception as e:  # noqa: BLE001
        st.error(f"数据未就绪：{e}"); return
    ts, tm = eq["strategy"].iloc[-1] - 1, eq["market"].iloc[-1] - 1
    c = st.columns(4)
    c[0].metric("Test 累计净收益", f"{ts:+.1%}", f"{ts - tm:+.1%} vs 大盘")
    c[1].metric("等权大盘", f"{tm:+.1%}")
    c[2].metric("交易笔数", f"{len(trades)}")
    c[3].metric("逐笔胜率", f"{(trades['excess'] > 0).mean():.0%}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["strategy"], name="策略(净值)", line=dict(color="crimson", width=2)))
    fig.add_trace(go.Scatter(x=eq["date"], y=eq["market"], name="等权大盘", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="净值（扣成本，归一）", height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    trades["盈亏"] = trades["excess"].apply(lambda x: "盈" if x > 0 else "亏")
    figs = px.scatter(trades, x="entry", y="excess", color="盈亏",
                      color_discrete_map={"盈": "#d62728", "亏": "#2ca02c"},
                      hover_data=["name", "symbol", "exit"], title="每一笔操作（买入时间 × 相对大盘超额）")
    figs.add_hline(y=0, line_dash="dot"); figs.update_layout(height=360, yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(figs, use_container_width=True)


if page.startswith("📊"):
    page_strategy_family()
else:
    page_event()
