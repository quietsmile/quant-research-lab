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
page = st.sidebar.radio("页面", ["🛡️ 大盘稳健族(低小盘)", "📊 策略族 & Barra 暴露",
                                 "🎛️ 策略调参", "📈 前瞻事件策略 · Test 操作"])


@st.cache_resource(show_spinner="加载策略引擎…")
def _engine():
    from examples import strategy_family as sf
    return sf


@st.cache_data(show_spinner="回测中…")
def _eval(name, items):
    sf = _engine()
    m = sf.eval_config(name, dict(items))
    return {k: v for k, v in m.items() if k not in ("port",)}  # nav 保留(画图用)


@st.cache_data(show_spinner="扫描参数网格…")
def _grid(name):
    return _engine().grid_sharpes(name)


COMMON_DOC = """**所有策略的共同设定**
- **股票池**：liq1500（按流动性筛的约 1500 只，含部分退市），回测 **2020 起**（主力资金数据起点）。
- **趋势过滤 trend**：收盘 ≥ 20 日均线 且 20 日均线上行（`close≥MA20 & MA20>MA20.shift(5)`）。
- **财报质量过滤 q**：扣非 ROE>0 **且** 单季归母净利同比>0 **且** 市值>5 亿（剔垃圾小盘妖股）。
- **建仓**：每 `rebal` 个交易日，在满足条件的股票里按打分取 **Top-N 等权**，持有到下次调仓。
- **成本**：双边 0.15% 换手成本；**强制分散**：持股 ≥15（<15 的配置不纳入，规避集中度运气）。
- **选参判据**：按"最差子区间夏普"挑稳健最优，而非最高年化。
"""

STRATEGY_DOCS = {
 "等权基准": """**等权全市场（基准，最难超越的对照）**
- **构造**：liq1500 池每日截面**等权平均收益**（= 每日再平衡到等权）。
- **为何强**：① 等权 = **小盘最大暴露**——2024下半年–2026 A股小/微盘普涨，等权吃满（池内最小市值300只 Test CAGR+51%/夏普1.47；沪深300只有+20%/1.06）；② **每日再平衡红利**(波动市里"低买高卖")；③ **未扣成本、股票池仅2.3%退市(轻幸存者偏差)**——基准本身偏理想化。
- **Test(2024-07~2026)**：CAGR **+40%**、夏普 **1.42**。**含义**：这条线很大程度是"小盘beta+理想化"，主动策略降了小盘暴露所以反而落后；更公平的可投基准是沪深300(夏普1.06)。
""",
 "S1 质量+动量": """**选股**：趋势 & 质量。**打分**：过去 `mom_win` 日涨幅（追动量，高分=近期强势）。
**参数**：topn(持股数)、rebal(调仓周期)、mom_win(动量窗口 20/40/60)。
**Barra 结论**：R² **64%**、α 年化 **−9.9%(不显著)** → 超额几乎全是**动量/高波风格 beta**，没有独立 alpha。A股纯动量本就弱。""",
 "S2 质量+价值EP": """**选股**：趋势 & 质量。**打分**：EP = 盈利收益率(E/P，越高越便宜)，选低估值。
**参数**：topn、rebal。
**Barra 结论**：VALUE 暴露 **+0.86**(强价值)、α 年化 **+12.1%**、R² 19%。**冻结 Test 夏普 0.83、DSR 71%** —— 三个候选里**衰减最小、最稳**，但仍未达标(DSR<90%、跑输等权)。""",
 "S3 质量+低波": """**选股**：趋势 & 质量。**打分**：−过去 `vol_win` 日日收益标准差(选**低波动**)。
**参数**：topn、rebal、vol_win(20/40)。
**Barra 结论**：R² 仅 **6%**(风格几乎解释不了它)、全样本 α **+15.2%(t2.2)** → 最像独立 alpha。**但冻结 Test 夏普从 dev 1.30 暴跌到 0.50、DSR 53%** —— 样本外衰减最严重，未确认。""",
 "S4 板块轮动+质量+复合": """**选股**：(板块20日动量 Top-`hot_k` **∪** 板块主力资金流20日 Top-`hot_k`) & 趋势 & 质量 —— 只在"热门板块"里选。
**打分**：复合 = 主力资金流排名 + 利润同比排名 + (close/MA20)排名。**参数**：hot_k、topn、rebal。
**Barra 结论**：R² 43%、α +4.3%(不显著)。**板块主力资金流只在 2020-2022 有效、2023 后失效**，超额主要是风格 beta。""",
 "S5 规模中性+质量+复合": """**选股**：趋势 & 质量。**打分**：复合打分**对 log 市值做截面回归取残差**(规模中性)，试图剔除小盘暴露。
**参数**：topn、rebal。
**Barra 结论**：α 不显著、**SIZE 仍 +0.34** —— 只中性化了打分，但选股池(趋势&质量)本身偏小盘，没真正中性，超额仍是小盘 beta。""",
 "S6 多因子融合": """**选股**：趋势 & 质量。**打分**：五因子排名和 = 动量(40d)+EP+低波(−20d vol)+主力资金流+利润同比。
**参数**：topn、rebal。
**Barra 结论**：α 年化 **+11.5%**、R² 18%、VALUE+0.69。**冻结 Test 夏普 0.75、DSR 67%** —— 次于 S2，仍未达标。""",
}


def page_tune():
    st.title("🎛️ 策略调参 · 自由调整参数看指标分布")
    st.caption("选任一策略，拖动参数实时回测：完整指标 + dev/冻结Test + Barra 暴露 + 净值，"
               "并显示当前配置在**全参数网格夏普分布**中的位置。回测 2020–2026、含成本、强制分散。")
    sf = _engine()
    name = st.sidebar.selectbox("策略", list(sf.STRATS))
    # 详细解释
    with st.expander("📖 策略详解（点开看全部细节）", expanded=True):
        st.markdown("### " + name)
        st.markdown(STRATEGY_DOCS.get(name, ""))
        st.markdown("---")
        st.markdown(STRATEGY_DOCS["等权基准"])
        st.markdown("---")
        st.markdown(COMMON_DOC)
    space = sf.STRATS[name][2]
    P = {}
    for k, opts in space.items():
        P[k] = st.sidebar.select_slider(k, options=opts, value=opts[len(opts) // 2])
    m = _eval(name, tuple(sorted(P.items())))

    c = st.columns(6)
    c[0].metric("年化", f"{m['cagr']*100:+.0f}%")
    c[1].metric("夏普(全)", f"{m['sharpe']:.2f}")
    c[2].metric("最大回撤", f"{m['maxdd']*100:+.0f}%")
    c[3].metric("Calmar", f"{m['calmar']:.2f}")
    c[4].metric("dev夏普", f"{m['dev_sharpe']:.2f}")
    c[5].metric("冻结Test夏普", f"{m['test_sharpe']:.2f}", f"{m['test_cagr']*100:+.0f}% CAGR")
    st.caption(f"持股 {m['nh']:.0f} | 子区间夏普 20-22 {m['sh1']:.2f} / 23-26 {m['sh2']:.2f} | "
               f"最差年 {m['worst']:+.0f}% | 当前参数 {P}")

    l, r = st.columns(2)
    nav = m["nav"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="策略净值", line=dict(color="crimson", width=2)))
    bn = (1 + sf.MKT.reindex(nav.index).fillna(0)).cumprod()
    fig.add_trace(go.Scatter(x=bn.index, y=bn.values, name="等权大盘", line=dict(color="gray", dash="dash")))
    fig.add_vline(x=sf.TEST_START.strftime("%Y-%m-%d"), line_dash="dot", line_color="gray")
    fig.update_layout(title="净值 vs 等权(灰虚线=冻结Test 2024-07 起)", height=330, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    l.plotly_chart(fig, use_container_width=True)

    # Barra 暴露
    exp = m["barra"].get("exposure", {})
    if exp:
        figb = px.bar(x=list(exp), y=list(exp.values()), title=f"Barra 风格暴露 | α年化 {m['barra']['alpha_ann']*100:+.0f}%(t{m['barra']['alpha_t']:+.1f}) R²{m['barra']['r2']:.0%}")
        figb.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10)); figb.add_hline(y=0, line_color="gray")
        r.plotly_chart(figb, use_container_width=True)

    # 参数网格夏普分布 + 当前位置
    gs = _grid(name)
    shs = [g["sharpe"] for g in gs]
    figd = px.histogram(x=shs, nbins=12, title=f"全参数网格夏普分布({len(shs)}个配置)，红线=当前配置")
    figd.add_vline(x=m["sharpe"], line_color="red", annotation_text="当前")
    figd.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="夏普", yaxis_title="配置数")
    st.plotly_chart(figd, use_container_width=True)
    st.info("⚠ 提醒：dev 夏普通常高于冻结 Test(过拟合衰减)；本族经确认 **DSR<90%、且 Test 跑输等权(夏普1.42)**，"
            "调参主要用于理解参数敏感性，不代表已是稳健可投策略。")


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


def page_largecap():
    st.title("🛡️ 大盘稳健族 · 低小盘暴露、看绝对收益")
    st.caption("针对'要稳定、对大多数股票成立、别压小盘、关注总收益'：股票池限**大中盘**(剔小盘)&趋势&质量，"
               "防御因子(价值/低波/质量)、打分**规模中性**、宽分散(30–50只)、月度调仓、含成本。"
               "对比**沪深300**(而非小盘等权)。")
    try:
        d = json.load(open(DD / "stable_largecap.json"))
    except Exception as e:  # noqa: BLE001
        st.error(f"结果未就绪，请先运行 examples/stable_largecap.py：{e}"); return
    rows = []
    for k, v in d.items():
        rows.append({"策略": k, "年化": v["cagr"], "夏普": v["sharpe"], "最大回撤": v["maxdd"],
                     "Calmar": v["calmar"], "夏普20-22": v["sh1"], "夏普23-26": v["sh2"],
                     "持股": round(v["nh"]), "SMB暴露": v["smb"], "α年化": v["alpha"],
                     "Test年化": v["test_cagr"], "Test夏普": v["test_sharpe"], "最优参数": str(v["P"])}
        )
    df = pd.DataFrame(rows)
    HS = {"cagr": 0.03, "sharpe": 0.25, "maxdd": -0.46, "test_cagr": 0.20, "test_sharpe": 1.06}
    fmt = df.copy()
    for c in ["年化", "最大回撤", "α年化", "Test年化"]: fmt[c] = (fmt[c] * 100).round(1)
    for c in ["夏普", "Calmar", "夏普20-22", "夏普23-26", "SMB暴露", "Test夏普"]: fmt[c] = fmt[c].round(2)
    st.dataframe(fmt, use_container_width=True)
    st.success(f"**沪深300 基准**：全期 年化 +3% / 夏普 0.25 / 回撤 −46%；Test 年化 +20% / 夏普 1.06。"
               f"→ 大盘族在**绝对收益、夏普、回撤**上**全面优于沪深300**，且 **SMB 暴露≈0(不靠小盘)**。")
    c1, c2 = st.columns(2)
    figs = go.Figure()
    figs.add_trace(go.Bar(x=df["策略"], y=df["夏普"], name="夏普"))
    figs.add_hline(y=HS["sharpe"], line_dash="dash", line_color="red", annotation_text="沪深300 0.25")
    figs.update_layout(title="夏普 vs 沪深300", height=330, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(figs, use_container_width=True)
    figm = px.bar(df, x="策略", y="SMB暴露", title="SMB(小盘)暴露 —— 越接近0越不依赖小盘")
    figm.add_hline(y=0, line_color="gray"); figm.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(figm, use_container_width=True)
    with st.expander("各策略最优参数 + 逐年"):
        for k, v in d.items():
            st.markdown(f"**{k}** — 参数 `{v['P']}` | SMB {v['smb']:+.2f} | α {v['alpha']*100:+.0f}% | 逐年 {v.get('by', {})}")


if page.startswith("🛡️"):
    page_largecap()
elif page.startswith("📊"):
    page_strategy_family()
elif page.startswith("🎛️"):
    page_tune()
else:
    page_event()
