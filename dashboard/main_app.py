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
try:
    _bench = pd.read_parquet(DD / "pullback_bench.parquet"); _bench.index = pd.to_datetime(_bench.index)
    HS300_DAILY = _bench["沪深300"]
except Exception:  # noqa: BLE001
    HS300_DAILY = None
try:
    _f500 = pathlib.Path.home() / ".local/share/quantlab/fundamentals/idx500.parquet"
    _s5 = pd.read_parquet(_f500)["close"]; _s5.index = pd.to_datetime(_s5.index)
    CSI500_DAILY = _s5.pct_change()                     # 中证500 日收益(分年度对冲基准)
except Exception:  # noqa: BLE001
    CSI500_DAILY = None
MT_CFG = DD / "ml_trade_saved_configs.json"             # 参数预设存档

st.set_page_config(page_title="Quant Research Lab", layout="wide")
page = st.sidebar.radio("页面", ["🛡️ 大盘稳健族(低小盘)", "🧪 ML交易调参", "📚 LightGBM详解",
                                 "🤖 ML Alpha(LightGBM复现)", "📊 策略族 & Barra 暴露",
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

    # 净值曲线
    try:
        nav = pd.read_parquet(DD / "stable_largecap_nav.parquet")
        fign = go.Figure()
        for col in nav.columns:
            is_bench = col == "沪深300"
            fign.add_trace(go.Scatter(x=nav.index, y=nav[col], name=col,
                                      line=dict(width=2.5 if not is_bench else 1.5,
                                                dash="dash" if is_bench else "solid",
                                                color="black" if is_bench else None)))
        fign.update_layout(title="净值曲线(全期 2020–2026，含成本；黑虚线=沪深300)", height=380,
                           margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fign, use_container_width=True)
    except Exception as e:  # noqa: BLE001
        st.warning(f"净值曲线未就绪(运行 examples/stable_largecap_nav.py)：{e}")

    c1, c2 = st.columns(2)
    figs = go.Figure()
    figs.add_trace(go.Bar(x=df["策略"], y=df["夏普"], name="夏普"))
    figs.add_hline(y=HS["sharpe"], line_dash="dash", line_color="red", annotation_text="沪深300 0.25")
    figs.update_layout(title="夏普 vs 沪深300", height=330, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(figs, use_container_width=True)
    figm = px.bar(df, x="策略", y="SMB暴露", title="SMB(小盘)暴露 —— 越接近0越不依赖小盘")
    figm.add_hline(y=0, line_color="gray"); figm.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(figm, use_container_width=True)

    # 完整 Barra 暴露热图 + α
    st.subheader("Barra 风格暴露（各策略对各风格因子的 beta）")
    exp_df = pd.DataFrame({k: v["barra"] for k, v in d.items()}).T
    b1, b2 = st.columns([3, 2])
    figh = px.imshow(exp_df, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto",
                     labels=dict(color="beta"), title="暴露热图(红=正/蓝=负)")
    figh.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10))
    b1.plotly_chart(figh, use_container_width=True)
    figal = px.bar(df, x="策略", y="α年化", title="Barra 剔除风格后年化 α")
    figal.add_hline(y=0, line_color="gray"); figal.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), yaxis_tickformat=".0%" if df["α年化"].abs().max() < 2 else None)
    b2.plotly_chart(figal, use_container_width=True)
    st.caption("关键看 **SIZE(小) 一列接近 0**(不靠小盘)；VALUE/VOL(低波) 正暴露=偏价值/低波(防御风格)。"
               "α 为剔除这些风格后的独立超额。")

    with st.expander("各策略最优参数 + 逐年"):
        for k, v in d.items():
            st.markdown(f"**{k}** — 参数 `{v['P']}` | SMB {v['smb']:+.2f} | α {v['alpha']*100:+.0f}% | 逐年 {v.get('by', {})}")


def page_ml():
    st.title("🤖 ML Alpha · 复现 QuantMind(Alpha158式因子 + LightGBM)")
    st.caption("复现 github.com/qusong0627/quantmind 核心：~25 个 Alpha158 式量价因子 + LightGBM 预测下月收益，"
               "**月度扩张窗口 walk-forward + purge**(每月用该月以前全部数据重训、近期也用上,标签做purge防泄漏)，"
               "大盘&质量内 Top-50 等权、月度、含成本。")
    try:
        d = json.load(open(DD / "ml_alpha.json"))
        nav = pd.read_parquet(DD / "ml_alpha_nav.parquet")
    except Exception as e:  # noqa: BLE001
        st.error(f"结果未就绪，请先运行 examples/ml_alpha.py：{e}"); return
    bl, ml = d["blend"], d["ml"]
    c = st.columns(6)
    c[0].metric("融合 全期年化", f"{bl['cagr']*100:+.0f}%")
    c[1].metric("融合 夏普", f"{bl['sharpe']:.2f}")
    c[2].metric("融合 回撤", f"{bl['max_drawdown']*100:+.0f}%")
    c[3].metric("融合 冻结Test夏普", f"{d['blend_test']['sharpe']:.2f}", f"{d['blend_test']['cagr']*100:+.0f}% CAGR")
    c[4].metric("纯ML 夏普", f"{ml['sharpe']:.2f}")
    c[5].metric("Barra α(t)", f"{d['alpha_ann']*100:+.0f}%", f"t={d['alpha_t']:.1f}")
    st.warning("**诚实结论**：ML(及融合)的 Barra **α 不显著(t≈0.3)、R²≈50-60%、MKT beta 0.5-0.7** → "
               "收益**主要是市场 beta，不是独立 alpha**；全期夏普(0.48-0.53)**未超过简单的 L5 大盘价值+低波(0.76/回撤-21%)**。"
               "冻结Test看着强(1.1-1.5)多来自 beta + 强 Test 窗口。**ML 在此数据上没带来真 alpha。**")
    fig = go.Figure()
    for col, c0 in [("ML+价值+低波", "crimson"), ("ML纯预测", "orange"), ("沪深300", "black")]:
        if col in nav:
            fig.add_trace(go.Scatter(x=nav["date"], y=nav[col], name=col,
                                     line=dict(width=2 if col != "沪深300" else 1.3, dash="dash" if col == "沪深300" else "solid", color=c0)))
    fig.update_layout(title="净值(2021–2026，含成本；黑虚=沪深300)", height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    exp = d["barra"]
    figb = px.bar(x=list(exp), y=list(exp.values()), title=f"Barra 暴露 | MKT高=市场beta, SIZE≈0=不靠小盘, α{d['alpha_ann']*100:+.0f}%(t{d['alpha_t']:.1f})")
    figb.add_hline(y=0, line_color="gray"); figb.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    c1.plotly_chart(figb, use_container_width=True)
    imp = d["top_features"]
    figi = px.bar(x=list(imp.values())[::-1], y=list(imp)[::-1], orientation="h", title="LightGBM Top 因子重要性")
    figi.update_layout(height=330, margin=dict(l=10, r=10, t=40, b=10))
    c2.plotly_chart(figi, use_container_width=True)
    st.caption(f"逐年收益(融合)：{d.get('yearly', {})}。Top因子以短期反转/动量/波动/换手为主。")


@st.cache_resource(show_spinner="加载 ML 交易引擎(首次约10秒)…")
def _ml_engine():
    from examples import ml_trade as mt
    sig = mt.load_signal(); mt._ohlc()  # 预热 OHLC 缓存
    return mt, sig


@st.cache_data(show_spinner="模拟中…")
def _ml_sim(hold, top_n, gap, stop, fund, realistic, exst):
    mt, sig = _ml_engine()
    port, trades = mt.simulate(sig, hold=hold, top_n=top_n, gap_thr=gap, stop_loss=stop, use_fund=fund,
                               realistic=realistic, exclude_st=exst)
    m = mt.metrics(port)
    return {"nav": m["nav"], "cagr": m["cagr"], "sharpe": m["sharpe"], "maxdd": m["maxdd"],
            "calmar": m["calmar"], "n": len(trades), "win": float((trades["ret"] > 0).mean()) if len(trades) else 0,
            "stop": float(trades["stopped"].mean()) if len(trades) else 0,
            "avg": float(trades["ret"].mean()) if len(trades) else 0}


def page_ml_trade():
    st.title("🧪 ML 交易调参 · 纯信号 + 后处理规则(全可调)")
    st.caption("**模型只学纯收益信号**(LightGBM 预测未来5/10/20日收益的横截面rank、多视野集成)；下面所有规则都是"
               "**模型之后的后处理**，可自由调：持有期、选股数、跳开过滤、止损、基本面池、真实撮合。次日开盘买入、含成本。")
    cfgs = json.load(open(MT_CFG)) if MT_CFG.exists() else {}
    with st.sidebar.expander("💾 参数预设", expanded=False):        # 加载放在控件之前(才能写 session_state)
        if cfgs:
            pick = st.selectbox("已存配置", ["—"] + list(cfgs))
            if pick != "—" and st.button("加载该配置"):
                for k, v in cfgs[pick].items():
                    st.session_state[k] = v
                st.rerun()
    hold = st.sidebar.select_slider("持有期(交易日)", [3, 5, 10, 20], value=10, key="mt_hold")
    top_n = st.sidebar.slider("选股数 Top-N", 10, 40, 20, 5, key="mt_topn")
    gap = st.sidebar.slider("跳开过滤(开盘相对昨收涨幅>此值则不买) %", 2, 12, 5, key="mt_gap") / 100
    stop = st.sidebar.slider("止损 %", 4, 20, 8, key="mt_stop") / 100
    fund = st.sidebar.checkbox("加基本面池(趋势&扣非ROE>0&利润增>0)", value=True, key="mt_fund")
    realistic = st.sidebar.checkbox("真实撮合(涨停不买/跌停·停牌不卖顺延)", value=True, key="mt_real")
    exst = st.sidebar.checkbox("排除 ST/*ST", value=True, key="mt_exst")
    with st.sidebar.expander("💾 保存当前参数", expanded=False):
        nm = st.text_input("命名", key="mt_savename")
        if st.button("保存") and nm:
            cfgs[nm] = {"mt_hold": hold, "mt_topn": top_n, "mt_gap": int(gap * 100), "mt_stop": int(stop * 100),
                        "mt_fund": fund, "mt_real": realistic, "mt_exst": exst}
            json.dump(cfgs, open(MT_CFG, "w"), ensure_ascii=False)
            st.success(f"已保存「{nm}」")
    try:
        r = _ml_sim(hold, top_n, gap, stop, fund, realistic, exst)
    except Exception as e:  # noqa: BLE001
        st.error(f"信号未就绪，请先运行 examples/ml_trade.py：{e}"); return
    c = st.columns(6)
    c[0].metric("年化", f"{r['cagr']*100:+.0f}%"); c[1].metric("夏普", f"{r['sharpe']:.2f}")
    c[2].metric("最大回撤", f"{r['maxdd']*100:+.0f}%"); c[3].metric("Calmar", f"{r['calmar']:.2f}")
    c[4].metric("交易笔数", f"{r['n']}"); c[5].metric("止损率", f"{r['stop']:.0%}")
    st.caption(f"逐笔胜率 {r['win']:.0%} | 平均每笔收益 {r['avg']:+.2%} | 当前: 持有{hold}日/Top{top_n}/跳开{gap:.0%}/止损{stop:.0%}/"
               f"基本面{'开' if fund else '关'}/真实撮合{'开' if realistic else '关'}/排除ST{'开' if exst else '关'}")
    if realistic:
        st.caption("✅ 真实撮合已开：涨停/停牌当日不买；止损或到期当天若跌停/停牌则**卖不掉、顺延到下一个可卖日按收盘价**"
                   "(真实反映暴跌时止损止不掉)。关掉则是理想成交(会高估止损保护)。")
    nav = r["nav"]
    bn = (1 + HS300_DAILY.reindex(nav.index).fillna(0)).cumprod() if HS300_DAILY is not None else None
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name="策略(净值)", line=dict(color="crimson", width=2)))
    if bn is not None:
        fig.add_trace(go.Scatter(x=bn.index, y=bn.values / bn.iloc[0], name="沪深300", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="净值(含成本)", height=360, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.info("注：这是把 ML 信号交给可调后处理规则的结果——**训练与规则解耦**。短持有(3/5日)通常被成本/反转吃掉，"
            "持有期拉长更稳(默认10日,夏普≈0.79);本族整体未超过 L5,主要是市场 beta。")
    st.subheader("📅 分年度：每年是否都有 alpha？")
    py = _per_year_alpha(r["nav"])
    disp = py.copy()
    for col in ["策略收益", "对冲后α收益"]:
        if col in disp:
            disp[col] = (disp[col] * 100).round(0).astype(int).astype(str) + "%"
    for col in ["策略夏普", "对冲后α夏普"]:
        if col in disp:
            disp[col] = disp[col].round(2)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    if "对冲后α夏普" in py:
        pos = int((py["对冲后α夏普"] > 0).sum()); tot = len(py)
        st.caption(f"「对冲后α」= 用**滚动60日β中性**对冲掉中证500 beta 后的残差(纯特质 alpha)。"
                   f"**{pos}/{tot} 年的 α 夏普为正**。若 α 只集中在个别年份(如2024)、其余≈0或负 → "
                   f"它不是稳定的独立 alpha,而是某些窗口的 beta/风格;若多数年份为正 → alpha 较稳。")
    _ml_decomp()


def _per_year_alpha(nav):
    """分年度:策略收益/夏普 + 对冲后(滚动60日β中性,中证500)残差α收益/夏普。"""
    ret = nav.pct_change().dropna()
    b = CSI500_DAILY.reindex(ret.index).fillna(0) if CSI500_DAILY is not None else None
    if b is not None:
        beta = (ret.rolling(60).cov(b) / (b.rolling(60).var() + 1e-12)).shift(1)
        resid = ret - beta * b
    rows = []
    for y in sorted({d.year for d in ret.index}):
        rr = ret[ret.index.year == y]
        if len(rr) < 60:                                 # 跳过数据稀疏的年份(OOS边界)
            continue
        row = {"年": y, "策略收益": float((1 + rr).prod() - 1),
               "策略夏普": float(rr.mean() / (rr.std() + 1e-12) * (242 ** 0.5))}
        if b is not None:
            res = resid[resid.index.year == y].dropna()
            row["对冲后α收益"] = float(res.sum())
            row["对冲后α夏普"] = float(res.mean() / (res.std() + 1e-12) * (242 ** 0.5)) if len(res) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _ml_decomp():
    """该策略(生产信号)的指标与拆解:信号质量/分层单调/breadth&对冲/容量/乐观度。"""
    try:
        a = json.load(open(DD / "ml_trade_analysis.json"))
    except Exception:  # noqa: BLE001
        st.caption("（拆解数据未就绪：运行 examples/precompute_ml_analysis.py）"); return
    st.markdown("---"); st.subheader("📐 这个策略的指标 & 拆解（同一生产信号）")
    sg = a["signal"]
    st.caption(f"信号：{sg['label']}｜{sg['retrain']}｜{sg['feats']} 因子")
    t1, t2, t3, t4, t5 = st.tabs(["信号质量", "分层单调性", "breadth & 对冲alpha", "资金容量", "乐观度拆解"])

    with t1:
        c = st.columns(4)
        c[0].metric("样本外 RankIC", f"{sg['mean_ic']:.4f}")
        c[1].metric("ICIR", f"{sg['icir']:.3f}"); c[2].metric("IC>0 胜率", f"{sg['win']*100:.0f}%")
        c[3].metric("IC t值", f"{sg['t']:.1f}")
        ts = pd.DataFrame(a["ic_ts"]); ts["date"] = pd.to_datetime(ts["date"])
        f = go.Figure(go.Scatter(x=ts["date"], y=ts["ic"], name="IC(20日均)"))
        f.add_hline(y=0, line_color="gray"); f.add_hline(y=sg["mean_ic"], line_dash="dot", line_color="red")
        f.update_layout(title="样本外 IC(20日平滑) 随时间", height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)

    with t2:
        d = a["deciles"]
        f = go.Figure(go.Bar(x=[f"D{i+1}" for i in range(10)], y=d["pure"], marker_color="steelblue"))
        f.update_layout(title="10 层未来10日平均收益%(D1低→D10高,纯信号全截面)", height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
        h = d["head"]
        st.caption(f"**头部细分**(Top-20≈top1.3%)：top10% {h['top10%']:+.2f} → top5% {h['top5%']:+.2f} → "
                   f"top2% {h['top2%']:+.2f} → top1% {h['top1%']:+.2f}%。头部越往里越强、无 roll-off → "
                   f"Top-N 选对了最强子区，低夏普是 breadth 问题；但中段(D2–D8)是噪声，edge 集中两端。")

    with t3:
        b = pd.DataFrame(a["breadth"])
        f = go.Figure()
        f.add_trace(go.Scatter(x=b["N"], y=b["long_sharpe"], name="多头夏普", mode="lines+markers", line=dict(color="crimson")))
        f.add_trace(go.Scatter(x=b["N"], y=b["hedge_net_ic"], name="对冲后净alpha夏普(中证500,滚动β扣贴水)",
                               mode="lines+markers", line=dict(color="seagreen")))
        f.add_hline(y=0, line_color="gray")
        f.update_layout(title="夏普 vs 持仓数 N(扩 breadth 不救:edge集中头部、衰减快过√breadth)",
                        height=300, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(f, use_container_width=True)
        st.dataframe(b.rename(columns={"N": "持仓数", "long_sharpe": "多头夏普", "maxdd": "回撤", "calmar": "Calmar",
                     "hedge_net_ic": "对冲净alpha", "ic_beta": "中证500β", "size": "市值%ile",
                     "q1ret": "2024Q1收益", "q1dd": "2024Q1回撤"})[["持仓数", "多头夏普", "回撤", "Calmar",
                     "对冲净alpha", "中证500β", "市值%ile", "2024Q1收益", "2024Q1回撤"]], use_container_width=True)
        st.caption("对冲用**滚动60日β中性(PIT)扣贴水**;只有最集中的 Top-20 对冲后净alpha为正(其余转负)→半中证500 beta+半弱alpha。")

    with t4:
        cap = pd.DataFrame(a["capacity"]); cap["aum"] = cap["aum_yi"].apply(lambda x: f"{x:.2f}亿" if x >= 1 else f"{x*1e4:.0f}万")
        f = go.Figure(go.Scatter(x=cap["aum_yi"], y=cap["net_cagr"], mode="lines+markers", line=dict(color="purple")))
        f.add_hline(y=0, line_color="gray")
        f.update_layout(title="净年化 vs 资金规模(亿元) — 净跌到0即容量天花板", height=280,
                        xaxis_type="log", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
        st.caption("Top-20 容量天花板 ≈ **5000万–1亿**:超过被市场冲击吃光收益(3亿时 27% 单子破 10% 参与率上限)。"
                   "edge 集中头部→既难分散又装不下钱,是小资金策略。")

    with t5:
        op = pd.DataFrame(a["optimism"])
        f = go.Figure(go.Bar(x=op["step"], y=op["cagr"] * 100,
                             marker_color=["#2c7", "#2c7", "#7b5", "#fb3", "#f93", "#e33"]))
        f.update_layout(title="乐观度拆解:同一信号逐步松绑严格项→年化%(末档=样本内泄漏)", height=300,
                        margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(f, use_container_width=True)
        st.caption("严格基线 +10% → 去成本/日频/小盘 +19% → **样本内泄漏(去walk-forward) +51%**。"
                   "别人 30–70% 的年化大头是样本内泄漏 + 小盘 beta,不是模型更强。")


def page_ml_pipeline():
    st.title("📚 LightGBM 数据构造 / 训练 / 测试 全过程")
    try:
        meta = json.load(open(DD / "ml_signal_meta.json"))
    except Exception as e:  # noqa: BLE001
        st.error(f"元数据未就绪：{e}"); return
    st.subheader("① 数据构造")
    st.markdown(f"- **股票池**：liq1500（大中盘为主，含部分退市）。\n"
                f"- **因子(X)**：{len(meta['feats'])} 个 Alpha158 式量价+基本面因子 → `{', '.join(meta['feats'])}`\n"
                f"- **标签(y)**：{meta['label']}（截面，clip ±50%）。\n"
                f"- **训练样本**：{meta['n_train_samples']:,} 行/视野（每 {meta['train_step']} 个交易日采样一次截面，降冗余）。\n"
                f"- 缺失值用**训练集中位数**填充（不泄漏测试信息）。")
    if meta.get("horizons"):
        st.success(f"**IC 提升技巧（经受控实验 examples/ml_ic_experiments*.py 确认）**：\n"
                   f"① 训练标签用「未来收益的**横截面 rank**」而非原始收益——让模型学排序、对离群与大盘整体涨跌更稳，"
                   f"是最大杠杆（季度口径 IC +50%+）；② **多视野集成**：对 {'/'.join(map(str, meta['horizons']))} 日收益各训一模型、"
                   f"预测取平均，跨视野去噪再 +6%。反例：对**特征**做横截面标准化反而有害（树模型本就尺度不敏感）、去市值中性也无益。")
    st.subheader(f"② 训练（{meta.get('retrain', '逐年')} walk-forward，严格防前视 + purge）")
    folds = pd.DataFrame(meta["folds"])
    st.markdown(f"**每预测一个月**，只用「该月之前的全部截面」重训 LightGBM(200树/叶31/lr0.03/行列采样)，"
                f"扩张窗口逐月滚动；训练样本须在预测起点前 **{meta.get('purge', 10)} 个交易日**（标签已实现、防泄漏）。"
                f"这样每个预测点都用满了它之前的所有数据（含最近几个月），而非冻结在去年底：")
    if "month" in folds.columns:
        st.dataframe(folds.rename(columns={"month": "预测月", "train_rows": "训练样本",
                     "train_end": "训练截止", "test_days": "预测交易日数"}), use_container_width=True, height=300)
    else:
        st.dataframe(folds.rename(columns={"year": "预测年", "train_rows": "训练样本", "test_days": "预测交易日数"}), use_container_width=True)
    st.subheader("③ 测试（样本外预测质量）")
    ic = pd.DataFrame(meta["ic"]);
    if len(ic):
        ic["date"] = pd.to_datetime(ic["date"])
        st.markdown(f"- **样本外 IC 均值 = {meta['mean_ic']}**（截面预测分 vs 实际未来收益的 Spearman 相关；>0.03~0.05 算有信息）。")
        figic = go.Figure(); figic.add_trace(go.Scatter(x=ic["date"], y=ic["ic"].rolling(20).mean(), name="IC(20日均)"))
        figic.add_hline(y=0, line_color="gray"); figic.add_hline(y=meta["mean_ic"], line_dash="dot", line_color="red")
        figic.update_layout(title="样本外 IC(20日平滑)随时间", height=300, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figic, use_container_width=True)
    imp = meta.get("importance", {})
    if imp:
        s = pd.Series(imp).sort_values().tail(15)
        figi = px.bar(x=s.values, y=s.index, orientation="h", title="LightGBM 因子重要性 Top15")
        figi.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(figi, use_container_width=True)
    st.info("信号训练**纯粹预测收益**、不含任何交易规则；持有期/选股数/跳开/止损/基本面都在「🧪 ML交易调参」页作后处理。")


if page.startswith("🛡️"):
    page_largecap()
elif page.startswith("🧪"):
    page_ml_trade()
elif page.startswith("📚"):
    page_ml_pipeline()
elif page.startswith("🤖"):
    page_ml()
elif page.startswith("📊"):
    page_strategy_family()
elif page.startswith("🎛️"):
    page_tune()
else:
    page_event()
