"""复现 QuantMind 思路：Alpha158 式量价因子 + LightGBM 预测截面收益，做大盘稳健组合。

- 因子(~30，Alpha158 风格)：多窗口收益/均线乖离/波动/量能/换手/振幅/Amihud/量价相关 + EP/ROE/增长/规模
- 模型：LightGBM 回归预测**下月收益**；**严格 walk-forward**(逐年只用当年以前数据训练→预测当年，防前视)
- 组合：大盘(剔小盘)&质量内，按预测分 Top-N 等权、月度、含成本；规模中性诉求靠大盘池+Barra验证
- 评估：完整面板 + 冻结Test + Barra + 对比 L5/沪深300

跑法：python examples/ml_alpha.py
"""
import warnings; warnings.filterwarnings("ignore")
import pathlib, json
import numpy as np, pandas as pd, lightgbm as lgb
from examples.strategy_family import idx, ANN, trend, q, mv, ep, roe, pft, MKT
from quantlab.data.tushare_adapter import load_daily_ohlcv
from quantlab import barra, eval as ev, report as rp

DD = pathlib.Path("/home/claudeuser/econ/quant-research-lab/dashboard_data")
HS300 = pd.read_parquet(DD / "pullback_bench.parquet"); HS300.index = pd.to_datetime(HS300.index); HS300 = HS300["沪深300"].reindex(idx)
TEST_START = pd.Timestamp("2024-07-01")
syms = list(mv.columns)


def build_features():
    o = load_daily_ohlcv(); o = o[o.symbol.isin(syms)]
    pv = lambda c: o.pivot_table(index="trade_date", columns="symbol", values=c).reindex(index=idx, columns=syms)
    close, high, low = pv("adj_close"), pv("adj_high"), pv("adj_low")
    amount, turn = pv("amount"), pv("turnover_rate")
    ret = close.pct_change(fill_method=None)
    F = {}
    for w in (1, 5, 10, 20, 60): F[f"ret{w}"] = close / close.shift(w) - 1
    for w in (5, 10, 20, 60): F[f"ma{w}"] = close / close.rolling(w).mean() - 1
    for w in (5, 20, 60): F[f"std{w}"] = ret.rolling(w).std()
    F["maxret20"] = ret.rolling(20).max(); F["minret20"] = ret.rolling(20).min()
    F["hlpos20"] = (close - low.rolling(20).min()) / (high.rolling(20).max() - low.rolling(20).min() + 1e-9)
    F["volma5_20"] = amount.rolling(5).mean() / (amount.rolling(20).mean() + 1e-9)
    F["turn5"] = turn.rolling(5).mean(); F["turn20"] = turn.rolling(20).mean()
    F["turnchg"] = turn.rolling(5).mean() / (turn.rolling(60).mean() + 1e-9)
    F["amihud20"] = (ret.abs() / (amount + 1e-9)).rolling(20).mean() * 1e6
    F["amp20"] = ((high - low) / close).rolling(20).mean()
    F["ep"] = ep; F["roe"] = roe; F["growth"] = pft; F["logmv"] = np.log(mv.clip(lower=1))
    label = (close.shift(-21) / close - 1).clip(-0.5, 0.5)        # 下月(~21日)收益
    return F, label, close


def stack_monthly(F, label, close):
    me = list(close.groupby(close.index.to_period("M")).tail(1).index)
    me = [d for d in me if d >= pd.Timestamp("2018-01-01")]
    rows = []
    for d in me:
        df = pd.DataFrame({k: v.loc[d] for k, v in F.items()})
        df["y"] = label.loc[d]; df["date"] = d; df["symbol"] = df.index
        rows.append(df)
    data = pd.concat(rows).dropna(subset=["y"])
    return data, me


def walk_forward_predict(data, feats):
    """逐年 walk-forward：用某年之前所有月训练，预测该年。返回带 pred 的 OOS 子集。"""
    data = data.copy(); data["year"] = data["date"].dt.year
    out = []
    for Y in range(2021, 2027):
        tr = data[data["year"] < Y]; te = data[data["year"] == Y]
        if len(tr) < 5000 or len(te) == 0:
            continue
        Xtr = tr[feats].fillna(tr[feats].median()); Xte = te[feats].fillna(tr[feats].median())
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.03,
                              subsample=0.8, colsample_bytree=0.7, min_child_samples=100,
                              n_jobs=4, verbosity=-1)
        m.fit(Xtr, tr["y"])
        te = te.copy(); te["pred"] = m.predict(Xte)
        out.append(te)
    return pd.concat(out), m


def main():
    print("构建 Alpha158 式因子 + 标签 ...", flush=True)
    F, label, close = build_features()
    feats = list(F)
    data, me = stack_monthly(F, label, close)
    print(f"样本 {len(data)} 行 / {len(feats)} 因子 / {data['date'].nunique()} 月", flush=True)
    pred, model = walk_forward_predict(data, feats)
    print(f"OOS 预测 {len(pred)} 行 ({pred['date'].min().date()}~{pred['date'].max().date()})", flush=True)

    largecap = mv.rank(axis=1, ascending=False, pct=True) <= 0.5
    me_pred = sorted(pred["date"].unique())
    ret_d = close.pct_change(fill_method=None)
    vol20 = ret_d.rolling(20).std()
    TOPN = 50
    zc = lambda s: (s - s.mean()) / (s.std() + 1e-9)

    def build(score_col):
        """score_col: 'pred' 或 'blend' → 日度净收益(含成本)。"""
        weights = pd.DataFrame(0.0, index=me_pred, columns=syms)
        for d in me_pred:
            sub = pred[pred["date"] == d].set_index("symbol")
            ok = sub.index[(trend.loc[d].reindex(sub.index).fillna(False)) &
                           (q.loc[d].reindex(sub.index).fillna(False)) &
                           (largecap.loc[d].reindex(sub.index).fillna(False))]
            if score_col == "blend":
                sc = (zc(sub.loc[ok, "pred"]) + zc(ep.loc[d].reindex(ok)) + zc(-vol20.loc[d].reindex(ok))).dropna()
            else:
                sc = sub.loc[ok, "pred"].dropna()
            if len(sc) < TOPN:
                continue
            weights.loc[d, sc.nlargest(TOPN).index] = 1.0 / TOPN
        w = weights.reindex(idx).ffill().fillna(0.0)
        gross = (w.shift(1) * ret_d).sum(axis=1)
        turnover = (w - w.shift(1)).abs().sum(axis=1).fillna(0) / 2
        return (gross - turnover * 0.003).fillna(0.0).loc[idx >= me_pred[0]]

    port = build("pred")
    blend = build("blend")

    # 评估
    style = barra.build_style_factors(ret_d.shift(-1).clip(-0.21, 0.21), market=MKT,
                                      logmv=np.log(mv.clip(lower=1)), ep=ep, mom=close / close.shift(60) - 1,
                                      vol=ret_d.rolling(20).std(), growth=pft)
    out = {}
    for nm, p in [("ML 纯预测", port), ("ML+价值+低波 融合", blend)]:
        full = rp.performance_report(p, periods=ANN)
        tp = p.loc[p.index >= TEST_START]; tr = rp.performance_report(tp, periods=ANN)
        b = barra.barra_exposure(p, style)
        print(f"\n===== {nm}(大盘&质量, Top50, 月度, 含成本) =====")
        print(rp.format_report(full, "全期"))
        print(f"冻结Test: CAGR {tr['cagr']*100:+.0f}% | 夏普 {tr['sharpe']:.2f} | 回撤 {tr['max_drawdown']*100:+.0f}%")
        print("Barra: " + barra.format_exposure(b))
        out[nm] = dict(full=full, test=tr, barra=b, port=p)
    print(f"\n沪深300 同期夏普 {ev.sharpe(HS300.loc[port.index],ANN):.2f} | L5(大盘价值+低波)对照: 全期夏普0.76/回撤-21%")

    # 存盘供看板(用融合版作主)
    main_p = blend; mb = out["ML+价值+低波 融合"]
    nav = (1 + main_p).cumprod()
    pd.DataFrame({"date": nav.index, "ML+价值+低波": nav.values, "ML纯预测": (1 + port).cumprod().values,
                  "沪深300": (1 + HS300.reindex(nav.index).fillna(0)).cumprod().values}).to_parquet(DD / "ml_alpha_nav.parquet", index=False)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    json.dump({"ml": {k: out["ML 纯预测"]["full"][k] for k in ["cagr","sharpe","max_drawdown","calmar"]},
               "blend": {k: mb["full"][k] for k in ["cagr","sharpe","max_drawdown","calmar","sortino"]},
               "blend_test": {"cagr": mb["test"]["cagr"], "sharpe": mb["test"]["sharpe"], "maxdd": mb["test"]["max_drawdown"]},
               "ml_test": {"sharpe": out["ML 纯预测"]["test"]["sharpe"]},
               "barra": mb["barra"]["exposure"], "alpha_ann": mb["barra"]["alpha_ann"],
               "alpha_t": mb["barra"]["alpha_t"], "r2": mb["barra"]["r2"],
               "top_features": imp.head(15).round(0).to_dict(),
               "yearly": {int(y): round(((1+main_p[main_p.index.year==y]).prod()-1)*100) for y in range(2021,2027)}},
              open(DD / "ml_alpha.json", "w"), ensure_ascii=False, default=float)
    print("\n结果存 ml_alpha.json + ml_alpha_nav.parquet | Top因子:", list(imp.head(8).index))


if __name__ == "__main__":
    main()
