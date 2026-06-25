"""轻量 GRU 序列模型 vs LightGBM：测「时序模型是否值得」的关键证据。

核心不是"GRU准不准",而是: ① GRU 与 LGBM 信号的相关性(去相关才有融合价值)
② 融合后(0.65*lgbm+0.35*gru rank)IC/策略夏普有没有涨。
轻量 GRU(hidden24/1层)、年度 walk-forward、CPU。若去相关且有增量→值得上TFT;否则砍掉。

跑法：python examples/ml_gru_fusion.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch, torch.nn as nn
torch.set_num_threads(4)
from examples.ml_alpha import build_features
from examples.ml_trade import simulate, metrics, load_signal, syms
from examples.strategy_family import idx

L = 20                                                   # 序列回看窗口
FEATS = ["ret1", "ret5", "ret20", "ret60", "std20", "turn20"]


class GRUNet(nn.Module):
    def __init__(self, k, h=24):
        super().__init__()
        self.gru = nn.GRU(k, h, batch_first=True)
        self.head = nn.Sequential(nn.Linear(h, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(o[:, -1]).squeeze(-1)


def main():
    print("构建因子张量 ...", flush=True)
    F, _l, close = build_features()
    F = {k: v.reindex(index=idx, columns=syms) for k, v in F.items()}
    # 横截面 rank 归一(scale-free), 堆成 [T,N,K]
    arr = np.stack([F[f].rank(axis=1, pct=True).fillna(0.5).values for f in FEATS], axis=2).astype(np.float32)
    raw10 = (close.shift(-10) / close - 1).clip(-0.5, 0.5)
    labr = raw10.rank(axis=1, pct=True)                   # 训练标签:未来10日收益 rank
    T, N, K = arr.shape
    ipos = {d: i for i, d in enumerate(idx)}

    def window_batch(t):                                 # 截至 t 的 [N,L,K]
        return arr[t - L + 1:t + 1].transpose(1, 0, 2)   # [N,L,K]

    train_days = [i for i, d in enumerate(idx) if d >= pd.Timestamp("2018-01-01") and i >= L][::5]
    pred = pd.DataFrame(np.nan, index=idx, columns=syms)
    for Y in range(2021, 2027):
        tr_idx = [t for t in train_days if idx[t].year < Y and ipos[idx[t]] + 10 < T]
        te_idx = [i for i, d in enumerate(idx) if d.year == Y and i >= L]
        if len(tr_idx) < 30 or not te_idx:
            continue
        # 构训练张量
        Xs, Ys = [], []
        for t in tr_idx:
            yb = labr.iloc[t].values
            xb = window_batch(t)
            ok = ~np.isnan(yb) & ~np.isnan(xb).any(axis=(1, 2))
            Xs.append(xb[ok]); Ys.append(yb[ok])
        X = torch.tensor(np.concatenate(Xs)); Yt = torch.tensor(np.concatenate(Ys), dtype=torch.float32)
        net = GRUNet(K); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.MSELoss()
        net.train()
        idx_all = np.arange(len(X))
        for ep in range(4):
            np.random.RandomState(ep).shuffle(idx_all)
            for s in range(0, len(idx_all), 4096):
                bi = idx_all[s:s + 4096]
                opt.zero_grad(); out = net(X[bi]); loss = lossf(out, Yt[bi]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            for t in te_idx:
                xb = window_batch(t); ok = ~np.isnan(xb).any(axis=(1, 2))
                if ok.sum() == 0:
                    continue
                p = np.full(N, np.nan); p[ok] = net(torch.tensor(xb[ok])).numpy()
                pred.iloc[t] = p
        print(f"  GRU 预测 {Y}: 训练序列 {len(X)} | 预测日 {len(te_idx)}", flush=True)

    pred.to_parquet("/home/claudeuser/econ/quant-research-lab/dashboard_data/ml_gru_signal.parquet")
    lgb = load_signal()
    # 评估
    common = [d for d in idx if d.year >= 2021]
    def ic(panel, how):
        v = [panel.loc[d].corr(raw10.loc[d], method=how) for d in common]
        return float(np.nanmean(v))
    # 信号相关性(逐日 GRU vs LGBM 截面相关,取均值)
    corr = np.nanmean([pred.loc[d].corr(lgb.loc[d]) for d in common if pred.loc[d].notna().any()])
    blend = (lgb.rank(axis=1, pct=True) * 0.65 + pred.rank(axis=1, pct=True) * 0.35)
    print("\n===== 时序模型(GRU) vs LightGBM 融合测试 =====")
    print(f"{'信号':22s} {'RankIC':>7s} {'PearsonIC':>9s}")
    for nm, P in [("LightGBM(生产)", lgb), ("GRU(轻量时序)", pred), ("融合 0.65LGB+0.35GRU", blend)]:
        print(f"{nm:22s} {ic(P,'spearman'):>7.4f} {ic(P,'pearson'):>9.4f}")
    print(f"\nGRU 与 LGBM 预测的横截面相关性(均值): {corr:.3f}  (越低越有融合价值)")
    for nm, P in [("LightGBM(生产)", lgb), ("融合", blend)]:
        port, tr = simulate(P, hold=10, realistic=True, exclude_st=True)
        m = metrics(port)
        print(f"  策略[{nm}] hold10: 年化{m['cagr']*100:+.0f}% 夏普{m['sharpe']:.2f} 回撤{m['maxdd']*100:+.0f}%")
    print("\n判据: 若 corr 高(>0.8) 且 融合IC/夏普≈LGBM → 时序模型无增量,不值得上 TFT。")


if __name__ == "__main__":
    main()
