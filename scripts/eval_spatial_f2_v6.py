#!/usr/bin/env python3
"""F2 비전 특징의 공간 정보 보존 효과: 평균풀링 768-d vs 4x4 격자(16x768) — 정직한 체크포인트 선택, 10-seed.
조건: pooled_concat / pooled_steered_reg / spatial_concat / spatial_steered_reg (bbox 입력은 동일)
공간 인코더: 격자 셀별 공유 Linear(768->32)+ReLU 후 flatten(512) → 기존 헤드 (파라미터는 풀링 대비 소폭 증가).
선택: 학습 에피소드 20% inner-val로 epoch 선택, 테스트는 선택에 미사용. 최종 epoch 값도 병기.
출력: docs/v5/subgoal_steering_eval/summary_spatial_f2_v6.json
"""
import importlib.util, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("f2v6", ROOT / "scripts/eval_subgoal_steering_florence2_v6.py")
f2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f2)
DEV = f2.DEVICE; CELL = 32
SP = np.load(ROOT / "docs/v5/bbox_nav_owl/v6_florence2_spatial.npz")
_all = np.concatenate([SP[f"{e['episode']}|sp"].astype(np.float32) for e in f2.load_dataset()])
MU, SD = _all.mean((0, 1)), _all.std((0, 1)) + 1e-6   # 채널별 표준화
del _all


def spatial_feats(eps):
    return np.concatenate([((SP[f"{e['episode']}|sp"].astype(np.float32) - MU) / SD).reshape(len(e["frames"]), -1) for e in eps])


class CellEnc(nn.Module):
    def __init__(self):
        super().__init__(); self.p = nn.Linear(768, CELL)
    def forward(self, x): return torch.relu(self.p(x.view(x.shape[0], 16, 768))).flatten(1)


class SpConcat(nn.Module):
    def __init__(self):
        super().__init__(); self.enc = CellEnc()
        self.net = nn.Sequential(nn.Linear(f2.BBOX_DIM + 16 * CELL, 256), nn.ReLU(), nn.Dropout(0.25), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, f2.NUM_CLASSES))
    def forward(self, b, x): return self.net(torch.cat([b, self.enc(x)], -1)), None


class SpSteered(nn.Module):
    def __init__(self, hidden=f2.HIDDEN):
        super().__init__(); self.enc = CellEnc()
        self.subgoal = nn.Sequential(nn.Linear(f2.BBOX_DIM, 32), nn.ReLU(), nn.Linear(32, hidden))
        self.film = nn.Linear(hidden, hidden * 2); self.proj = nn.Linear(16 * CELL, hidden)
        self.trunk = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, f2.NUM_CLASSES))
    def forward(self, b, x):
        g, be = self.film(self.subgoal(b)).chunk(2, -1)
        return self.trunk(g * self.proj(self.enc(x)) + be), (g, be)


def run(model, A, seed, reg_w):
    (Xb, Xi, y, meta), (Xbv, Xiv, yv, _), (Xbt, Xit, yt, _) = A
    hb = torch.tensor(np.array([m["has_bbox"] for m in meta]), dtype=torch.bool, device=DEV)
    torch.manual_seed(seed); model = model.to(DEV)
    c = np.bincount(y, minlength=8).astype(np.float32); c[c == 0] = 1
    w = torch.tensor(1 / c, device=DEV); w = w / w.sum() * 8
    lf = nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, device=DEV)
    Xb_, Xi_, y_, Xbv_, Xiv_, yv_, Xbt_, Xit_, yt_ = map(T, (Xb, Xi, y, Xbv, Xiv, yv, Xbt, Xit, yt))
    va, ta, tra = [], [], []
    for ep in range(f2.EPOCHS):
        model.train(); idx = torch.randperm(len(Xb_))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]; lg, gate = model(Xb_[b], Xi_[b]); loss = lf(lg, y_[b])
            if reg_w > 0 and gate is not None:
                nb = ~hb[b]
                if nb.any(): loss = loss + reg_w * (((gate[0][nb] - 1) ** 2).mean() + (gate[1][nb] ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va.append((model(Xbv_, Xiv_)[0].argmax(-1) == yv_).float().mean().item())
            ta.append((model(Xbt_, Xit_)[0].argmax(-1) == yt_).float().mean().item())
            if ep == f2.EPOCHS - 1: tra.append((model(Xb_[:4000], Xi_[:4000])[0].argmax(-1) == y_[:4000]).float().mean().item())
    k = int(np.argmax(va))
    return dict(honest=ta[k], final=ta[-1], oracle_best=max(ta), train_final=tra[0], n_params=sum(p.numel() for p in model.parameters()))


def main():
    ds = f2.load_dataset(); res = {k: [] for k in ("pooled_concat", "pooled_steered_reg", "spatial_concat", "spatial_steered_reg")}
    for seed in f2.SEEDS:
        tr_all, te = f2.make_episode_split(ds, seed); tr, inner = f2.make_episode_split(tr_all, seed + 100)
        P = [f2.build_windows(e) for e in (tr, inner, te)]
        S = [(a[0], spatial_feats(e), a[2], a[3]) for a, e in zip(P, (tr, inner, te))]
        for name, mk, A, rw in [("pooled_concat", f2.ConcatMLP, P, 0.0), ("pooled_steered_reg", f2.SubgoalSteeredMLP, P, 0.3),
                                ("spatial_concat", SpConcat, S, 0.0), ("spatial_steered_reg", SpSteered, S, 0.3)]:
            r = run(mk(), A, seed, rw); res[name].append(r)
            print(f"[seed {seed}] {name:20s} honest={r['honest']:.4f} final={r['final']:.4f} train={r['train_final']:.3f} params={r['n_params']}", flush=True)
    summ = {n: {k: dict(mean=float(np.mean([r[k] for r in rs])), std=float(np.std([r[k] for r in rs]))) for k in ("honest", "final", "oracle_best", "train_final")}
            | {"raw_honest": [r["honest"] for r in rs], "n_params": rs[0]["n_params"]} for n, rs in res.items()}
    summ["paired"] = {}
    for a, b in [("spatial_concat", "pooled_concat"), ("spatial_steered_reg", "pooled_steered_reg"), ("spatial_steered_reg", "spatial_concat"), ("pooled_steered_reg", "pooled_concat")]:
        d, t, p = f2.paired_ttest(summ[a]["raw_honest"], summ[b]["raw_honest"])
        summ["paired"][f"{a}_vs_{b}"] = dict(mean_diff=float(d), t=float(t), p=None if p is None else float(p))
    (f2.OUT_DIR / "summary_spatial_f2_v6.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    for n in res: print(n, {k: f"{v['mean']*100:.2f}±{v['std']*100:.2f}" for k, v in summ[n].items() if isinstance(v, dict) and "mean" in v}, summ[n]["n_params"])
    print(summ["paired"])


if __name__ == "__main__":
    main()
