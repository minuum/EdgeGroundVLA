#!/usr/bin/env python3
"""PM 천장 분석 (V6, F2 캐시 입력, steered_reg).
(1) 허용오차 PM(±k프레임)·전환 근처 오류 귀속·STOP 기여 (2) 라벨 타이밍 지터 시뮬레이션 천장
(3) 학습 데이터 양별 학습곡선(과적합 격차) (4) 시간 소프트 라벨(연속형 대리: ±2프레임 삼각 가중) vs 하드 라벨.
모든 수치는 최종 epoch(테스트 선택 없음) 기준. 출력: docs/v5/subgoal_steering_eval/pm_ceiling_analysis_v6.json
"""
import importlib.util, json
from pathlib import Path
import numpy as np, torch, torch.nn as nn
ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("f2v6", ROOT / "scripts/eval_subgoal_steering_florence2_v6.py")
f2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f2)
DEV = f2.DEVICE; SEEDS = [0, 1, 2, 3, 4]; K = 2


def soft_targets(eps, k=K):
    out = []
    for ep in eps:
        g = np.array([f["gt_class"] for f in ep["frames"]]); n = len(g)
        for t in range(n):
            v = np.zeros(8, np.float32)
            for j in range(-k, k + 1):
                v[g[min(n - 1, max(0, t + j))]] += (k + 1 - abs(j))
            out.append(v / v.sum())
    return np.asarray(out, np.float32)


def train(tr_eps, te_eps, seed, soft=False, epochs=120):
    Xb, Xi, y, meta = f2.build_windows(tr_eps); Xbt, Xit, yt, _ = f2.build_windows(te_eps)
    hb = torch.tensor(np.array([m["has_bbox"] for m in meta]), dtype=torch.bool, device=DEV)
    ys = torch.tensor(soft_targets(tr_eps) if soft else np.eye(8, dtype=np.float32)[y], device=DEV)
    torch.manual_seed(seed); m = f2.SubgoalSteeredMLP().to(DEV)
    c = np.bincount(y, minlength=8).astype(np.float32); c[c == 0] = 1
    w = torch.tensor(1 / c, device=DEV); w = w / w.sum() * 8
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, device=DEV)
    Xb_, Xi_, y_, Xbt_, Xit_, yt_ = T(Xb), T(Xi), T(y), T(Xbt), T(Xit), T(yt)
    for ep in range(epochs):
        m.train(); idx = torch.randperm(len(Xb_))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]; lg, (ga, be) = m(Xb_[b], Xi_[b])
            loss = -(ys[b] * torch.log_softmax(lg, -1) * w).sum(-1).mean()
            nb = ~hb[b]
            if nb.any(): loss = loss + 0.3 * (((ga[nb] - 1) ** 2).mean() + (be[nb] ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        ptr = m(Xb_, Xi_)[0].argmax(-1).cpu().numpy(); pte = m(Xbt_, Xit_)[0].argmax(-1).cpu().numpy()
    return dict(train_acc=float((ptr == y).mean()), test_acc=float((pte == yt).mean()), pred=pte, y=yt)


def eps_gt(eps): return [np.array([f["gt_class"] for f in ep["frames"]]) for ep in eps]


def analyze(te_eps, pred):
    gts = eps_gt(te_eps); off = 0; res = {f"tol{k}": [] for k in (0, 1, 2, 3)}; dist = []; err = []; stop_err = []
    stable = {k: [] for k in (1, 2)}
    for g in gts:
        n = len(g); p = pred[off:off + n]; off += n
        ch = [i for i in range(1, n) if g[i] != g[i - 1]]
        for t in range(n):
            for k in (0, 1, 2, 3):
                res[f"tol{k}"].append(p[t] in set(g[max(0, t - k):t + k + 1]))
            d = min([abs(t - c) for c in ch], default=99); dist.append(d); err.append(p[t] != g[t]); stop_err.append(g[t] == 0 or p[t] == 0)
            for k in (1, 2):
                if d > k: stable[k].append(p[t] == g[t])
    err = np.array(err); dist = np.array(dist); stop_err = np.array(stop_err)
    return dict({k: float(np.mean(v)) for k, v in res.items()},
                stable_frame_acc_beyond1=float(np.mean(stable[1])), stable_frame_acc_beyond2=float(np.mean(stable[2])),
                err_share_within2=float(err[dist <= 2].sum() / err.sum()), err_share_3to5=float(err[(dist >= 3) & (dist <= 5)].sum() / err.sum()),
                err_share_far=float(err[dist >= 6].sum() / err.sum()), err_share_stop_involved=float((err & stop_err).sum() / err.sum()))


def jitter_ceiling(te_eps, s, rng, trials=20):
    accs = []
    for _ in range(trials):
        agree = []
        for g in eps_gt(te_eps):
            n = len(g); ch = [0] + [i for i in range(1, n) if g[i] != g[i - 1]] + [n]
            nb = [0] + [int(np.clip(c + rng.integers(-s, s + 1), 1, n - 1)) for c in ch[1:-1]] + [n]
            nb = sorted(nb); q = np.empty(n, int)
            for a, b, i in zip(nb[:-1], nb[1:], range(len(nb))):
                q[a:b] = g[ch[i]]
            agree.append((q == g).mean() * 1.0 * n)
        accs.append(sum(agree) / sum(len(g) for g in eps_gt(te_eps)))
    return float(np.mean(accs))


def main():
    ds = f2.load_dataset(); out = {"per_seed": []}; rng = np.random.default_rng(0)
    agg = {}
    def add(k, v): agg.setdefault(k, []).append(v)
    for seed in SEEDS:
        tr, te = f2.make_episode_split(ds, seed)
        for s_ in (1, 2, 3): add(f"jitter_ceiling_pm_s{s_}", jitter_ceiling(te, s_, rng))
        for frac in (0.25, 0.5, 0.75, 1.0):
            sub = [tr[i] for i in np.random.default_rng(seed).permutation(len(tr))[:int(len(tr) * frac)]]
            r = train(sub, te, seed); add(f"hard_frac{frac}_train", r["train_acc"]); add(f"hard_frac{frac}_test", r["test_acc"])
            if frac == 1.0:
                for k, v in analyze(te, r["pred"]).items(): add(f"hard_{k}", v)
        r = train(tr, te, seed, soft=True); add("soft_train", r["train_acc"]); add("soft_test", r["test_acc"])
        for k, v in analyze(te, r["pred"]).items(): add(f"soft_{k}", v)
        print("seed", seed, {k: round(v[-1], 4) for k, v in agg.items() if k.endswith(("hard_frac1.0_test", "soft_test"))}, flush=True)
    summ = {k: dict(mean=float(np.mean(v)), std=float(np.std(v))) for k, v in agg.items()}
    (ROOT / "docs/v5/subgoal_steering_eval/pm_ceiling_analysis_v6.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    for k, v in summ.items(): print(f"{k:34s} {v['mean']*100:6.2f} ± {v['std']*100:.2f}")


if __name__ == "__main__":
    main()
