#!/usr/bin/env python3
"""정직한 체크포인트 선택으로 F2 입력 서브골 비교 재측정 (CH75 낙관편향 제거).

기존 프로토콜은 테스트셋 acc가 최고인 epoch를 채택(낙관 편향). 여기서는 학습 에피소드의 20%를
inner-val로 떼어 그 acc가 최고인 epoch를 고르고, 그 epoch의 테스트 acc만 보고한다(테스트는 선택에 미사용).
함께 최종 epoch(120) 테스트 acc, 기존 방식(테스트 최고)도 기록해 편향 크기를 정량화.
TEXT_MODE=none(기본): concat/concat_small/steered/steered_reg, clip|onehot|curve|path: steered_reg vs text_steered_reg
출력: docs/v5/subgoal_steering_eval/summary_honest_f2_v6_{mode}.json
"""
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
MODE = os.environ.get("TEXT_MODE", "none")
if MODE != "none":
    spec = importlib.util.spec_from_file_location("txt", ROOT / "scripts/eval_subgoal_steering_text_v6.py")
    txt = importlib.util.module_from_spec(spec); spec.loader.exec_module(txt); f2 = txt.f2
else:
    spec = importlib.util.spec_from_file_location("f2v6", ROOT / "scripts/eval_subgoal_steering_florence2_v6.py")
    f2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f2); txt = None
OUT = f2.OUT_DIR / f"summary_honest_f2_v6_{MODE}.json"
DEV = f2.DEVICE


def run(model, tr, inner, te, seed, reg_w, text_fn=None):
    def arrays(eps):
        Xb, Xi, y, meta = f2.build_windows(eps)
        return Xb, Xi, y, meta
    A = [arrays(e) for e in (tr, inner, te)]
    if text_fn is not None:
        rng = np.random.default_rng(1000 + seed)
        A = [(np.hstack([a[0], text_fn(e, rng)]), a[1], a[2], a[3]) for a, e in zip(A, (tr, inner, te))]
    (Xb, Xi, y, meta), (Xbv, Xiv, yv, _), (Xbt, Xit, yt, _) = A
    hb = torch.tensor(np.array([m["has_bbox"] for m in meta]), dtype=torch.bool, device=DEV)
    torch.manual_seed(seed); model = model.to(DEV)
    c = np.bincount(y, minlength=8).astype(np.float32); c[c == 0] = 1
    w = torch.tensor(1 / c, device=DEV); w = w / w.sum() * 8
    lf = nn.CrossEntropyLoss(weight=w); opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, device=DEV)
    Xb_, Xi_, y_, Xbv_, Xiv_, yv_, Xbt_, Xit_, yt_ = map(T, (Xb, Xi, y, Xbv, Xiv, yv, Xbt, Xit, yt))
    va, ta = [], []
    for ep in range(f2.EPOCHS):
        model.train(); idx = torch.randperm(len(Xb_))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits, gate = model(Xb_[b], Xi_[b]); loss = lf(logits, y_[b])
            if reg_w > 0 and gate is not None:
                nb = ~hb[b]
                if nb.any():
                    loss = loss + reg_w * (((gate[0][nb] - 1) ** 2).mean() + (gate[1][nb] ** 2).mean())
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va.append((model(Xbv_, Xiv_)[0].argmax(-1) == yv_).float().mean().item())
            ta.append((model(Xbt_, Xit_)[0].argmax(-1) == yt_).float().mean().item())
    k = int(np.argmax(va))
    return dict(honest=ta[k], honest_epoch=k + 1, final=ta[-1], oracle_best=max(ta), inner_val=va[k])


def main():
    ds = f2.load_dataset()
    if MODE == "none":
        specs = {"concat": (lambda: f2.ConcatMLP(), 0.0, None), "concat_small": (lambda: f2.ConcatMLP(**f2.CONCAT_SMALL_DIMS), 0.0, None),
                 "steered": (lambda: f2.SubgoalSteeredMLP(), 0.0, None), "steered_reg": (lambda: f2.SubgoalSteeredMLP(), 0.3, None)}
    else:
        text = txt.load_text()
        specs = {"steered_reg": (lambda: f2.SubgoalSteeredMLP(), 0.3, None),
                 "text_steered_reg": (lambda: txt.TextBBoxSteeredMLP(), 0.3, lambda eps, rng: txt.text_for(eps, text, rng))}
    res = {k: [] for k in specs}
    for seed in f2.SEEDS:
        tr_all, te = f2.make_episode_split(ds, seed)
        tr, inner = f2.make_episode_split(tr_all, seed + 100)   # inner-val 20% (path_type 층화)
        for name, (mk, rw, tf) in specs.items():
            r = run(mk(), tr, inner, te, seed, rw, tf); res[name].append(r)
            print(f"[seed {seed}] {name:16s} honest={r['honest']:.4f}(ep{r['honest_epoch']}) final={r['final']:.4f} oracle_best={r['oracle_best']:.4f}", flush=True)
    summ = {}
    for n, rs in res.items():
        summ[n] = {k: dict(mean=float(np.mean([r[k] for r in rs])), std=float(np.std([r[k] for r in rs]))) for k in ("honest", "final", "oracle_best")}
        summ[n]["raw_honest"] = [r["honest"] for r in rs]
    names = list(specs)
    base = "concat" if MODE == "none" else "steered_reg"
    summ["paired_honest"] = {}
    for n in names:
        if n != base:
            d, t, p = f2.paired_ttest(summ[n]["raw_honest"], summ[base]["raw_honest"])
            summ["paired_honest"][f"{n}_vs_{base}"] = dict(mean_diff=float(d), t=float(t), p=None if p is None else float(p))
    if MODE == "none":
        d, t, p = f2.paired_ttest(summ["steered_reg"]["raw_honest"], summ["steered"]["raw_honest"])
        summ["paired_honest"]["steered_reg_vs_steered"] = dict(mean_diff=float(d), t=float(t), p=None if p is None else float(p))
    OUT.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    for n in names:
        print(n, {k: f"{v['mean']*100:.2f}±{v['std']*100:.2f}" for k, v in summ[n].items() if k != "raw_honest"})
    print(summ["paired_honest"])


if __name__ == "__main__":
    main()
