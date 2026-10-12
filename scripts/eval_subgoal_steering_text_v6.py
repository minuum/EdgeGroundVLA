#!/usr/bin/env python3
"""EGVLA2 텍스트 서브골 FiLM — V6 10-seed PM 비교 + 텍스트 영향력 프로브.

plan: docs/plans/plan_20261012_egvla2_text_subgoal_film.md
조건: steered_reg(bbox만, 기준) vs text_steered_reg(bbox + frozen CLIP text 서브골을 FiLM 조건에 concat)
입력: Stage 1 F2 캐시(bbox + 768-d 비전 특징) + 텍스트 임베딩 캐시(precompute_v6_text_embeddings.py)
주의: 텍스트 템플릿이 에피소드 direction과 1:1이므로 '방향 지시를 오라클로 준' 조건이다. 사용자 지시를
  이해한다는 일반화 주장이 아니라, 텍스트 채널이 조향에 쓰이는지만 본다.
프로브(학습된 text 모델, 테스트셋 전체): 정상 text / 다른 방향 text로 교체(shuffled) / 0벡터 → PM 변화·예측 flip rate.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("f2v6", ROOT / "scripts/eval_subgoal_steering_florence2_v6.py")
f2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f2)

TEXT_CACHE = ROOT / "docs/v5/bbox_nav_owl/v6_text_embed_cache.npz"
OUT_JSON = f2.OUT_DIR / "summary_text_steering_v6.json"
TEXT_DIM, TEXT_PROJ = 768, 32
DIRS = ["center", "weak_left", "strong_left", "weak_right", "strong_right"]


class TextBBoxSteeredMLP(nn.Module):
    """입력 x = [bbox(BBOX_DIM) | text(TEXT_DIM)] (train_eval 인터페이스 재사용)."""

    def __init__(self, hidden=f2.HIDDEN):
        super().__init__()
        self.text_proj = nn.Linear(TEXT_DIM, TEXT_PROJ)
        self.subgoal_enc = nn.Sequential(nn.Linear(f2.BBOX_DIM + TEXT_PROJ, 32), nn.ReLU(), nn.Linear(32, hidden))
        self.film = nn.Linear(hidden, hidden * 2)
        self.img_proj = nn.Linear(f2.IMG_DIM, hidden)
        self.trunk = nn.Sequential(nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden, 64), nn.ReLU(),
                                   nn.Linear(64, f2.NUM_CLASSES))

    def forward(self, xb, img_feat):
        bbox, text = xb[:, :f2.BBOX_DIM], xb[:, f2.BBOX_DIM:]
        s = self.subgoal_enc(torch.cat([bbox, self.text_proj(text)], dim=-1))
        gamma, beta = self.film(s).chunk(2, dim=-1)
        return self.trunk(gamma * self.img_proj(img_feat) + beta), (gamma, beta)


def load_text():
    c = np.load(TEXT_CACHE)
    allv = np.concatenate([c[d] for d in DIRS])
    mu, sd = allv.mean(0), allv.std(0) + 1e-6   # 15개 템플릿 기준 고정 표준화(방향 간 미세 차이 증폭)
    return {d: (c[d] - mu) / sd for d in DIRS}


def text_for(eps, text, rng, force_dirs=None):
    rows = []
    for ep in eps:
        for t in range(len(ep["frames"])):
            d = ep["direction"] if force_dirs is None else force_dirs(ep["direction"], rng)
            rows.append(text[d][rng.integers(len(text[d]))])
    return np.asarray(rows, dtype=np.float32)


def other_dir(d, rng):
    return rng.choice([x for x in DIRS if x != d])


@torch.no_grad()
def predict(model, xb, xi):
    model.eval()
    return model(torch.tensor(xb, device=f2.DEVICE), torch.tensor(xi, device=f2.DEVICE))[0].argmax(-1).cpu().numpy()


def main():
    dataset = f2.load_dataset()
    text = load_text()
    res = {"steered_reg": [], "text_steered_reg": [], "probe_correct": [], "probe_shuffled": [], "probe_zero": [],
           "flip_shuffled": [], "flip_zero": []}
    for seed in f2.SEEDS:
        tr, te = f2.make_episode_split(dataset, seed)
        Xb_tr, Xi_tr, y_tr, meta_tr = f2.build_windows(tr)
        Xb_te, Xi_te, y_te, meta_te = f2.build_windows(te)
        hb_tr = np.array([m["has_bbox"] for m in meta_tr])
        rng = np.random.default_rng(1000 + seed)
        T_tr, T_te = text_for(tr, text, rng), text_for(te, text, rng)

        acc, *_ = f2.train_eval(f2.SubgoalSteeredMLP(), Xb_tr, Xi_tr, y_tr, Xb_te, Xi_te, y_te, meta_te, seed,
                                has_bbox_tr=hb_tr, identity_reg_weight=0.3)
        res["steered_reg"].append(acc)

        m = TextBBoxSteeredMLP()
        acc_t, *_ = f2.train_eval(m, np.hstack([Xb_tr, T_tr]), Xi_tr, y_tr, np.hstack([Xb_te, T_te]), Xi_te, y_te,
                                  meta_te, seed, has_bbox_tr=hb_tr, identity_reg_weight=0.3)
        res["text_steered_reg"].append(acc_t)

        # 프로브: 학습 종료 시점 가중치 기준
        p_ok = predict(m, np.hstack([Xb_te, T_te]), Xi_te)
        T_sh = text_for(te, text, np.random.default_rng(2000 + seed), force_dirs=other_dir)
        p_sh = predict(m, np.hstack([Xb_te, T_sh]), Xi_te)
        p_z = predict(m, np.hstack([Xb_te, np.zeros_like(T_te)]), Xi_te)
        res["probe_correct"].append(float((p_ok == y_te).mean()))
        res["probe_shuffled"].append(float((p_sh == y_te).mean()))
        res["probe_zero"].append(float((p_z == y_te).mean()))
        res["flip_shuffled"].append(float((p_sh != p_ok).mean()))
        res["flip_zero"].append(float((p_z != p_ok).mean()))
        print(f"[seed {seed}] steered_reg={acc:.4f} text_steered_reg={acc_t:.4f} | probe correct={res['probe_correct'][-1]:.4f} "
              f"shuffled={res['probe_shuffled'][-1]:.4f} zero={res['probe_zero'][-1]:.4f} "
              f"flip_sh={res['flip_shuffled'][-1]:.3f}", flush=True)

    summ = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), raw=v) for k, v in res.items()}
    d, t, p = f2.paired_ttest(res["text_steered_reg"], res["steered_reg"])
    summ["text_vs_bbox_only"] = dict(mean_diff=float(d), t_stat=float(t), p_value=None if p is None else float(p))
    d2, t2, p2 = f2.paired_ttest(res["probe_correct"], res["probe_shuffled"])
    summ["probe_correct_vs_shuffled"] = dict(mean_diff=float(d2), t_stat=float(t2), p_value=None if p2 is None else float(p2))
    OUT_JSON.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    for k, v in summ.items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items() if a != "raw"})
    print("저장:", OUT_JSON)


if __name__ == "__main__":
    main()
