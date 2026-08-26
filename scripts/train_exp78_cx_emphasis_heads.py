#!/usr/bin/env python3
"""exp78 — cx를 강하게 반영하는 신규 헤드 3종 비교 (2026-08-26).

배경: docs/plans/research_20260826_cx_emphasis_head.md — cx를 "별도 브랜치로
추가"하는 기존 접근(cxgeom, hybrid, bbox_scale↑)은 mlp 대비 ±1p 이내로 효과가
미미했다. 이번엔 구조적으로 다른 3가지를 시도한다:
  film     — cx가 vis에 곱셈적 변조(FiLM)를 가함
  deltacx  — cx 시간 변화율(Δcx)을 명시 채널로 추가
  cxaux    — cx 회귀를 보조손실로 붙여 gradient에서 cx 활용 강제

exp77(Florence-2 phrase 그라운더 + Florence-2 vision, 현재 무작위 split 최고
성적 75.58%±0.07%p)의 기존 vis 캐시를 그대로 재사용 — 재인코딩 없음.
mlp/cxgeom도 같은 캐시로 재학습해 직접 비교 기준을 통일한다(기존 exp73
표는 다른 캐시 규격이라 완전히 apples-to-apples는 아니었음).

출력: docs/v5/closed_loop_eval/exp78_cx_emphasis_heads.json
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_exp73_stage1v3_heads import (
    MLPActionHead, CxGeomHead, FiLMHead, DeltaCxHead,
    build_windows, train_one, train_cxaux,
    WINDOW, BBOX_SCALE, NUM_CLASSES, VAL_RATIO, SPLIT_SEED,
)

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp78_cx_emphasis_heads.json"
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
EPOCHS = 300

HEADS = {"mlp": MLPActionHead, "cxgeom": CxGeomHead, "film": FiLMHead, "deltacx": DeltaCxHead}


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    print(f"[CACHE] 로드 {CACHE.name} ({len(eps)}ep)")

    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]
    print(f"[SPLIT] train={len(tr_eps)} / val={len(val_eps)}ep")

    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(val_eps)
    print(f"[WINDOW] train {X_tr.shape} / val {X_va.shape} (window={WINDOW}, bbox_scale={BBOX_SCALE})")

    results = {}
    for name, head_cls in HEADS.items():
        t0 = time.time()
        accs, best_overall, best_per_class = [], 0.0, None
        for seed in SEEDS:
            if name == "cxaux":
                continue  # cxaux는 아래 별도 분기
            acc, state, per_class = train_one(head_cls, X_tr, y_tr, X_va, y_va, seed, epochs=EPOCHS)
            accs.append(acc)
            if acc > best_overall:
                best_overall, best_per_class = acc, per_class
            print(f"  [exp78/{name}] seed={seed} val_acc={acc*100:.2f}%", flush=True)
        results[name] = {
            "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
            "val_acc_best": best_overall, "seeds": accs,
            "per_class_best": {CLASS_NAMES[c]: v for c, v in (best_per_class or {}).items()},
            "elapsed_min": (time.time() - t0) / 60,
        }
        print(f"  [exp78/{name}] mean={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}% "
              f"best={best_overall*100:.2f}%  ({(time.time()-t0)/60:.1f}min)", flush=True)

    # cxaux — 별도 학습 함수(train_cxaux) 사용
    t0 = time.time()
    accs, best_overall, best_per_class = [], 0.0, None
    for seed in SEEDS:
        acc, state, per_class = train_cxaux(X_tr, y_tr, X_va, y_va, seed, epochs=EPOCHS)
        accs.append(acc)
        if acc > best_overall:
            best_overall, best_per_class = acc, per_class
        print(f"  [exp78/cxaux] seed={seed} val_acc={acc*100:.2f}%", flush=True)
    results["cxaux"] = {
        "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
        "val_acc_best": best_overall, "seeds": accs,
        "per_class_best": {CLASS_NAMES[c]: v for c, v in (best_per_class or {}).items()},
        "elapsed_min": (time.time() - t0) / 60,
    }
    print(f"  [exp78/cxaux] mean={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}% "
          f"best={best_overall*100:.2f}%  ({(time.time()-t0)/60:.1f}min)", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")
    print("\n=== 요약(exp77 vis 캐시 공통) ===")
    for k, v in results.items():
        print(f"  {k:10s} mean={v['val_acc_mean']*100:5.2f}±{v['val_acc_std']*100:.2f}%  best={v['val_acc_best']*100:.2f}%")
    print("  (참고: exp77 원본 mlp 결과 mean=75.58%±0.07%p best=75.65% — 별도 build_windows 순서차 있을 수 있어 이 표의 mlp행과 비교)")


if __name__ == "__main__":
    main()
