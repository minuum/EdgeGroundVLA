#!/usr/bin/env python3
"""bbox_scale 재검증 — phrase 그라운더(exp77) 기준 (2026-08-21).

기존 재검증(2026-08-17)은 OWL-v2 그라운더 기준으로 1.0/2.0/3.0이 73.8/74.2/73.8%로
무영향이라고 결론 냈다. 그라운더 자체가 훨씬 정확해진 지금(phrase 그라운딩), bbox
채널의 정보량이 달라졌을 수 있어 같은 실험을 exp77 캐시로 재현한다.

exp77의 캐시된 vis 특징을 그대로 쓰고 bbox_scale만 바꿔서 build_windows() 호출 —
재인코딩 없음.

주의: 파이프라인에 끼워넣지 않는다. 오프라인 하이퍼파라미터 재검증 전용.

출력: docs/v5/detector/bbox_scale_phrase_grounder.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, build_windows, train_one, VAL_RATIO, SPLIT_SEED, DEVICE,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/detector/bbox_scale_phrase_grounder.json"
SCALES = [1.0, 2.0, 3.0]
SEEDS = [0, 1, 2]


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]

    results = {}
    for scale in SCALES:
        X_tr, y_tr, _ = build_windows(tr_eps, bbox_scale=scale)
        X_va, y_va, _ = build_windows(val_eps, bbox_scale=scale)
        accs = []
        for seed in SEEDS:
            acc, _, _ = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
            accs.append(acc)
        results[str(scale)] = dict(val_acc_mean=float(np.mean(accs)),
                                    val_acc_std=float(np.std(accs)),
                                    val_acc_best=float(max(accs)), seeds=accs)
        print(f"bbox_scale={scale}: mean={np.mean(accs)*100:.2f}% "
              f"±{np.std(accs)*100:.2f}%p best={max(accs)*100:.2f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")


if __name__ == "__main__":
    main()
