#!/usr/bin/env python3
"""A-2: Step2 동일 레시피로 PG2 vs OWL bbox 데이터셋 헤드 학습 비교 (apples-to-apples).

test_v5_bbox_nav_step2.py의 build_windows/make_episode_split/train_eval을 그대로 재사용하고
입력 데이터셋 파일만 바꿔서 5-seed 학습 → PM(test acc) mean±std 비교.
split은 원본과 동일(rng 42, path_type별 20% test), 학습 seed만 0~4 변화.

Usage: .venv/bin/python3 scripts/train_step2_owl_head.py
출력: docs/v5/bbox_nav_owl/head_compare.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 원본 스크립트에서 함수 재사용 (복사 아님 — 레시피 동일성 보장)
spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "test_v5_bbox_nav_step2.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)

DATASETS = {
    "pg2_baseline": ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json",
    "owl_th025": ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_owl.json",
}
SEEDS = [0, 1, 2, 3, 4]
OUT = ROOT / "docs" / "v5" / "bbox_nav_owl" / "head_compare.json"


def main():
    # 두 데이터셋의 공통 에피소드만 사용 (billy 전용 에피소드 제외 — apples-to-apples)
    loaded = {n: json.loads(p.read_text()) for n, p in DATASETS.items()}
    common = set.intersection(*({ep["episode"] for ep in d} for d in loaded.values()))
    print(f"공통 에피소드: {len(common)}")

    results = {}
    for name in DATASETS:
        dataset = [ep for ep in loaded[name] if ep["episode"] in common]
        train_eps, test_eps = step2.make_episode_split(dataset)
        X_tr, y_tr, _ = step2.build_windows(train_eps)
        X_te, y_te, meta_te = step2.build_windows(test_eps)
        print(f"\n=== {name}: train {len(X_tr)} / test {len(X_te)} ===")
        accs = []
        per_path_last = None
        for seed in SEEDS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            acc, preds = step2.train_eval(X_tr, y_tr, X_te, y_te)
            accs.append(acc)
            print(f"  seed{seed}: PM={acc:.3f}")
            # path_type별 (마지막 seed 기준)
            per_path = {}
            for p, gt, m in zip(preds, y_te, meta_te):
                pt = m["path_type"]
                d = per_path.setdefault(pt, {"correct": 0, "total": 0})
                d["total"] += 1
                d["correct"] += int(p == gt)
            per_path_last = per_path
        results[name] = {
            "pm_mean": float(np.mean(accs)), "pm_std": float(np.std(accs)),
            "pm_seeds": accs, "per_path_lastseed": per_path_last,
            "n_train": len(X_tr), "n_test": len(X_te),
        }
        print(f"  → PM {100*np.mean(accs):.1f}% ± {100*np.std(accs):.1f}%")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")
    a, b = results["pg2_baseline"]["pm_mean"], results["owl_th025"]["pm_mean"]
    print(f"\nPG2 {100*a:.1f}% vs OWL {100*b:.1f}%  (Δ {100*(b-a):+.1f}%p)")


if __name__ == "__main__":
    main()
