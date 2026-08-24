#!/usr/bin/env python3
"""leave-one-direction-out 결과 confusion matrix 재분석 (2026-08-21).

eval_leave_one_direction_out.py는 per-class recall만 남기고 confusion matrix를
저장하지 않았다. weak_right/strong_right held-out에서 F(FORWARD) 클래스
정확도가 유독 붕괴(19.6%/16.4%)하는 이유를 확인하려면 "F가 틀렸을 때 뭐로
예측했는지"가 필요하다 — 이 스크립트가 그걸 계산한다.

주의: 파이프라인에 끼워넣지 않는다. 분석 전용, 결과 재현(같은 seed=0, 1회).

출력: docs/v5/closed_loop_eval/exp73_leave_one_direction_out_confusion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import MLPActionHead, build_windows, train_one, DEVICE

CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp73_leave_one_direction_out_confusion.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
NUM_CLASSES = 8


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    for ep in eps:
        ep["direction"] = get_direction(ep["path_type"])

    results = {}
    for held in DIRECTIONS:
        val_eps = [e for e in eps if e["direction"] == held]
        tr_eps = [e for e in eps if e["direction"] != held]
        X_tr, y_tr, _ = build_windows(tr_eps)
        X_va, y_va, _ = build_windows(val_eps)

        acc, state, per_class = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed=0)
        model = MLPActionHead().to(DEVICE)
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_va, device=DEVICE)).argmax(1).cpu().numpy()

        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(y_va, pred):
            cm[t, p] += 1

        results[held] = dict(
            val_acc=float(acc),
            confusion=cm.tolist(),
            class_names=CLASS_NAMES,
            f_true_dist=(cm[1].tolist() if (y_va == 1).sum() else None),  # F가 뭘로 예측됐는지 분포
        )
        print(f"\n[{held}] acc={acc*100:.1f}% (seed=0)")
        print(" " * 8 + "".join(f"{n:>7s}" for n in CLASS_NAMES))
        for i, row in enumerate(cm):
            print(f"{CLASS_NAMES[i]:8s}" + "".join(f"{v:7d}" for v in row))

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")


if __name__ == "__main__":
    main()
