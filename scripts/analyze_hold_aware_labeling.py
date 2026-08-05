#!/usr/bin/env python3
"""hold-aware 라벨링 실측 — stride 5 다수결이 실제로 무엇을 하는가.

배경:
  학습 샘플 구성이 `build_windows_hold_aware(stride=5)`인데, 문서에는
  "결정 시점마다 1개 샘플, 라벨은 구간 다수결"이라고만 적혀 있었다.
  이 한 줄이 감추고 있는 것이 세 가지 있다.

  ① 윈도우의 6프레임은 **연속 프레임이 아니다.**
     idx = t - (window-1-k)*stride  →  t-25, t-20, t-15, t-10, t-5, t
     즉 6프레임이 **25프레임 구간에 걸쳐** 있다. "최근 6프레임"이라는 표현은 부정확하다.
  ② 라벨은 t 시점의 행동이 아니라 **[t, t+5) 구간의 다수결**이다.
     즉 **미래를 보고** 라벨을 만든다(구간 안에서 조작자가 실제로 유지한 행동).
  ③ `majority()`는 `np.bincount(...).argmax()`이므로 **동표일 때 가장 낮은 클래스 인덱스**가
     이긴다. 그리고 클래스 0은 STOP이다 → 동표 시 STOP/FORWARD 쪽으로 쏠릴 수 있다.

  이 스크립트는 ②③이 실제로 얼마나 일어나는지 센다.

측정 (사전 고정):
  A. 표본 수 변화 — 프레임 단위 대비 stride 5로 몇 배 줄어드는가
  B. 다수결 라벨이 순간 라벨(gts[t])과 다른 비율
  C. 동표 발생률과, 동표 시 선택된 클래스 분포 (인덱스 편향 확인)
  D. 클래스 분포 변화 — 프레임 단위 vs 다수결. FORWARD 편중이 완화되는가 악화되는가
  E. 윈도우 시간 폭 — stride 5 × (window-1) = 25 프레임이 실제 몇 초인가

출력: docs/v5/detector/hold_aware_labeling.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (          # noqa: E402
    CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, WINDOW,
)

OUT = ROOT / "docs/v5/detector/hold_aware_labeling.json"
STRIDE = 5
NAMES = ["STOP", "FWD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=STRIDE)
    a = ap.parse_args()
    st = a.stride

    base = torch.load(str(CACHE_V6), weights_only=False)
    print(f"에피소드 {len(base)} · window {WINDOW} · stride {st}", flush=True)

    n_frames = sum(len(e["gts"]) for e in base)
    inst, maj = [], []          # 결정 시점의 순간 라벨 / 다수결 라벨
    ties, tie_pick, tie_cands = 0, [], []
    seg_len = []
    for ep in base:
        g = np.asarray(ep["gts"], dtype=np.int64)
        n = len(g)
        for t in range(0, n, st):
            seg = g[t:t + st]
            seg_len.append(len(seg))
            c = np.bincount(seg, minlength=NUM_CLASSES)
            top = c.max()
            winners = np.flatnonzero(c == top)
            m = int(c.argmax())          # 실제 구현과 동일 — 동표 시 최소 인덱스
            inst.append(int(g[t]))
            maj.append(m)
            if len(winners) > 1:
                ties += 1
                tie_pick.append(m)
                tie_cands.append(sorted(int(w) for w in winners))

    inst = np.asarray(inst); maj = np.asarray(maj)
    n_dec = len(maj)
    rep = {"n_episodes": len(base), "n_frames": int(n_frames), "n_decisions": int(n_dec),
           "stride": st, "window": WINDOW,
           "window_span_frames": st * (WINDOW - 1)}

    print("\n" + "=" * 80)
    print("hold-aware 라벨링 실측")
    print("=" * 80)

    print(f"\n  A. 표본 수 — 프레임 {n_frames:,} → 결정 시점 {n_dec:,} "
          f"({n_frames/n_dec:.2f}배 감소)")
    print(f"     구간 길이 분포: 평균 {np.mean(seg_len):.2f} "
          f"(마지막 구간은 {st}보다 짧을 수 있음, 최소 {min(seg_len)})")

    diff = float((inst != maj).mean())
    rep["label_differs_from_instant"] = diff
    print(f"\n  B. 다수결 라벨 ≠ 순간 라벨(gts[t]) 비율: {diff*100:.2f}% "
          f"({int((inst!=maj).sum()):,}/{n_dec:,})")
    print("     → 이만큼은 '지금 하고 있는 행동'이 아니라 '앞으로 5프레임 동안 주로 할 행동'을 배운다")

    rep["tie_rate"] = ties / n_dec
    print(f"\n  C. 동표 발생 {ties:,}/{n_dec:,} = {ties/n_dec*100:.2f}%")
    if ties:
        tp = Counter(tie_pick)
        print("     동표에서 선택된 클래스(= 최소 인덱스):")
        for k, v in tp.most_common():
            print(f"       {NAMES[k]:6s} {v:5d} ({v/ties*100:5.1f}%)")
        cc = Counter(tuple(x) for x in tie_cands)
        print("     가장 흔한 동표 조합:")
        for k, v in cc.most_common(5):
            print(f"       {' vs '.join(NAMES[i] for i in k):28s} {v:5d}")
        rep["tie_pick"] = {NAMES[k]: v for k, v in tp.items()}
        rep["tie_top_combos"] = [{"cands": [NAMES[i] for i in k], "n": v}
                                 for k, v in cc.most_common(5)]
        # 동표에서 인덱스 편향의 크기 — 최소 인덱스가 아닌 무작위 선택 대비
        low_share = float(np.mean([1.0 / len(c) for c in tie_cands]))
        rep["tie_random_expected_share"] = low_share
        print(f"     ※ 무작위로 골랐다면 특정 클래스가 뽑힐 기대 확률 {low_share*100:.1f}% — "
              f"현 구현은 항상 최소 인덱스")

    print("\n  D. 클래스 분포 — 프레임 단위 vs 다수결(결정 시점)")
    allg = np.concatenate([np.asarray(e["gts"], dtype=np.int64) for e in base])
    fd = np.bincount(allg, minlength=NUM_CLASSES) / len(allg)
    md = np.bincount(maj, minlength=NUM_CLASSES) / n_dec
    print(f"{'클래스':8s} {'프레임 단위':>11s} {'다수결':>9s} {'변화':>9s}")
    dist = {}
    for i in range(NUM_CLASSES):
        print(f"{NAMES[i]:8s} {fd[i]*100:10.2f}% {md[i]*100:8.2f}% "
              f"{(md[i]-fd[i])*100:+8.2f}%p")
        dist[NAMES[i]] = {"frame": float(fd[i]), "majority": float(md[i])}
    rep["class_dist"] = dist
    lr_f = (fd[[2, 4, 6]].sum(), fd[[3, 5, 7]].sum())
    lr_m = (md[[2, 4, 6]].sum(), md[[3, 5, 7]].sum())
    rep["lr_balance"] = {"frame_left": float(lr_f[0]), "frame_right": float(lr_f[1]),
                         "maj_left": float(lr_m[0]), "maj_right": float(lr_m[1])}
    print(f"\n     좌계열 {lr_f[0]*100:.2f}% → {lr_m[0]*100:.2f}%  |  "
          f"우계열 {lr_f[1]*100:.2f}% → {lr_m[1]*100:.2f}%")
    print(f"     좌우 격차: 프레임 {abs(lr_f[0]-lr_f[1])*100:.2f}%p → "
          f"다수결 {abs(lr_m[0]-lr_m[1])*100:.2f}%p")

    print(f"\n  E. 윈도우 시간 폭 — stride {st} × (window {WINDOW} − 1) = "
          f"{st*(WINDOW-1)} 프레임 구간")
    print(f"     윈도우가 참조하는 인덱스: " +
          ", ".join(f"t−{(WINDOW-1-k)*st}" for k in range(WINDOW)).replace("t−0", "t"))
    print("     ※ 즉 '최근 6프레임'이 아니라 '25프레임에 걸친 6개 시점'이다")

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
