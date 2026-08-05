#!/usr/bin/env python3
"""CH68 68-9 — 액션 헤드는 cx에 단조적으로 반응하는가 (68-8의 이상 결과 원인 규명).

68-8 결과: 지시문을 "chair"로 바꾸면 그라운딩 cx는 0.484 → 0.089(좌측)로 확실히 이동하는데,
헤드의 좌질량은 **기대와 반대 방향으로** 움직였다(방향 일치율 35.1%, 우연 50% 미만).
그리고 |Δ좌질량| 0.104가 bbox 셔플 대조군 0.141보다 오히려 작았다.

가능한 설명 두 가지를 분리해야 한다:
  (A) 헤드가 cx를 거의 쓰지 않는다 (vis·시간 문맥에 의존)
  (B) 헤드는 cx를 쓰지만 chair의 cx=0.089가 **학습 분포 밖**이라 반응이 무너진다
      (V6 라벨 cx는 0.036~0.908이지만 실제 밀집 구간은 중앙부. bbox_scale=3.0으로
       스케일되므로 극단값은 더 크게 벗어난다)

68-8의 설계로는 (A)와 (B)를 구분할 수 없다 — chair cx가 0.089에 거의 고정돼 있어
"기대 방향"이 사실상 상수이고, 지표가 "좌질량이 늘었나" 하나로 축퇴한다
(68-7 ③에서 경고한 한계가 그대로 나타난 것이다).

그래서 이 스크립트는 검출기를 아예 빼고 **cx만 통제 변수로 스윕**한다:
  val 결정 시점의 실제 윈도우를 그대로 쓰고, **윈도우 전체 프레임의 cx만** 지정값으로
  덮어쓴다(cy·area·has_bbox·vis는 원본 유지). cx를 0.05~0.95로 19단계 스윕하며
  좌질량(좌계열[2,4,6] − 우계열[3,5,7])의 평균을 본다.

사전 고정 판정:
  · 좌질량이 cx에 대해 **단조 감소**(cx가 커질수록 우측 선호)해야 정상이다.
  · 중앙부(0.3~0.7)에서만 단조이고 극단에서 무너지면 → (B) 분포 밖 문제
  · 전 구간에서 기울기가 거의 0이면 → (A) cx를 거의 쓰지 않음
  · Spearman 상관과 구간별 기울기를 함께 보고한다.

출력: docs/v5/detector/l1_head_cx_response.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (          # noqa: E402
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, DEVICE, BBOX_SCALE, WINDOW,
)
from scripts.exp73_held_aware_train import build_windows_hold_aware   # noqa: E402
from scripts.exp73_window_cadence import swap_bboxes, OWL_ANN         # noqa: E402

OUT = ROOT / "docs/v5/detector/l1_head_cx_response.json"
HEADS = sorted((ROOT / "runs/v5_nav/mlp/exp73").glob(
    "exp73_owl_trackF_v6_mlp_holdaware_seed*.pt"))
MIRROR_HEADS = sorted((ROOT / "runs/v5_nav/mlp/exp73").glob(
    "exp73_owl_trackF_v6_mlp_mirroraug_seed*.pt"))
LEFT_CLS, RIGHT_CLS = [2, 4, 6], [3, 5, 7]
CX_GRID = np.round(np.linspace(0.05, 0.95, 19), 3)
FRAME_DIM = 4 + 256          # bbox 4 + vis 256


def val_eps(base):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(base)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    vs = set(idx[:nv])
    return [e for i, e in enumerate(base) if i in vs]


def load_head(hp):
    head = MLPActionHead().to(DEVICE)
    sd = torch.load(str(hp), map_location=DEVICE, weights_only=False)
    # head 표기가 arm에 따라 "mlp"/"exp73_mlp"로 다르다 — 구조는 동일하므로 둘 다 허용하고,
    # 입력 차원을 결정하는 window/bbox_scale만 엄격히 검증한다.
    assert sd["head"] in ("mlp", "exp73_mlp"), f"{hp.name}: head={sd['head']}"
    assert sd["window"] == WINDOW and sd["bbox_scale"] == BBOX_SCALE, \
        f"{hp.name}: window/bbox_scale 불일치 {sd['window']}/{sd['bbox_scale']}"
    head.load_state_dict(sd["model"])
    head.eval()
    return head


@torch.no_grad()
def left_mass(head, X):
    p = F.softmax(head(torch.from_numpy(X).float().to(DEVICE)), dim=1).cpu().numpy()
    return p[:, LEFT_CLS].sum(1) - p[:, RIGHT_CLS].sum(1)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def sweep(head, X):
    """윈도우 전체 프레임의 cx(각 프레임 벡터의 0번 원소)만 덮어쓴다."""
    curve = []
    for cx in CX_GRID:
        Xm = X.copy()
        Xm[:, :, 0] = cx * BBOX_SCALE          # 학습과 동일하게 스케일된 값으로 넣는다
        curve.append(float(left_mass(head, Xm).mean()))
    return np.array(curve)


def main():
    ap = argparse.ArgumentParser(); ap.parse_args()
    base = torch.load(str(CACHE_V6), weights_only=False)
    va = swap_bboxes(val_eps(base), OWL_ANN)
    X, _ = build_windows_hold_aware(va)
    print(f"val {len(va)}ep · 결정 시점 {len(X)} · cx 스윕 {len(CX_GRID)}단계", flush=True)

    # 학습 분포 참고 — 어디까지가 '분포 안'인지 판단 근거
    real_cx = np.concatenate([[b[0] for b in ep["bboxes"]] for ep in va])
    q = np.percentile(real_cx, [1, 5, 25, 50, 75, 95, 99])
    print("  실제 cx 분위(1/5/25/50/75/95/99): " + " ".join(f"{v:.3f}" for v in q), flush=True)

    rep = {"cx_grid": CX_GRID.tolist(), "n_decisions": int(len(X)),
           "real_cx_pct": {k: float(v) for k, v in
                           zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], q)},
           "arms": {}}

    for label, paths in [("holdaware (배포)", HEADS), ("mirroraug (65-9)", MIRROR_HEADS)]:
        if not paths:
            continue
        curves = np.stack([sweep(load_head(p), X) for p in paths])
        mean, std = curves.mean(0), curves.std(0)
        rho = float(np.mean([spearman(CX_GRID, c) for c in curves]))
        # 구간별 기울기 (좌질량 변화 / cx 변화)
        def slope(lo, hi):
            m = (CX_GRID >= lo) & (CX_GRID <= hi)
            return float(np.polyfit(CX_GRID[m], mean[m], 1)[0])
        seg = {"극좌 0.05~0.25": slope(0.05, 0.25), "중앙 0.30~0.70": slope(0.30, 0.70),
               "극우 0.75~0.95": slope(0.75, 0.95), "전체": slope(0.05, 0.95)}
        rep["arms"][label] = dict(n_seeds=len(paths), curve_mean=mean.tolist(),
                                  curve_std=std.tolist(), spearman=rho, slopes=seg)

        print(f"\n{'='*84}\n  {label}  ({len(paths)} seed)\n{'='*84}")
        print(f"{'cx':>7s} " + " ".join(f"{v:>7.3f}" for v in CX_GRID))
        print(f"{'좌질량':>7s} " + " ".join(f"{v:>7.3f}" for v in mean))
        print(f"{'±std':>7s} " + " ".join(f"{v:>7.3f}" for v in std))
        print(f"\n  Spearman(cx, 좌질량) = {rho:+.3f}   (정상이면 −1에 가까움)")
        for k, v in seg.items():
            print(f"    기울기 {k:14s} {v:+.4f}  {'정상(음수)' if v < 0 else '역전(양수)'}")

    print("\n  판정:")
    print("    · 전 구간 Spearman이 −1 근처 + 모든 구간 기울기 음수 → 헤드는 cx를 올바르게 쓴다")
    print("    · 중앙만 음수이고 극단이 양수 → 분포 밖 cx에서 반응이 무너진다 (68-8 설명 (B))")
    print("    · 전 구간 기울기 ~0 → 헤드가 cx를 거의 쓰지 않는다 (설명 (A))")
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
