#!/usr/bin/env python3
"""OWL-v2 점수 분포 실측 — threshold 0.20이 분포의 어디에 놓여 있는가.

왜 필요한가:
  threshold를 "정탐/오탐 트레이드"로만 설명하면, **왜 0.05만 움직여도 실기 검출률이
  크게 흔들리는지**를 설명할 수 없다. 그 답은 분포에 있다 —
  임계값이 점수 분포의 평탄한 구간에 있으면 둔감하고, 밀집 구간(mode)에 있으면 민감하다.

  기존 자료의 한계:
   · 주석 JSON의 `confidence` 필드는 전부 1.0인 플레이스홀더다(CH65 Step 0에서 확인).
   · CH64의 "54.8%가 0.10~0.20 밴드"는 **젯슨에서 미검출된 프레임만** 다시 돌린 값이므로
     조건부 분포이고, 전체 분포의 최빈 구간이라고 말할 수 없다.
  → 그래서 threshold를 걸지 않은 **원점수(raw max score)** 분포를 직접 측정한다.

측정 방법:
  V6 val 프레임을 무작위 표집해 OWL-v2 "gray basket"을 돌리고,
  **threshold=0.0으로 후처리해 프레임별 최고 점수**를 기록한다(박스가 없으면 None).
  이렇게 하면 각 프레임이 "임계값을 얼마로 두면 검출되는가"를 알 수 있다.

보고 지표 (사전 고정):
  ① 점수 분위수 — 0.20 / 0.25가 분포의 몇 번째 백분위인가
  ② 구간별 히스토그램 — 0.20 주변에 프레임이 밀집해 있는가
  ③ threshold별 검출률 곡선 — 0.15 / 0.20 / 0.25 / 0.30에서 몇 %가 살아남는가
     그리고 **0.05 이동당 검출률 변화량**(민감도)
  ④ 라벨 위치별 분해 — CH65 65-3의 "강우가 가장 어렵다"가 점수에서도 보이는가

⚠️ 한계: 이 표본은 **주석에서 detected=True였던 프레임 위주**라 실제 운영 분포보다
  낙관적일 수 있다. 절대 검출률이 아니라 **0.20이 분포 어디에 있는지**를 보는 것이 목적이다.

출력: docs/v5/detector/owlv2_score_distribution.json
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT = ROOT / "docs/v5/detector/owlv2_score_distribution.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PHRASE = "gray basket"
GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]


def sample_frames(n, seed=0):
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = []
    for ei, ep in enumerate(ann):
        if ei not in val:
            continue
        for f in ep["frames"]:
            if f["grounding_cached"]:
                continue          # 상속값은 독립 표본이 아님
            rows.append(dict(ep=ep["episode"], fi=f["frame_idx"],
                             path=ep.get("path_type", "?"),
                             cx=f.get("cx_det")))
    r2 = np.random.default_rng(seed)
    pick = r2.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEV).eval()

    rows = sample_frames(args.n)
    print(f"표본 {len(rows)} 프레임 (V6 val, grounding_cached 제외) · phrase '{PHRASE}'",
          flush=True)

    cur, hf = None, None
    recs = []
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height
        inp = proc(text=[[PHRASE]], images=pil, return_tensors="pt").to(DEV)
        with torch.no_grad():
            o = model(**inp)
        # threshold=0.0 → 억제 없이 전부 받고 최고 점수만 취한다
        res = proc.post_process_object_detection(o, threshold=0.0,
                                                target_sizes=[(H, W)])[0]
        if len(res["scores"]) == 0:
            recs.append(dict(score=None, path=r["path"]))
        else:
            b = int(res["scores"].argmax())
            x1, y1, x2, y2 = res["boxes"][b].cpu().tolist()
            recs.append(dict(score=float(res["scores"][b]), path=r["path"],
                             cx=(x1 + x2) / 2 / W))
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    s = np.array([x["score"] for x in recs if x["score"] is not None])
    rep = {"n": len(recs), "n_scored": int(len(s)), "phrase": PHRASE}

    print("\n" + "=" * 82)
    print("OWL-v2 원점수 분포 — threshold 0.20이 분포의 어디에 있는가")
    print("=" * 82)
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    q = np.percentile(s, pcts)
    rep["percentiles"] = {f"p{p}": float(v) for p, v in zip(pcts, q)}
    print("\n  ① 점수 분위수")
    print("     " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(pcts, q)))
    for t in (0.20, 0.25):
        below = float((s < t).mean())
        rep[f"pct_below_{t}"] = below
        print(f"     → threshold {t:.2f} 는 분포의 상위 {100*(1-below):.1f}% 지점 "
              f"(이 값 미만이 {below*100:.1f}%)")

    print("\n  ② 구간별 히스토그램")
    edges = [0, .05, .10, .15, .20, .25, .30, .40, .60, 1.01]
    hist = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & (s < hi)
        hist.append(dict(lo=lo, hi=hi, n=int(m.sum()), frac=float(m.mean())))
        bar = "█" * int(m.mean() * 60)
        mark = "  ← threshold 0.20" if lo == .15 else ("  ← 0.25" if lo == .20 else "")
        print(f"     {lo:.2f}~{hi:.2f}  {int(m.sum()):4d}  {m.mean()*100:5.1f}%  {bar}{mark}")
    rep["histogram"] = hist

    print("\n  ③ threshold별 검출률과 민감도")
    print(f"{'threshold':>10s} {'검출률':>8s} {'직전 대비 변화':>14s}")
    prev = None
    curve = []
    for t in GRID:
        rate = float((s >= t).mean())
        d = "" if prev is None else f"{(rate-prev)*100:+.1f}%p"
        curve.append(dict(thresh=t, rate=rate))
        print(f"{t:10.2f} {rate*100:7.1f}% {d:>14s}")
        prev = rate
    rep["curve"] = curve
    # 0.20 주변 민감도
    r15, r20, r25 = [float((s >= t).mean()) for t in (0.15, 0.20, 0.25)]
    rep["sensitivity"] = {"0.15": r15, "0.20": r20, "0.25": r25,
                          "d_15_20": r15 - r20, "d_20_25": r20 - r25}
    print(f"\n     0.15 → 0.20 : {(r15-r20)*100:+.1f}%p 손실")
    print(f"     0.20 → 0.25 : {(r20-r25)*100:+.1f}%p 손실   "
          f"(0.05 이동당 이만큼 움직인다)")

    print("\n  ④ 경로 유형별 중앙 점수 — 어느 위치가 점수가 낮은가")
    by = {}
    for x in recs:
        if x["score"] is None:
            continue
        by.setdefault(x["path"], []).append(x["score"])
    for k in sorted(by, key=lambda k: np.median(by[k])):
        v = np.array(by[k])
        by20 = float((v >= 0.20).mean())
        by[k] = dict(n=len(v), median=float(np.median(v)), rate20=by20)
        print(f"     {k:22s} n={len(v):4d}  중앙 {np.median(v):.3f}  "
              f"0.20 통과 {by20*100:5.1f}%")
    rep["by_path"] = by

    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
