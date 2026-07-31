#!/usr/bin/env python3
"""저장된 세션 H5의 프레임을 OWL-v2로 재실행해 per-frame max score를 백필한다.

배경 (plan_20260731_future_research_dashboard_repurpose.md, T1-1):
  per-frame score 저장은 2026-07-31에 추가돼서 그 이전 세션(176개)에는 값이 없다.
  그런데 논문에 쓸 100개 세트가 전부 그 이전이라, threshold 오프라인 재판정과
  confidence 분포 Figure를 만들려면 소급 계산이 필요하다.

설계 결정 — 원본 H5를 수정하지 않고 사이드카 JSON으로 쓴다:
  - 원본 3.7GB는 이미 minum에게 전송됐고, 재실험의 1차 사료다. 중간에 끊기거나
    버그가 나도 원본이 훼손되면 안 된다.
  - 사이드카는 작아서(세션당 수 KB) minum에게 따로 보내기도 쉽다.
  출력: docs/inference_sessions/backfill_scores/session_<sid>.json

주의:
  GPU를 쓰므로 추론 서버(vla-stage2)와 경합한다. 실행 전 서버를 내리는 것을 권장
  (이 세션에서 메모리 압박으로 OOM이 난 이력이 있음).

사용:
  python3 scripts/backfill_grounding_scores.py --list            # 대상만 확인
  python3 scripts/backfill_grounding_scores.py --real-only       # 실그라운딩 프레임만(빠름)
  python3 scripts/backfill_grounding_scores.py --all-frames      # 캐시 프레임까지 전부
  python3 scripts/backfill_grounding_scores.py --sid 20260731_042235
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H5_DIR = ROOT / "docs" / "inference_sessions"
OUT_DIR = H5_DIR / "backfill_scores"
# 서버(stage2_v2_inference_server._OWL_SCORE_FLOOR)와 동일하게 유지 — 두 경로가
# 다른 floor를 쓰면 백필분과 신규 수집분의 분포가 미묘하게 어긋난다.
SCORE_FLOOR = float(os.getenv("VLA_OWL_SCORE_FLOOR", "0.01"))
PHRASE = os.getenv("VLA_BACKFILL_PHRASE", "gray basket")


def targets(sid_filter=None):
    out = []
    for p in sorted(H5_DIR.glob("session_*.h5")):
        sid = p.stem.replace("session_", "")
        if sid_filter and sid != sid_filter:
            continue
        try:
            with h5py.File(p, "r") as f:
                g = f.get("grounding")
                if g is None or "bbox" not in f.get("grounding", {}):
                    continue
                n = len(g["bbox"])
                # 이미 실측 score가 있으면(2026-07-31 이후 수집분) 건너뜀
                if "score" in g and float(np.max(g["score"][:])) >= 0:
                    continue
                cached = [int(v) for v in g["cached"][:]] if "cached" in g else [0] * n
        except Exception as e:
            print(f"  ! {sid} 열기 실패: {e}", file=sys.stderr)
            continue
        if (OUT_DIR / f"session_{sid}.json").exists():
            continue  # 이미 백필됨(resume)
        out.append((sid, p, n, cached))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="대상만 출력하고 종료")
    ap.add_argument("--real-only", action="store_true",
                    help="cached==0 (실제 그라운딩 호출) 프레임만 — 기본값")
    ap.add_argument("--all-frames", action="store_true",
                    help="캐시 재사용 프레임까지 전부 재계산(캐싱 손실 정량화용)")
    ap.add_argument("--sid", default=None, help="특정 세션만")
    ap.add_argument("--limit", type=int, default=0, help="N개 세션만 처리")
    args = ap.parse_args()
    real_only = not args.all_frames

    tg = targets(args.sid)
    if args.limit:
        tg = tg[:args.limit]
    n_frames = sum(sum(1 for c in cached if c == 0) if real_only else n
                   for _, _, n, cached in tg)
    print(f"대상 세션 {len(tg)}개 · 처리 프레임 {n_frames}개 "
          f"({'실호출만' if real_only else '전체 프레임'})")
    print(f"예상 소요: 약 {n_frames * 2 / 60:.0f}분 (GPU 2s/frame 가정)")
    if args.list or not tg:
        for sid, _, n, cached in tg:
            print(f"  {sid}  frames={n} real={sum(1 for c in cached if c==0)}")
        return

    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from PIL import Image

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev} · floor={SCORE_FLOOR} · phrase='{PHRASE}'")
    if dev.type == "cpu":
        print("⚠️  CPU 모드 — 프레임당 ~23초라 매우 느립니다. 중단 권장.", file=sys.stderr)
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(dev).eval()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    done_f = 0
    for si, (sid, path, n, cached) in enumerate(tg, 1):
        with h5py.File(path, "r") as f:
            imgs = f["observations/images"][:]
        idxs = [i for i in range(min(n, len(imgs)))
                if (not real_only) or (i < len(cached) and cached[i] == 0)]
        scores = [-1.0] * n
        for i in idxs:
            pil = Image.fromarray(imgs[i].astype(np.uint8)).convert("RGB")
            inp = proc(text=[[PHRASE]], images=pil, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp)
            res = proc.post_process_object_detection(
                out, threshold=SCORE_FLOOR,
                target_sizes=[(pil.height, pil.width)])[0]
            sc = res["scores"]
            scores[i] = round(float(sc.max().item()), 5) if len(sc) else 0.0
            done_f += 1
        (OUT_DIR / f"session_{sid}.json").write_text(json.dumps({
            "sid": sid, "scores": scores, "backfilled": True,
            "floor": SCORE_FLOOR, "phrase": PHRASE,
            "mode": "real_only" if real_only else "all_frames",
        }))
        el = time.time() - t_start
        rate = done_f / el if el > 0 else 0
        eta = (n_frames - done_f) / rate / 60 if rate > 0 else 0
        print(f"[{si}/{len(tg)}] {sid}  {len(idxs)}프레임  "
              f"누적 {done_f}/{n_frames}  {rate:.2f}f/s  ETA {eta:.0f}분", flush=True)

    print(f"완료: {done_f}프레임 / {time.time()-t_start:.0f}초 → {OUT_DIR}")


if __name__ == "__main__":
    main()
