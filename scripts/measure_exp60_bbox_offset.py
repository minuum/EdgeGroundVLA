#!/usr/bin/env python3
"""
Exp60 S1: VLM(PaliGemma2 Exp59) bbox vs HSV GT bbox 오차 분포 측정

목적: Stage2 MLP가 학습한 HSV GT 분포와, 추론 시 들어오는 PaliGemma2 grounding
      bbox 분포의 차이(계통 오프셋·스케일·미검출)를 통계로 산출한다.
      이 통계가 Exp60 합성 노이즈 증강의 캘리브레이션 값이 된다.

방법:
  - 학습 데이터(bbox_dataset_full.json, HSV GT)의 이미지를 PaliGemma2 Exp59로 re-ground
  - GT has_bbox=True 프레임에 대해:
      Δcx = cx_vlm - cx_gt,  Δcy = cy_vlm - cy_gt,  area_ratio = area_vlm / area_gt
      miss = VLM이 None 반환 (GT엔 있는데 못 찾음)
  - GT has_bbox=False 프레임에 대해: VLM false detection 비율

산출물: docs/v5/exp60_bbox_offset_stats.json

Usage:
  .venv/bin/python3 scripts/measure_exp60_bbox_offset.py
  .venv/bin/python3 scripts/measure_exp60_bbox_offset.py --n-eps 30 --max-frames 6
"""
import sys, json, argparse, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import h5py
from PIL import Image

# Exp59Grounder는 CL 평가 스크립트에서 재사용 (중복 정의 방지)
from scripts.eval_exp59_closedloop import Exp59Grounder, PG2_PATH, EXP59_PATH

DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_PATH  = ROOT / "docs" / "v5" / "exp60_bbox_offset_stats.json"


def load_images(h5_path, frame_indices):
    with h5py.File(str(h5_path), "r") as f:
        imgs = f["observations"]["images"]
        return [Image.fromarray(imgs[i].astype("uint8")) for i in frame_indices]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-eps",      type=int, default=40,
                   help="측정에 사용할 에피소드 수 (고정 시드로 샘플)")
    p.add_argument("--max-frames", type=int, default=8,
                   help="에피소드당 측정 프레임 수 (균등 샘플)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    data = json.loads(DATA_PATH.read_text())
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(data))[:args.n_eps]
    eps = [data[i] for i in idx]
    print(f"[DEVICE] {device}  측정 ep: {len(eps)}  ep당 최대 {args.max_frames} frame", flush=True)

    print("[LOAD] PaliGemma2 Exp59 grounder...", flush=True)
    grounder = Exp59Grounder(PG2_PATH, EXP59_PATH, device)
    print("로드 완료\n", flush=True)

    dcx, dcy, area_ratio = [], [], []   # GT 有 & VLM 검출 성공
    n_pos = n_pos_hit = 0               # GT has_bbox=True
    n_neg = n_neg_falsedet = 0          # GT has_bbox=False

    for ei, ep in enumerate(eps):
        frames = ep["frames"]
        # 균등 샘플 프레임 인덱스
        n = len(frames)
        if n > args.max_frames:
            sel = np.linspace(0, n - 1, args.max_frames).round().astype(int)
        else:
            sel = np.arange(n)
        frame_idxs = [frames[s]["frame_idx"] for s in sel]
        try:
            imgs = load_images(ep["episode"], frame_idxs)
        except Exception as e:
            print(f"  skip {ep['episode']}: {e}", flush=True)
            continue

        for s, img in zip(sel, imgs):
            fr = frames[s]
            det = grounder.detect(img)   # (cx, cy, area) or None
            if fr["has_bbox"]:
                n_pos += 1
                if det is not None:
                    n_pos_hit += 1
                    cx, cy, area = det
                    dcx.append(cx - fr["cx"])
                    dcy.append(cy - fr["cy"])
                    if fr["area"] > 1e-6:
                        area_ratio.append(area / fr["area"])
            else:
                n_neg += 1
                if det is not None:
                    n_neg_falsedet += 1

        if (ei + 1) % 10 == 0 or (ei + 1) == len(eps):
            print(f"  {ei+1}/{len(eps)} ep done  (pos_hit={n_pos_hit}/{n_pos})", flush=True)

    def stat(a):
        a = np.array(a, dtype=np.float64)
        if len(a) == 0:
            return {"n": 0}
        return {
            "n": int(len(a)),
            "mean": float(a.mean()), "std": float(a.std()),
            "p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "min": float(a.min()), "max": float(a.max()),
        }

    miss_rate = 1.0 - (n_pos_hit / n_pos) if n_pos > 0 else 0.0
    false_det_rate = (n_neg_falsedet / n_neg) if n_neg > 0 else 0.0

    result = {
        "exp": "exp60_bbox_offset_stats",
        "config": {"n_eps": args.n_eps, "max_frames": args.max_frames},
        "n_pos_frames": n_pos, "n_pos_hit": n_pos_hit,
        "n_neg_frames": n_neg, "n_neg_falsedet": n_neg_falsedet,
        "miss_rate": miss_rate,            # GT 有인데 VLM 미검출 (→ MISS_P)
        "false_det_rate": false_det_rate,  # GT 無인데 VLM 검출
        "delta_cx":   stat(dcx),           # → OFFSET_MU_X, OFFSET_SD
        "delta_cy":   stat(dcy),           # → OFFSET_MU_Y
        "area_ratio": stat(area_ratio),    # → SCALE_J
    }
    OUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"\n{'='*60}")
    print(f"  VLM(PaliGemma2 Exp59) vs HSV GT bbox 오차")
    print(f"{'='*60}")
    print(f"  GT 有 프레임: {n_pos}  (VLM 검출 {n_pos_hit}, miss_rate={miss_rate:.3f})")
    print(f"  GT 無 프레임: {n_neg}  (VLM 오검출 {n_neg_falsedet}, rate={false_det_rate:.3f})")
    print(f"  Δcx : mean={result['delta_cx'].get('mean',0):+.4f} std={result['delta_cx'].get('std',0):.4f}")
    print(f"  Δcy : mean={result['delta_cy'].get('mean',0):+.4f} std={result['delta_cy'].get('std',0):.4f}")
    print(f"  area_ratio: mean={result['area_ratio'].get('mean',0):.3f} "
          f"p05={result['area_ratio'].get('p05',0):.3f} p95={result['area_ratio'].get('p95',0):.3f}")
    print(f"\n[SAVE] {OUT_PATH}")


if __name__ == "__main__":
    main()
