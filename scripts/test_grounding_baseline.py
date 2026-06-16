#!/usr/bin/env python3
"""
VLM Grounding Baseline Test — L0 계층 검증

교수님 5/22 미팅 핵심:
  "객체를 인식해야 얘가 목표물이 될 수 있는 거고..."
  "그거부터 시작하자고. 객체가 인식하는지 안하는지부터."

Pure Kosmos-2 (`.vlms/kosmos-2-patch14-224`)가 우리 복도 이미지에서
"gray basket"을 grounding하는지 측정.

비교:
  - Base Kosmos-2 (학습 없음)
  - Stage 1 LoRA (exp53) — LoRA가 grounding을 개선/파괴했는가?

평가:
  - basket_present 프레임: grounding bbox IoU ≥ 0.3 → hit
  - basket_absent 프레임: grounding bbox 출력 → false positive

Usage:
  .venv/bin/python3 scripts/test_grounding_baseline.py
  .venv/bin/python3 scripts/test_grounding_baseline.py --model base
  .venv/bin/python3 scripts/test_grounding_baseline.py --model lora
  .venv/bin/python3 scripts/test_grounding_baseline.py --n-episodes 10
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH     = ROOT / ".vlms" / "kosmos-2-patch14-224"
LORA_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "exp53"
DATA_PATH    = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
DATA_PATH_FB = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"  # fallback

IOU_THRESHOLD = 0.3  # 교수님 기준: "grounding이 됐다"의 최소 IoU
MAX_NEW_TOKENS = 128
SAMPLE_PER_EP  = 5   # 에피소드당 샘플링할 프레임 수 (속도)


# ─── 유틸 ────────────────────────────────────────────────

def load_frame(h5_path, frame_idx):
    with h5py.File(h5_path, "r") as f:
        arr = f["observations"]["images"][frame_idx]
    return Image.fromarray(arr)


def compute_iou(pred_box, gt_cx, gt_cy, gt_area):
    """
    pred_box: (x1, y1, x2, y2) normalized [0,1]
    gt: cx, cy, area normalized
    gt_area를 정사각형으로 가정해 gt bbox 복원.
    """
    side = gt_area ** 0.5
    gx1 = gt_cx - side / 2
    gy1 = gt_cy - side / 2
    gx2 = gt_cx + side / 2
    gy2 = gt_cy + side / 2

    px1, py1, px2, py2 = pred_box
    ix1 = max(px1, gx1)
    iy1 = max(py1, gy1)
    ix2 = min(px2, gx2)
    iy2 = min(py2, gy2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    pa = (px2 - px1) * (py2 - py1)
    ga = (gx2 - gx1) * (gy2 - gy1)
    return inter / (pa + ga - inter + 1e-8)


# ─── 모델 로드 ───────────────────────────────────────────

def load_base_model(device):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("[MODEL] Base Kosmos-2 로딩...", flush=True)
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("[MODEL] Base Kosmos-2 로딩 완료", flush=True)
    return proc, model


def load_lora_model(device):
    """exp53 CLIP LoRA 어댑터를 붙인 Kosmos-2."""
    from transformers import AutoProcessor, AutoModelForVision2Seq
    try:
        from peft import PeftModel
    except ImportError:
        print("[WARN] peft 없음 → base 모델로 대체", flush=True)
        return load_base_model(device)

    print("[MODEL] Stage1 LoRA (exp53) 로딩...", flush=True)
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    base  = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    )
    # exp53 PEFT adapter 로드
    # fixed: 키 경로 수정된 어댑터 사용 (vision_model 경로 prefix 수정됨)
    adapter_dir = LORA_DIR / "clip_lora_adapter_fixed"
    if not adapter_dir.exists():
        adapter_dir = LORA_DIR / "clip_lora_adapter"  # fallback
    if not adapter_dir.exists():
        print(f"[WARN] LoRA adapter not found → base 모델로 대체", flush=True)
        return load_base_model(device)

    model = PeftModel.from_pretrained(base, str(adapter_dir)).to(device)
    model.eval()
    print(f"[MODEL] LoRA adapter 로딩 완료: {adapter_dir}", flush=True)
    return proc, model


# ─── 그라운딩 추론 ───────────────────────────────────────

def grounding_inference(proc, model, pil_image, device):
    """
    Kosmos-2 grounding 추론.
    Returns: list of (x1, y1, x2, y2) normalized boxes, or [] if none.
    """
    prompt = "<grounding><phrase>gray basket</phrase>"
    inputs = proc(text=prompt, images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}

    with torch.no_grad():
        gen = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=MAX_NEW_TOKENS,
        )

    decoded = proc.batch_decode(gen, skip_special_tokens=False)[0]
    _, entities = proc.post_process_generation(decoded, cleanup_and_extract=True)

    boxes = []
    for phrase, _, bboxes in entities:
        if "basket" in phrase.lower() or "gray" in phrase.lower():
            boxes.extend(bboxes)
    return boxes


# ─── 평가 루프 ───────────────────────────────────────────

def evaluate(proc, model, episodes, device, sample_per_ep=SAMPLE_PER_EP):
    """
    Returns dict with:
      present_hits, present_total, absent_fps, absent_total
      per_path: {path_type: {hits, total, fps, abs_total}}
    """
    rng = np.random.default_rng(42)
    stats = {
        "present_hits": 0, "present_total": 0,
        "absent_fps": 0,  "absent_total": 0,
    }
    per_path = defaultdict(lambda: {"hits": 0, "total": 0, "fps": 0, "abs_total": 0})
    per_iou  = []

    for ep_i, ep in enumerate(episodes):
        h5_path  = ep["episode"]
        pt       = ep.get("path_type", "unknown")
        frames   = ep["frames"]
        n        = len(frames)

        # 프레임 샘플링
        sample_n = min(sample_per_ep, n)
        sampled_idx = rng.choice(n, size=sample_n, replace=False)

        for fi in sampled_idx:
            fr = frames[fi]
            # 두 가지 스키마 지원: bbox_nav_exp46 (has_bbox/cx) vs bbox_frame_level (detected/cx_det)
            has_bbox = fr.get("has_bbox", fr.get("detected", False))
            cx   = fr.get("cx",   fr.get("cx_det",   0.5))
            cy   = fr.get("cy",   fr.get("cy_det",   0.5))
            area = fr.get("area", fr.get("area_det", 0.0))

            try:
                img = load_frame(h5_path, fr.get("frame_idx", fi))
            except Exception as e:
                print(f"  [skip] {h5_path} frame {fi}: {e}", flush=True)
                continue

            boxes = grounding_inference(proc, model, img, device)

            if has_bbox:
                # basket 있는 프레임: IoU 측정
                best_iou = 0.0
                for box in boxes:
                    iou = compute_iou(box, cx, cy, area)
                    best_iou = max(best_iou, iou)
                hit = best_iou >= IOU_THRESHOLD
                stats["present_hits"]  += int(hit)
                stats["present_total"] += 1
                per_path[pt]["hits"]   += int(hit)
                per_path[pt]["total"]  += 1
                per_iou.append(best_iou)
            else:
                # basket 없는 프레임: bbox 출력하면 false positive
                fp = len(boxes) > 0
                stats["absent_fps"]       += int(fp)
                stats["absent_total"]     += 1
                per_path[pt]["fps"]       += int(fp)
                per_path[pt]["abs_total"] += 1

        if (ep_i + 1) % 10 == 0 or (ep_i + 1) == len(episodes):
            pr = stats["present_hits"]
            pt_cnt = stats["present_total"]
            hit_r = pr / pt_cnt * 100 if pt_cnt else 0
            print(f"  [{ep_i+1}/{len(episodes)}] hit_rate={hit_r:.1f}% ({pr}/{pt_cnt})", flush=True)

    stats["mean_iou"] = float(np.mean(per_iou)) if per_iou else 0.0
    return stats, per_path


# ─── 결과 출력 ───────────────────────────────────────────

def print_results(label, stats, per_path, iou_thr=IOU_THRESHOLD):
    pr = stats["present_hits"]
    pt = stats["present_total"]
    fa = stats["absent_fps"]
    at = stats["absent_total"]
    hit_r = pr / pt * 100 if pt else 0
    fp_r  = fa / at * 100 if at else 0
    mean_iou = stats.get("mean_iou", 0.0)

    print(f"\n{'='*60}")
    print(f"[{label}] Grounding Results")
    print(f"{'='*60}")
    print(f"  basket_present: hit_rate={hit_r:.1f}%  ({pr}/{pt})  [IoU≥{iou_thr}]")
    print(f"  basket_absent:  fp_rate={fp_r:.1f}%   ({fa}/{at})")
    print(f"  mean_iou (when basket present): {mean_iou:.3f}")

    print("\n  Per path_type (basket_present only):")
    for path, d in sorted(per_path.items()):
        if d["total"] > 0:
            r = d["hits"] / d["total"] * 100
            print(f"    {path:20s}: {r:.1f}%  ({d['hits']}/{d['total']})")

    print()
    if hit_r >= 60:
        print("  ✅ Kosmos-2가 basket을 grounding한다 → L0 증거 확보")
    elif hit_r >= 30:
        print("  ⚠️  부분적 grounding (시야각/거리에 따라 실패)")
    else:
        print("  ❌ Grounding 실패 → 색상 기반 bbox로만 basket 위치 파악")


# ─── 메인 ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "lora", "both"], default="both",
                        help="테스트할 모델 (base=Pure Kosmos-2, lora=exp53 LoRA, both=둘 다)")
    parser.add_argument("--n-episodes", type=int, default=None,
                        help="테스트에 사용할 에피소드 수 (None=전체)")
    parser.add_argument("--sample-per-ep", type=int, default=SAMPLE_PER_EP,
                        help="에피소드당 샘플링할 프레임 수")
    parser.add_argument("--iou-threshold", type=float, default=IOU_THRESHOLD,
                        help="hit 기준 IoU threshold")
    args = parser.parse_args()

    iou_thr = args.iou_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}", flush=True)

    # 데이터 로드 (frame-level JSON 우선, fallback으로 full)
    if DATA_PATH.exists():
        data = json.loads(DATA_PATH.read_text())
        print(f"[DATA] frame-level: {DATA_PATH.name} ({len(data)} episodes)", flush=True)
    else:
        data = json.loads(DATA_PATH_FB.read_text())
        print(f"[DATA] fallback: {DATA_PATH_FB.name} ({len(data)} episodes)", flush=True)

    if args.n_episodes is not None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(data), size=min(args.n_episodes, len(data)), replace=False)
        data = [data[i] for i in sorted(idx)]
        print(f"[DATA] {len(data)} episodes 선택", flush=True)

    # 모델별 평가
    models_to_test = []
    if args.model in ("base", "both"):
        models_to_test.append(("Base Kosmos-2", load_base_model))
    if args.model in ("lora", "both"):
        models_to_test.append(("Stage1 LoRA (exp53)", load_lora_model))

    for label, load_fn in models_to_test:
        print(f"\n{'='*60}", flush=True)
        print(f"Testing: {label}", flush=True)
        print(f"{'='*60}", flush=True)

        proc, model = load_fn(device)
        stats, per_path = evaluate(proc, model, data, device, args.sample_per_ep)
        print_results(label, stats, per_path, iou_thr)

        # 메모리 해제
        del proc, model
        torch.cuda.empty_cache()

    print("\n[완료] 교수님 보고: basket present 프레임에서 hit_rate 확인 필요")


if __name__ == "__main__":
    main()
