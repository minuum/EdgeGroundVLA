#!/usr/bin/env python3
"""
Track 4: Background Inverse Masking Ablation (배경 역-마스킹 인과성 검증)

목적: 
  바스켓 영역(목표물)만 원본 이미지를 유지하고, 그 이외의 모든 복도 배경을 
  회색(128, 128, 128)으로 완전히 가렸을 때(역-마스킹), Stage 1 v2 모델이 
  여전히 원래의 조향 방향(LEFT/RIGHT/CENTER)을 맞출 수 있는지 검증합니다.
  
  이 실험을 통해 모델이 배경을 단순히 외워서 주행하는 것이 아니라, 
  목표물의 시각적 위치 특징(BBox Offset)만을 독립적으로 활용해 조향할 수 있음을 입증합니다.

Usage:
  .venv/bin/python3 scripts/exp54_background_mask_ablation.py
"""

import json
import sys
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"

PROJ_DIM  = 256
LM_DIM    = 2048
VIS_DIM   = 1024
DIR_IDX   = {"left": 0, "center": 1, "right": 2}
DIRS      = ["left", "center", "right"]

MIN_AREA      = 0.005   # 최소 바스켓 면적 임계치
N_SAMPLE      = 15      # 방향별 샘플 수
EPISODE_PHASE_MAX = 0.66   # 초기+중기만 사용
MASK_COLOR    = (128, 128, 128) # 배경을 채울 단색

OUT_DIR = ROOT / "docs" / "v5" / "background_mask_ablation"

def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    )
    vm = base.vision_model.to(device).eval()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(ckpt["image_proj"])
    image_proj.eval()

    text_proj = nn.Linear(LM_DIM, PROJ_DIM).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    anchor_feats = F.normalize(text_proj(ckpt["anchor_raw"].to(device)), dim=-1)
    return processor, vm, image_proj, anchor_feats

@torch.no_grad()
def get_conf(vm, image_proj, processor, anchor_feats, img, device, gt_idx):
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(image_proj(feat), dim=-1)
    sims = (proj @ anchor_feats.T)[0]
    pred_idx = sims.argmax().item()
    return sims[gt_idx].item(), pred_idx

def mask_background(img_pil, cx, cy, area, scale=1.0):
    """
    바스켓 영역(scale 배율)만 원본 이미지를 유지하고, 
    나머지 배경은 128 회색 단색으로 역-마스킹하여 채웁니다.
    """
    W, H = img_pil.size
    side = int(np.sqrt(area) * min(W, H) * scale)
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)

    # 회색 배경 이미지 생성
    masked = Image.new("RGB", (W, H), MASK_COLOR)
    # 바스켓 영역만 원본에서 크롭하여 마스킹 캔버스에 안착
    crop_basket = img_pil.crop((x1, y1, x2, y2))
    masked.paste(crop_basket, (x1, y1))
    return masked, (x1, y1, x2, y2)

def _try_font(size):
    for name in ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def save_pair(orig: Image.Image, bg_masked: Image.Image, row: dict, idx: int, out_dir: Path):
    """원본 | 배경차폐 나란히 붙여 PNG 저장 (글씨 오버레이 탑재)"""
    W, H = orig.size
    PAD = 4
    LABEL_H = 28
    canvas_w = W * 2 + PAD * 3
    canvas_h = H + LABEL_H * 2 + PAD * 2

    orig_overlay = orig.copy()
    draw_orig = ImageDraw.Draw(orig_overlay)
    font_large = _try_font(14)
    draw_orig.rectangle([10, 10, 140, 36], fill=(15, 23, 42, 200))
    draw_orig.text((16, 14), f"ORIG: {row['pred_orig'].upper()}", fill=(251, 191, 36), font=font_large)

    bg_masked_overlay = bg_masked.copy()
    draw_masked = ImageDraw.Draw(bg_masked_overlay)
    correct = row["correct"]
    
    if correct:
        # 배경을 가렸는데도 원래 방향을 올바르게 맞추었을 때 녹색 오버레이로 극찬 강조
        draw_masked.rectangle([10, 10, 200, 36], fill=(16, 185, 129, 220))
        draw_masked.text((16, 14), f"✔KEEP → {row['pred_mask'].upper()}", fill=(255, 255, 255), font=font_large)
    else:
        # 배경 소실로 실패(탈선) 시 빨간색 오버레이
        draw_masked.rectangle([10, 10, 200, 36], fill=(239, 68, 68, 220))
        draw_masked.text((16, 14), f"✘FAIL → {row['pred_mask'].upper()}", fill=(255, 255, 255), font=font_large)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    canvas.paste(orig_overlay,      (PAD,         PAD + LABEL_H))
    canvas.paste(bg_masked_overlay, (W + PAD * 2, PAD + LABEL_H))

    draw = ImageDraw.Draw(canvas)
    font_sm = _try_font(11)
    font_md = _try_font(13)

    d = row["direction"]
    col = {"left": (59, 130, 246), "center": (34, 197, 94), "right": (249, 115, 22)}[d]
    border_col = (16, 185, 129) if correct else (239, 68, 68)

    draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=border_col, width=2)

    # 상단 정보
    tag = f"[{d.upper()}] Inverse-Masking  cx={row['cx']:.2f}  area={row['area']:.4f}  phase={row['phase']:.2f}"
    draw.text((PAD + 2, 6), tag, fill=col, font=font_md)

    # 하단 정보
    conf_txt = f"orig_conf={row['conf_orig']:+.4f}  →  bg_masked_conf={row['conf_mask']:+.4f}"
    status_txt = "  ✔ SUCCESS" if correct else f"  ✘ FAILED (pred={row['pred_mask']})"
    draw.text((PAD + 2, H + LABEL_H + PAD + 4), conf_txt + status_txt,
              fill=(16, 185, 129) if correct else (239, 68, 68), font=font_sm)

    # 구분선
    draw.line([(W + PAD, PAD), (W + PAD, H + LABEL_H + PAD)], fill=(51, 65, 85), width=PAD)
    draw.text((PAD + W // 2 - 20, H + LABEL_H + PAD - 14), "original", fill=(148, 163, 184), font=font_sm)
    draw.text((W + PAD * 2 + W // 2 - 20, H + LABEL_H + PAD - 14), "bg_masked", fill=(148, 163, 184), font=font_sm)

    fname = f"bg_{d}_{idx:02d}_{'SUCCESS' if correct else 'FAIL'}.png"
    canvas.save(out_dir / fname)
    return fname

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    processor, vm, image_proj, anchor_feats = load_model(device)

    data = json.loads(DATA_PATH.read_text())

    # 샘플링
    dir_samples = defaultdict(list)
    for ep in data:
        d = ep["direction"]
        if len(dir_samples[d]) >= N_SAMPLE:
            continue

        all_idxs = [f["frame_idx"] for f in ep["frames"]]
        max_idx = max(all_idxs) if all_idxs else 0

        frames = [
            f for f in ep["frames"]
            if f["consistent"] and f["label"]
            and f.get("area_det") and f["area_det"] >= MIN_AREA
            and (f["frame_idx"] / max(max_idx, 1)) <= EPISODE_PHASE_MAX
        ]
        for fr in frames:
            if len(dir_samples[d]) < N_SAMPLE:
                fr["_phase"] = round(fr["frame_idx"] / max(max_idx, 1), 3)
                dir_samples[d].append((ep["episode"], fr))

    results = []
    SCALES = [1.5, 1.2, 1.0, 0.8, 0.5]
    # 스케일별 통계
    scale_stats = {sc: defaultdict(lambda: {"correct": 0, "total": 0}) for sc in SCALES}

    print(f"\n[START] 배경 역-마스킹(Inverse-Masking) Ablation 스윕 시작\n")
    print(f"  {'방향':<8} {'cx':>5} {'area':>6} {'phase':>6} {'conf_orig':>10} | {'예측 성공여부 (by scale)':<35}")
    print("  " + "-" * 90)

    dir_counters = defaultdict(int)
    for direction in DIRS:
        samples = dir_samples[direction]
        for ep_path, fr in samples:
            gt_idx = DIR_IDX[fr["label"]]
            cx     = fr["cx_det"]
            cy     = fr["cy_det"]
            area   = fr["area_det"]
            phase  = fr.get("_phase", -1)

            try:
                with h5py.File(ep_path, "r") as f:
                    img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]]).convert("RGB")
            except:
                continue

            conf_orig, pred_orig = get_conf(vm, image_proj, processor, anchor_feats, img, device, gt_idx)
            
            scale_results = {}
            for sc in SCALES:
                bg_masked_img, _ = mask_background(img, cx, cy, area, scale=sc)
                conf_mask, pred_mask = get_conf(vm, image_proj, processor, anchor_feats, bg_masked_img, device, gt_idx)
                
                correct = (pred_orig == pred_mask)
                scale_results[sc] = {
                    "conf_mask": round(conf_mask, 4),
                    "pred_mask": DIRS[pred_mask],
                    "correct": correct
                }
                
                scale_stats[sc][direction]["total"] += 1
                if correct:
                    scale_stats[sc][direction]["correct"] += 1

            row = {
                "direction": direction,
                "cx": round(cx, 3), "cy": round(cy, 3), "area": round(area, 4),
                "phase": round(phase, 3),
                "conf_orig": round(conf_orig, 4),
                "pred_orig": DIRS[pred_orig],
                "scales": {str(sc): scale_results[sc] for sc in SCALES}
            }

            # 1.0x 기준으로 이미지 시각화물 저장
            dir_counters[direction] += 1
            bg_masked_1x, _ = mask_background(img, cx, cy, area, scale=1.0)
            img_row = {
                "direction": direction,
                "cx": round(cx, 3), "area": round(area, 4), "phase": round(phase, 3),
                "conf_orig": round(conf_orig, 4),
                "conf_mask": scale_results[1.0]["conf_mask"],
                "pred_orig": DIRS[pred_orig],
                "pred_mask": scale_results[1.0]["pred_mask"],
                "correct": scale_results[1.0]["correct"]
            }
            fname = save_pair(img, bg_masked_1x, img_row, dir_counters[direction], OUT_DIR)
            row["img_file"] = fname
            results.append(row)

            # 스케일별 성공 여부 문자열 조립
            success_strs = "/".join([("OK" if scale_results[sc]["correct"] else "FAIL") for sc in SCALES])
            print(f"  {direction:<8} {cx:>5.2f} {area:>6.4f} {phase:>6.2f} {conf_orig:>10.4f} | {success_strs:<35}")

    # 최종 결과 통계 출력
    print(f"\n{'='*90}")
    print(f"  Track 4: 배경 역-마스킹(Inverse Masking) 스윕 최종 통계")
    print(f"{'='*90}")
    
    summary = {
        "min_area": MIN_AREA,
        "scales_evaluated": SCALES,
        "n_sample": N_SAMPLE,
        "scale_stats": {},
        "rows": results
    }

    for sc in SCALES:
        print(f"\n  [Mask Scale = {sc:.1f}x (바스켓 유지 배율)]")
        print(f"  {'방향':<8} {'n':>4} {'예측 성공률 (Stable Rate)':>28}")
        print("  " + "-" * 50)
        
        summary["scale_stats"][str(sc)] = {"per_direction": {}, "overall": {}}
        overall_correct = 0
        overall_total = 0
        
        for d in DIRS:
            s = scale_stats[sc][d]
            total = s["total"]
            correct = s["correct"]
            rate = (correct / total * 100) if total > 0 else 0
            
            overall_correct += correct
            overall_total += total
            
            print(f"  {d:<8} {total:>4} {rate:>25.1f}%")
            summary["scale_stats"][str(sc)]["per_direction"][d] = {
                "n": total, "correct": correct, "success_rate": round(rate, 1)
            }
            
        overall_rate = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        print(f"  --------------------------------------------------")
        print(f"  전체평균:  {overall_total:>4} {overall_rate:>25.1f}%")
        
        summary["scale_stats"][str(sc)]["overall"] = {
            "total": overall_total, "correct": overall_correct, "success_rate": round(overall_rate, 1)
        }
        
    print(f"{'='*90}")
    
    result_json = OUT_DIR / "results_background_mask.json"
    result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] JSON 결과 저장 완료 ➔ {result_json}")

if __name__ == "__main__":
    main()
