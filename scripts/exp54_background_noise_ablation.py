#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
배경 가우시안 노이즈 및 가우시안 블러링 강건성 분석 실험 스크립트.
목적:
  복도 배경에 다양한 세기의 노이즈(Gaussian Noise std=0.05, 0.15, 0.30) 및 
  블러(Gaussian Blur 15x15, 31x31)를 가했을 때, 에이전트의 조향 방향 판단이 
  얼마나 강건하게 보존되는지 정량 평가합니다.
  이를 통해 교수님의 OOD 일반화 및 맵 암기 여부 의구심에 대응하는 
  학술적 대조 데이터를 확보합니다.

주석 언어: 한국어
"""

import json
import sys
import warnings
import argparse
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"

PROJ_DIM  = 256
LM_DIM    = 2048
VIS_DIM   = 1024
DIR_IDX   = {"left": 0, "center": 1, "right": 2}
DIRS      = ["left", "center", "right"]

MIN_AREA      = 0.005   # 최소 바스켓 면적 임계치
N_SAMPLE      = 12      # 각 방향별 평가 샘플 수 (총 36개 프레임)
EPISODE_PHASE_MAX = 0.66   # 초기+중기 프레임만 사용

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

def apply_background_perturbation(img_pil, cx, cy, area, mode="noise", param=0.1):
    """
    바스켓 영역(1.0x)은 원본으로 유지하고,
    그 외 배경 영역에만 노이즈(Gaussian Noise) 또는 블러(Gaussian Blur)를 적용합니다.
    """
    img_np = np.array(img_pil)
    H, W, C = img_np.shape

    # 바스켓 영역 경계 계산
    side = int(np.sqrt(area) * min(W, H))
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)

    # 배경 이미지 생성
    bg_np = img_np.copy()
    if mode == "noise":
        # 가우시안 노이즈 생성 (0~255 스케일)
        noise = np.random.normal(0, param * 255, (H, W, C)).astype(np.float32)
        noisy_img = np.clip(bg_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        bg_np = noisy_img
    elif mode == "blur":
        # 가우시안 블러 적용
        ksize = int(param)
        if ksize % 2 == 0:
            ksize += 1
        bg_np = cv2.GaussianBlur(bg_np, (ksize, ksize), 0)

    # 바스켓 영역만 원본에서 크롭하여 덮어쓰기
    bg_np[y1:y2, x1:x2] = img_np[y1:y2, x1:x2]
    
    return Image.fromarray(bg_np), (x1, y1, x2, y2)

def _try_font(size):
    for name in ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def save_perturbation_pair(orig: Image.Image, perturbed: Image.Image, row: dict, idx: int, condition_name: str, out_dir: Path):
    """원본 | 변형 이미지를 나란히 붙여 PNG 저장 (글씨 오버레이 포함)"""
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

    perturbed_overlay = perturbed.copy()
    draw_pert = ImageDraw.Draw(perturbed_overlay)
    correct = row["correct"]
    
    if correct:
        draw_pert.rectangle([10, 10, 200, 36], fill=(16, 185, 129, 220))
        draw_pert.text((16, 14), f"✔KEEP → {row['pred_pert'].upper()}", fill=(255, 255, 255), font=font_large)
    else:
        draw_pert.rectangle([10, 10, 200, 36], fill=(239, 68, 68, 220))
        draw_pert.text((16, 14), f"✘FAIL → {row['pred_pert'].upper()}", fill=(255, 255, 255), font=font_large)

    canvas = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    canvas.paste(orig_overlay,      (PAD,         PAD + LABEL_H))
    canvas.paste(perturbed_overlay, (W + PAD * 2, PAD + LABEL_H))

    draw = ImageDraw.Draw(canvas)
    font_sm = _try_font(11)
    font_md = _try_font(13)

    d = row["direction"]
    col = {"left": (59, 130, 246), "center": (34, 197, 94), "right": (249, 115, 22)}[d]
    border_col = (16, 185, 129) if correct else (239, 68, 68)

    draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=border_col, width=2)

    # 상단 정보
    tag = f"[{d.upper()}] Perturb ({condition_name})  cx={row['cx']:.2f}  area={row['area']:.4f}"
    draw.text((PAD + 2, 6), tag, fill=col, font=font_md)

    # 하단 정보
    conf_txt = f"orig_conf={row['conf_orig']:+.4f}  →  pert_conf={row['conf_pert']:+.4f}"
    status_txt = "  ✔ SUCCESS" if correct else f"  ✘ FAILED (pred={row['pred_pert']})"
    draw.text((PAD + 2, H + LABEL_H + PAD + 4), conf_txt + status_txt,
              fill=(16, 185, 129) if correct else (239, 68, 68), font=font_sm)

    draw.line([(W + PAD, PAD), (W + PAD, H + LABEL_H + PAD)], fill=(51, 65, 85), width=PAD)
    draw.text((PAD + W // 2 - 20, H + LABEL_H + PAD - 14), "original", fill=(148, 163, 184), font=font_sm)
    draw.text((W + PAD * 2 + W // 2 - 20, H + LABEL_H + PAD - 14), condition_name, fill=(148, 163, 184), font=font_sm)

    fname = f"pert_{d}_{idx:02d}_{condition_name}_{'SUCCESS' if correct else 'FAIL'}.png"
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
    
    # 평가할 변형(perturbation) 조건 정의
    CONDITIONS = {
        "blur_15": {"mode": "blur", "param": 15, "name": "Blur 15x15"},
        "blur_31": {"mode": "blur", "param": 31, "name": "Blur 31x31"},
        "noise_05": {"mode": "noise", "param": 0.05, "name": "Noise std=0.05"},
        "noise_15": {"mode": "noise", "param": 0.15, "name": "Noise std=0.15"},
        "noise_30": {"mode": "noise", "param": 0.30, "name": "Noise std=0.30"},
    }

    # 조건별 통계 데이터 구조
    cond_stats = {cond: defaultdict(lambda: {"correct": 0, "total": 0}) for cond in CONDITIONS}

    print(f"\n[START] 배경 변형(Perturbation) 강건성 평가 스윕 시작\n")
    
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
            except Exception as e:
                print(f"Error loading h5 file: {e}")
                continue

            conf_orig, pred_orig = get_conf(vm, image_proj, processor, anchor_feats, img, device, gt_idx)
            
            cond_results = {}
            dir_counters[direction] += 1
            
            for cond_key, cond_info in CONDITIONS.items():
                pert_img, _ = apply_background_perturbation(
                    img, cx, cy, area, mode=cond_info["mode"], param=cond_info["param"]
                )
                conf_pert, pred_pert = get_conf(vm, image_proj, processor, anchor_feats, pert_img, device, gt_idx)
                
                correct = (pred_orig == pred_pert)
                cond_results[cond_key] = {
                    "conf_pert": round(conf_pert, 4),
                    "pred_pert": DIRS[pred_pert],
                    "correct": correct
                }
                
                cond_stats[cond_key][direction]["total"] += 1
                if correct:
                    cond_stats[cond_key][direction]["correct"] += 1

                # 일부 대표 케이스에 대해서 시각화물 저장 (각 방향별 첫번째 샘플)
                if dir_counters[direction] == 1:
                    save_perturbation_pair(
                        img, pert_img, 
                        {
                            "direction": direction,
                            "cx": round(cx, 3), "area": round(area, 4),
                            "conf_orig": round(conf_orig, 4), "conf_pert": round(conf_pert, 4),
                            "pred_orig": DIRS[pred_orig], "pred_pert": DIRS[pred_pert],
                            "correct": correct
                        }, 
                        dir_counters[direction], cond_info["name"].replace(" ", "_"), OUT_DIR
                    )

            row = {
                "direction": direction,
                "cx": round(cx, 3), "cy": round(cy, 3), "area": round(area, 4),
                "phase": round(phase, 3),
                "conf_orig": round(conf_orig, 4),
                "pred_orig": DIRS[pred_orig],
                "conditions": cond_results
            }
            results.append(row)

            # 간단하게 출력
            pert_status = "/".join([("OK" if cond_results[ck]["correct"] else "FAIL") for ck in CONDITIONS])
            print(f"  {direction:<8} cx={cx:.2f} area={area:.4f} | {pert_status}")

    # 최종 요약 저장
    summary = {
        "min_area": MIN_AREA,
        "n_sample": N_SAMPLE,
        "conditions_evaluated": list(CONDITIONS.keys()),
        "cond_stats": {},
        "rows": results
    }

    print(f"\n{'='*90}")
    print(f"  배경 변형(Perturbation) 스윕 최종 통계")
    print(f"{'='*90}")

    for cond_key, cond_info in CONDITIONS.items():
        print(f"\n  [조건: {cond_info['name']}]")
        print(f"  {'방향':<8} {'n':>4} {'성공률 (Stable Rate)':>28}")
        print("  " + "-" * 50)
        
        summary["cond_stats"][cond_key] = {"name": cond_info["name"], "per_direction": {}, "overall": {}}
        overall_correct = 0
        overall_total = 0
        
        for d in DIRS:
            s = cond_stats[cond_key][d]
            total = s["total"]
            correct = s["correct"]
            rate = (correct / total * 100) if total > 0 else 0
            
            overall_correct += correct
            overall_total += total
            
            print(f"  {d:<8} {total:>4} {rate:>25.1f}%")
            summary["cond_stats"][cond_key]["per_direction"][d] = {
                "n": total, "correct": correct, "success_rate": round(rate, 1)
            }
            
        overall_rate = (overall_correct / overall_total * 100) if overall_total > 0 else 0
        print(f"  --------------------------------------------------")
        print(f"  전체평균:  {overall_total:>4} {overall_rate:>25.1f}%")
        
        summary["cond_stats"][cond_key]["overall"] = {
            "total": overall_total, "correct": overall_correct, "success_rate": round(overall_rate, 1)
        }

    result_json = OUT_DIR / "results_background_perturbation.json"
    result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] JSON 결과 저장 완료 ➔ {result_json}")

if __name__ == "__main__":
    main()
