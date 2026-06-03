#!/usr/bin/env python3
"""
Track 3: Basket Masking Ablation

basket 영역을 회색(128,128,128)으로 가리면 Stage 1 v2 예측이 바뀌는가?
인과적 증거 — basket이 사라지면 confidence 떨어지면 "basket을 보고 있었다"

방법:
  1. consistent=True 프레임, area_det > MIN_AREA 인 것만 (basket이 충분히 커야 마스킹 의미 있음)
  2. cx_det, cy_det, area_det로 basket 위치 특정 → 1.5배 영역 gray masking
  3. Stage 1 v2에 원본/마스킹 각각 입력
  4. confidence(정답 클래스 cosine similarity) 변화 측정

결과 해석:
  confidence 감소 30%↑ → basket 영역에 의존 ✅
  confidence 감소 10~30% → 부분 의존
  confidence 변화 없음  → basket 외 정보로 분류 ⚠️

Usage:
  .venv/bin/python3 scripts/exp54_basket_mask_ablation.py
"""

import json, sys, warnings, argparse
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
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"

PROJ_DIM  = 256
LM_DIM    = 2048
VIS_DIM   = 1024
DIR_IDX   = {"left": 0, "center": 1, "right": 2}
DIRS      = ["left", "center", "right"]

MIN_AREA      = 0.005   # 마스킹 의미 있으려면 최소 이 면적 이상
MASK_SCALE    = 1.5     # bbox 크기의 1.5배로 마스킹 (약간 넉넉하게)
MASK_COLOR    = (128, 128, 128)
N_SAMPLE      = 15      # 방향별 샘플 수
# 에피소드 내 상대 위치 0~1 중 이 비율 이하 프레임만 사용 (도착 직전 제외)
# 0.33 = 초기만, 0.66 = 초기+중기, 1.0 = 전체
EPISODE_PHASE_MAX = 0.66   # 초기+중기(앞 66%)만 사용

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "v5" / "masking_ablation_earlymid"


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
    """gt 클래스에 대한 cosine similarity 반환"""
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(image_proj(feat), dim=-1)
    sims = (proj @ anchor_feats.T)[0]  # (3,)
    pred_idx = sims.argmax().item()
    return sims[gt_idx].item(), pred_idx


def mask_basket(img_pil, cx, cy, area, scale=MASK_SCALE):
    """cx/cy/area 기반 영역을 gray로 마스킹"""
    W, H = img_pil.size
    side = int(np.sqrt(area) * min(W, H) * scale)
    half = side // 2
    bx = int(cx * W)
    by = int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)

    masked = img_pil.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x1, y1, x2, y2], fill=MASK_COLOR)
    return masked, (x1, y1, x2, y2)


DIR_COLORS = {"left": (59, 130, 246), "center": (34, 197, 94), "right": (249, 115, 22)}
LABEL_BG   = (15, 23, 42)


def _try_font(size):
    for name in ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def save_pair(orig: Image.Image, masked: Image.Image, row: dict, idx: int, out_dir: Path):
    """원본 | 마스킹 나란히 붙여 PNG로 저장."""
    W, H = orig.size
    PAD = 4
    LABEL_H = 28
    canvas_w = W * 2 + PAD * 3
    canvas_h = H + LABEL_H * 2 + PAD * 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), LABEL_BG)
    canvas.paste(orig,   (PAD,         PAD + LABEL_H))
    canvas.paste(masked, (W + PAD * 2, PAD + LABEL_H))

    draw = ImageDraw.Draw(canvas)
    font_sm = _try_font(11)
    font_md = _try_font(13)

    d   = row["direction"]
    col = DIR_COLORS[d]
    flipped = row["flipped"]
    border_col = (239, 68, 68) if flipped else (71, 85, 105)

    # 테두리
    draw.rectangle([0, 0, canvas_w - 1, canvas_h - 1], outline=border_col, width=2)

    # 상단 레이블
    tag = f"[{d.upper()}]  cx={row['cx']:.2f}  area={row['area']:.4f}  phase={row.get('phase','?'):.2f}"
    draw.text((PAD + 2, 6), tag, fill=col, font=font_md)

    # 하단 레이블 (conf)
    conf_txt = (f"orig={row['conf_orig']:+.4f}  →  mask={row['conf_mask']:+.4f}"
                f"  drop={row['conf_drop']:+.4f}")
    flip_txt = f"  ★FLIP {row['pred_orig']}→{row['pred_mask']}" if flipped else "  stable"
    draw.text((PAD + 2, H + LABEL_H + PAD + 4), conf_txt + flip_txt,
              fill=(239, 68, 68) if flipped else (148, 163, 184), font=font_sm)

    # 중앙 구분선
    draw.line([(W + PAD, PAD), (W + PAD, H + LABEL_H + PAD)], fill=(51, 65, 85), width=PAD)

    # 열 제목
    draw.text((PAD + W // 2 - 20, H + LABEL_H + PAD - 14), "original", fill=(148, 163, 184), font=font_sm)
    draw.text((W + PAD * 2 + W // 2 - 20, H + LABEL_H + PAD - 14), "masked",   fill=(148, 163, 184), font=font_sm)

    fname = f"{d}_{idx:02d}_{'FLIP' if flipped else 'stable'}.png"
    canvas.save(out_dir / fname)
    return fname


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-images", action="store_true", default=True,
                        help="before/after 이미지 저장 (기본 ON)")
    parser.add_argument("--no-save-images", dest="save_images", action="store_false")
    args = parser.parse_args()

    if args.save_images:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    processor, vm, image_proj, anchor_feats = load_model(device)

    data = json.loads(DATA_PATH.read_text())

    # 방향별로 area_det 충분한 프레임 수집 (초기+중기 프레임만 — 도착 직전 제외)
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
    # 각 스케일별 통계를 관리하기 위한 딕셔너리 정의
    SCALES = [1.5, 1.2, 1.0, 0.8, 0.5]
    scale_dir_stats = {sc: defaultdict(lambda: {"conf_drop": [], "flip": 0, "total": 0}) for sc in SCALES}
    
    saved_images = defaultdict(list)
    dir_counters = defaultdict(int)

    print(f"\n[MASK] basket 마스킹 ablation 시작 (MIN_AREA={MIN_AREA})\n")
    print(f"  {'방향':<8} {'cx':>5} {'area':>6} {'phase':>6} {'conf_orig':>10} | {'conf_drop (by scale)':<40} | {'pred변화 (by scale)':<30}")
    print("  " + "-" * 115)

    for direction in DIRS:
        samples = dir_samples[direction]
        if not samples:
            print(f"  [{direction}] 유효 샘플 없음 (area_det >= {MIN_AREA} 없음)")
            continue

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

            # 원본 이미지의 baseline confidence 측정
            conf_orig, pred_orig = get_conf(vm, image_proj, processor, anchor_feats, img, device, gt_idx)

            # 다중 스케일 스윕 수행
            scale_results = {}
            for sc in SCALES:
                masked_img, bbox_px = mask_basket(img, cx, cy, area, scale=sc)
                conf_mask, pred_mask = get_conf(vm, image_proj, processor, anchor_feats, masked_img, device, gt_idx)
                
                drop = conf_orig - conf_mask
                flipped = (pred_orig != pred_mask)
                
                scale_results[sc] = {
                    "conf_mask": round(conf_mask, 4),
                    "conf_drop": round(drop, 4),
                    "pred_mask": DIRS[pred_mask],
                    "flipped": flipped
                }
                
                # 통계 누적
                scale_dir_stats[sc][direction]["conf_drop"].append(drop)
                scale_dir_stats[sc][direction]["total"] += 1
                if flipped:
                    scale_dir_stats[sc][direction]["flip"] += 1

            row = {
                "direction": direction,
                "cx": round(cx, 3), "cy": round(cy, 3), "area": round(area, 4),
                "phase": round(phase, 3),
                "conf_orig": round(conf_orig, 4),
                "pred_orig": DIRS[pred_orig],
                "scales": {str(sc): scale_results[sc] for sc in SCALES}
            }

            # 이미지 저장: 사용자가 요청한 '마스크 크기 축소'를 반영하여 scale=1.0을 기본으로 하여 기존 갤러리 이미지들을 덮어씀
            if args.save_images:
                dir_counters[direction] += 1
                # scale=1.0에 해당하는 마스킹 이미지를 구함
                target_masked_img, _ = mask_basket(img, cx, cy, area, scale=1.0)
                # save_pair에서 정보를 텍스트로 그릴 수 있도록 row 형식을 간이 매칭
                img_row = {
                    "direction": direction,
                    "cx": round(cx, 3), "area": round(area, 4), "phase": round(phase, 3),
                    "conf_orig": round(conf_orig, 4),
                    "conf_mask": scale_results[1.0]["conf_mask"],
                    "conf_drop": scale_results[1.0]["conf_drop"],
                    "pred_orig": DIRS[pred_orig],
                    "pred_mask": scale_results[1.0]["pred_mask"],
                    "flipped": scale_results[1.0]["flipped"]
                }
                fname = save_pair(img, target_masked_img, img_row, dir_counters[direction], OUT_DIR)
                row["img_file"] = fname
                saved_images[direction].append(fname)

            results.append(row)
            
            # 스케일별 conf_drop 값 문자열 조립
            drop_strs = "/".join([f"{scale_results[sc]['conf_drop']:+.2f}" for sc in SCALES])
            # 스케일별 pred 변동 여부 (flipped 시 스케일값 표시)
            flip_list = []
            for sc in SCALES:
                if scale_results[sc]["flipped"]:
                    flip_list.append(f"{sc:.1f}")
            flip_str = ",".join(flip_list) if flip_list else "stable"
            
            print(
                f"  {direction:<8} {cx:>5.2f} {area:>6.4f} {phase:>6.2f} "
                f"{conf_orig:>10.4f} | {drop_strs:<40} | {flip_str:<30}"
            )

        print()

    # 최종 결과 요약 표 출력 (스케일별 비교 테이블)
    print(f"\n{'='*95}")
    print(f"  Track 3: Basket Masking Ablation 다중 스케일 스윕 결과")
    print(f"{'='*95}")
    
    for sc in SCALES:
        print(f"\n  [Mask Scale = {sc:.1f}x]")
        print(f"  {'방향':<8} {'n':>4} {'conf_drop 평균':>14} {'flip 비율':>12}  {'판정':>20}")
        print("  " + "-" * 62)
        
        all_drops = []
        for d in DIRS:
            s = scale_dir_stats[sc][d]
            if not s["conf_drop"]:
                print(f"  {d:<8}   -          N/A             N/A")
                continue
            drops = s["conf_drop"]
            mean_drop = np.mean(drops)
            flip_rate = s["flip"] / s["total"] * 100
            all_drops.extend(drops)

            if mean_drop >= 0.10:
                v = "basket 의존 ✅"
            elif mean_drop >= 0.03:
                v = "부분 의존"
            else:
                v = "독립적 ⚠️"
            print(f"  {d:<8} {s['total']:>4} {mean_drop:>+14.4f} {flip_rate:>11.1f}%  {v}")
            
        if all_drops:
            overall_mean = np.mean(all_drops)
            overall_flip = sum(scale_dir_stats[sc][d]["flip"] for d in DIRS)
            overall_total = sum(scale_dir_stats[sc][d]["total"] for d in DIRS)
            flip_pct = overall_flip / overall_total * 100 if overall_total > 0 else 0
            print(f"  --------------------------------------------------------------")
            print(f"  전체평균:  {overall_total:>4} {overall_mean:>+14.4f} {flip_pct:>11.1f}%")

    print(f"{'='*95}")

    # JSON 결과 저장
    if args.save_images:
        summary = {
            "episode_phase_max": EPISODE_PHASE_MAX,
            "min_area": MIN_AREA,
            "scales_evaluated": SCALES,
            "n_sample": N_SAMPLE,
            "scale_stats": {},
            "rows": results,
        }
        for sc in SCALES:
            summary["scale_stats"][str(sc)] = {
                "per_direction": {},
                "overall": {}
            }
            all_drops_sc = []
            overall_flip_sc = 0
            overall_total_sc = 0
            
            for d in DIRS:
                s = scale_dir_stats[sc][d]
                if not s["conf_drop"]:
                    continue
                mean_drop = float(np.mean(s["conf_drop"]))
                flip_pct = s["flip"] / s["total"] * 100
                all_drops_sc.extend(s["conf_drop"])
                overall_flip_sc += s["flip"]
                overall_total_sc += s["total"]
                
                summary["scale_stats"][str(sc)]["per_direction"][d] = {
                    "n":          s["total"],
                    "mean_drop":  round(mean_drop, 4),
                    "flip":       s["flip"],
                    "flip_pct":   round(flip_pct, 1),
                }
            if all_drops_sc:
                overall_mean_sc = float(np.mean(all_drops_sc))
                flip_pct_sc = overall_flip_sc / overall_total_sc * 100
                if overall_mean_sc >= 0.10:
                    verdict_sc = "basket 영역에 유의미하게 의존 ✅"
                elif overall_mean_sc >= 0.03:
                    verdict_sc = "약한 의존 (보조 신호로 사용)"
                else:
                    verdict_sc = "basket 외 정보만으로 분류 ⚠️"
                    
                summary["scale_stats"][str(sc)]["overall"] = {
                    "mean_drop": round(overall_mean_sc, 4),
                    "flip":      overall_flip_sc,
                    "total":     overall_total_sc,
                    "flip_pct":  round(flip_pct_sc, 1),
                    "verdict":   verdict_sc
                }
                
        result_json = OUT_DIR / "results_multiscale.json"
        result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"\n[SAVED] 이미지 (scale=1.0) → {OUT_DIR}/")
        print(f"[SAVED] JSON (multiscale)  → {result_json}")


if __name__ == "__main__":
    main()
