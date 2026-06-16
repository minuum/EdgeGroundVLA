#!/usr/bin/env python3
"""
실험 B v2: Stage 1 v2 (frame-level label) 모델 Attention Map 시각화

v1과의 차이:
  - LoRA 없음 (frozen base Kosmos-2 vision)
  - stage1_v2_projs.pt 사용 (image_proj + text_proj)
  - bbox 위치: cx_det/cy_det/area_det (HSV 탐지 or Kosmos-2 cx)
  - 데이터: bbox_dataset_frame_level.json (consistent=True)

목적:
  v2 모델이 basket 위치를 학습했다면 → 어텐션이 basket cx 쪽에 집중돼야
  "basket을 보는가 vs 복도 패턴을 기억하는가"

출력:
  docs/v5/exp54_attention_v2/
    ├── left_early.png, ..., right_late.png (9장)
    ├── grid_summary.png
    └── attention_stats_v2.json

Usage:
  .venv/bin/python3 scripts/exp54_exp_b_v2_attention.py
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
CKPT_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
OUT_DIR   = ROOT / "docs" / "v5" / "exp54_attention_v2"

PROJ_DIM  = 256
LM_DIM    = 2048
VIS_DIM   = 1024
PATCH_GRID = 16
DIR_IDX   = {"left": 0, "center": 1, "right": 2}


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
def get_attention_and_pred(vm, image_proj, processor, anchor_feats, img, device):
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)

    out = vm(pixel_values=pv, output_attentions=True)

    # last layer CLS → patches attention
    attn = out.attentions[-1].float()   # (1, heads, 257, 257)
    cls_attn = attn[0, :, 0, 1:]       # (heads, 256)
    cls_attn = cls_attn.mean(0)
    cls_attn = cls_attn / (cls_attn.sum() + 1e-8)
    attn_map = cls_attn.reshape(PATCH_GRID, PATCH_GRID).cpu().numpy()

    feat = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(image_proj(feat), dim=-1)
    sims = (proj @ anchor_feats.T)[0]
    pred_idx = sims.argmax().item()
    dirs = ["left", "center", "right"]
    return attn_map, dirs[pred_idx], sims[pred_idx].item()


def bbox_attention_ratio(attn_map, cx, cy, area, threshold_top=0.3):
    side = max(1, int(np.sqrt(area) * PATCH_GRID))
    half = side // 2
    bx = int(cx * PATCH_GRID)
    by = int(cy * PATCH_GRID)
    x1, x2 = max(0, bx - half), min(PATCH_GRID, bx + half + 1)
    y1, y2 = max(0, by - half), min(PATCH_GRID, by + half + 1)

    flat = attn_map.flatten()
    k = max(1, int(len(flat) * threshold_top))
    top_k_mask = np.zeros(PATCH_GRID * PATCH_GRID, dtype=bool)
    top_k_mask[np.argsort(flat)[-k:]] = True
    top_k_mask = top_k_mask.reshape(PATCH_GRID, PATCH_GRID)

    bbox_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    return (top_k_mask & bbox_mask).sum() / k


def make_overlay(img_pil, attn_map, cx=None, cy=None, area=None):
    img_np = np.array(img_pil.resize((224, 224)))
    H, W = img_np.shape[:2]

    attn_up = np.array(
        Image.fromarray((attn_map / attn_map.max() * 255).astype(np.uint8)).resize(
            (W, H), Image.BICUBIC
        )
    ).astype(float)
    attn_up = attn_up / attn_up.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")
    if cx is not None and area is not None and area > 0:
        bx = int(cx * W)
        by = int(cy * H)
        side = int(np.sqrt(area) * W)
        half = side // 2
        rect = mpatches.Rectangle(
            (bx - half, by - half), side, side,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(bx - half, by - half - 4, "basket cx", color="lime", fontsize=8)

    axes[1].imshow(attn_up, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Attention (CLS→patches)", fontsize=11)
    axes[1].axis("off")

    cmap = plt.get_cmap("jet")
    overlay = (img_np / 255.0) * 0.45 + cmap(attn_up)[:, :, :3] * 0.55
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")
    if cx is not None and area is not None and area > 0:
        bx = int(cx * W)
        by = int(cy * H)
        side = int(np.sqrt(area) * W)
        half = side // 2
        rect2 = mpatches.Rectangle(
            (bx - half, by - half), side, side,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[2].add_patch(rect2)

    plt.tight_layout()
    return fig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processor, vm, image_proj, anchor_feats = load_model(device)

    data = json.loads(DATA_PATH.read_text())

    # consistent=True 프레임만 사용, 방향별 분류
    dir_eps = defaultdict(list)
    for ep in data:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"]]
        if frames:
            dir_eps[ep["direction"]].append({
                "episode":   ep["episode"],
                "direction": ep["direction"],
                "frames":    frames,
            })

    stats = []
    grid_imgs = {}

    print("\n각 방향 × 구간별 attention map 생성 중...\n")

    for direction in ["left", "center", "right"]:
        eps = dir_eps[direction]
        seg_samples = {"early": None, "mid": None, "late": None}

        for ep in eps:
            n = len(ep["frames"])
            if n < 3:
                continue
            e_cut = max(1, n // 3)
            l_cut = n - max(1, n // 3)
            candidates = {
                "early": ep["frames"][:e_cut],
                "mid":   ep["frames"][e_cut:l_cut],
                "late":  ep["frames"][l_cut:],
            }
            for seg, seg_frames in candidates.items():
                if seg_samples[seg] is None and seg_frames:
                    mid_fr = seg_frames[len(seg_frames) // 2]
                    seg_samples[seg] = (ep, mid_fr)
            if all(v is not None for v in seg_samples.values()):
                break

        for seg, sample in seg_samples.items():
            if sample is None:
                print(f"  [{direction}/{seg}] 샘플 없음, 스킵")
                continue

            ep, fr = sample
            try:
                with h5py.File(ep["episode"], "r") as f:
                    img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]])
            except Exception as e:
                print(f"  [{direction}/{seg}] 이미지 로드 실패: {e}")
                continue

            attn_map, pred_dir, conf = get_attention_and_pred(
                vm, image_proj, processor, anchor_feats, img, device
            )

            cx   = fr.get("cx_det")
            cy   = fr.get("cy_det")
            area = fr.get("area_det")
            gt   = fr.get("label")

            bbox_ratio = None
            if cx is not None and area is not None and area > 0:
                bbox_ratio = bbox_attention_ratio(attn_map, cx, cy, area)

            correct = (pred_dir == gt)
            print(
                f"  [{direction}/{seg:5s}] gt={gt:<7} pred={pred_dir:<7} {'✅' if correct else '❌'}  "
                f"conf={conf:.3f}  "
                + (f"bbox_attn={bbox_ratio:.3f}" if bbox_ratio is not None else "no_cx")
            )

            stats.append({
                "direction": direction, "seg": seg,
                "gt": gt, "pred_dir": pred_dir, "correct": correct,
                "conf": float(conf),
                "cx_det": cx, "area_det": area,
                "bbox_ratio": float(bbox_ratio) if bbox_ratio is not None else None,
            })

            fig = make_overlay(img, attn_map, cx, cy, area)
            title = f"{direction.upper()} / {seg}  |  gt: {gt}  pred: {pred_dir} {'✅' if correct else '❌'}"
            fig.suptitle(title, fontsize=13, fontweight="bold")
            out_path = OUT_DIR / f"{direction}_{seg}.png"
            fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
            plt.close(fig)
            grid_imgs[(direction, seg)] = str(out_path)

    # 3×3 그리드
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
    fig2.suptitle("Stage 1 v2 Attention Maps: direction × episode position", fontsize=15, fontweight="bold")
    for r, seg in enumerate(["early", "mid", "late"]):
        for c, direction in enumerate(["left", "center", "right"]):
            ax = axes2[r][c]
            key = (direction, seg)
            if key in grid_imgs:
                ax.imshow(np.array(Image.open(grid_imgs[key])))
            ax.set_title(f"{direction} / {seg}", fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    grid_path = OUT_DIR / "grid_summary.png"
    fig2.savefig(str(grid_path), dpi=80, bbox_inches="tight")
    plt.close(fig2)
    print(f"\n[SAVED] {grid_path}")

    print(f"\n{'='*60}")
    print("  실험 B v2 요약")
    print(f"{'='*60}")
    print(f"\n  {'방향':<8} {'구간':<8} {'bbox_attn':>10}  {'gt':>7}  {'pred':>7}  {'정답':>5}")
    print("  " + "-" * 55)
    for s in stats:
        r = f"{s['bbox_ratio']:.3f}" if s["bbox_ratio"] is not None else "  N/A"
        print(f"  {s['direction']:<8} {s['seg']:<8} {r:>10}  {(s['gt'] or '?'):>7}  {s['pred_dir']:>7}  {'✅' if s['correct'] else '❌':>5}")

    has_bbox = [s for s in stats if s["bbox_ratio"] is not None]
    if has_bbox:
        avg_ratio = np.mean([s["bbox_ratio"] for s in has_bbox])
        avg_area  = np.mean([s["area_det"] for s in has_bbox if s["area_det"] and s["area_det"] > 0])
        rv_random = avg_ratio / (avg_area + 1e-8)
        print(f"\n  bbox_ratio 평균: {avg_ratio:.3f}")
        print(f"  random 기대값 (bbox 면적 비율): {avg_area:.3f}")
        print(f"  주목 집중도: {rv_random:.2f}×")
        if rv_random > 2.0:
            verdict = "basket 영역 집중 ✅ (2× 이상)"
        elif rv_random > 1.3:
            verdict = "약한 집중 (1.3~2×)"
        else:
            verdict = "복도 전체 분산 ⚠️ (1× 이하)"
        print(f"  → 판정: {verdict}")

    json.dump(stats, open(str(OUT_DIR / "attention_stats_v2.json"), "w"), indent=2)
    print(f"\n  이미지: {OUT_DIR}/")
    print(f"  통계:   {OUT_DIR}/attention_stats_v2.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
