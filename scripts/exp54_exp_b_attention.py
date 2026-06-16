#!/usr/bin/env python3
"""
실험 B: Stage 1 CLIP LoRA Attention Map 시각화

목적:
  Stage 1 모델이 이미지의 "어느 픽셀"을 보고 방향을 판단하는지 확인
  - basket 영역에 집중 → "basket을 본다" ✓
  - 복도 전체 / 원근감 영역 → "복도를 외웠다"

방법:
  - ViT 마지막 레이어 CLS → patch attention (16×16) 추출
  - 이미지 위에 heatmap overlay 저장
  - bbox 영역 내 attention 비율 정량화

출력:
  docs/v5/exp54_attention/
    ├── left_early.png   (left/center/right × early/mid/late = 9장)
    ├── grid_summary.png (3×3 그리드)
    └── attention_stats.json

Usage:
  .venv/bin/python3 scripts/exp54_exp_b_attention.py
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1"
CKPT_PATH  = STAGE1_DIR / "stage1_projs.pt"
LORA_DIR   = STAGE1_DIR / "clip_lora_adapter"
OUT_DIR    = ROOT / "docs" / "v5" / "exp54_attention"

PATH_TO_DIR = {
    "left_straight":"left",  "left_left":"left",   "left_right":"left",
    "center_straight":"center","center_left":"center","center_right":"center",
    "right_straight":"right", "right_left":"right", "right_right":"right",
}
PATCH_GRID = 16  # 224/14 = 16


# ─────────────────────────────────────────────────────────
# 모델 로드
# ─────────────────────────────────────────────────────────

def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel
    import torch.nn as nn

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    )
    vm = PeftModel.from_pretrained(base.vision_model, str(LORA_DIR)).to(device).eval()

    # proj (Stage 1 검증용 — 방향 예측에 쓸 것)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    VIS_DIM, PROJ_DIM = 1024, 256
    LM_DIM = 2048

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    proj_state = {k[len("proj."):]: v for k, v in ckpt["clip_lora"].items() if k.startswith("proj.")}
    image_proj.load_state_dict(proj_state)
    image_proj.eval()

    text_proj = nn.Linear(LM_DIM, PROJ_DIM).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    anchor_feats = F.normalize(text_proj(ckpt["anchor_raw"].to(device)), dim=-1)  # (3, 256)

    return processor, vm, image_proj, anchor_feats


@torch.no_grad()
def get_attention_and_pred(vm, image_proj, processor, anchor_feats, img, device):
    """
    Returns:
        attn_map: (16, 16) ndarray — CLS→patches avg over heads
        pred_dir: str — 예측 방향
        conf: float — cosine similarity
    """
    inputs = processor(images=[img], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)

    out = vm(pixel_values=pv, output_attentions=True)

    # last layer attention: (1, 16 heads, 257, 257)
    attn = out.attentions[-1].float()          # (1, 16, 257, 257)
    # CLS(idx=0) → patch tokens(idx=1..256)
    cls_attn = attn[0, :, 0, 1:]              # (16, 256)
    cls_attn = cls_attn.mean(0)               # (256,) — avg over heads
    cls_attn = cls_attn / (cls_attn.sum() + 1e-8)
    attn_map = cls_attn.reshape(PATCH_GRID, PATCH_GRID).cpu().numpy()

    # 방향 예측
    feat = out.last_hidden_state.mean(dim=1).float()  # (1, 1024)
    proj = F.normalize(image_proj(feat), dim=-1)       # (1, 256)
    sims = (proj @ anchor_feats.T)[0]                  # (3,) left/center/right
    pred_idx = sims.argmax().item()
    dirs = ["left", "center", "right"]
    return attn_map, dirs[pred_idx], sims[pred_idx].item()


# ─────────────────────────────────────────────────────────
# 시각화 유틸
# ─────────────────────────────────────────────────────────

def make_overlay(img_pil, attn_map, bbox_cx=None, bbox_cy=None, bbox_area=None, has_bbox=False):
    """
    img_pil: PIL Image (H, W, 3)
    attn_map: (16, 16) float ndarray
    Returns: fig with 3 subplots (original | heatmap | overlay)
    """
    img_np = np.array(img_pil.resize((224, 224)))
    H, W = img_np.shape[:2]

    # attention map을 224×224로 upsample
    attn_up = np.array(
        Image.fromarray((attn_map / attn_map.max() * 255).astype(np.uint8)).resize(
            (W, H), Image.BICUBIC
        )
    ).astype(float)
    attn_up = attn_up / attn_up.max()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. 원본
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    # bbox 그리기
    if has_bbox and bbox_cx is not None:
        bx = int(bbox_cx * W)
        by = int(bbox_cy * H)
        area_side = int(np.sqrt(bbox_area) * W)
        half = area_side // 2
        rect = mpatches.Rectangle(
            (bx - half, by - half), area_side, area_side,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[0].add_patch(rect)
        axes[0].text(bx - half, by - half - 4, "bbox", color="lime", fontsize=8)

    # 2. Attention heatmap
    axes[1].imshow(attn_up, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Attention (CLS→patches)", fontsize=11)
    axes[1].axis("off")

    # 3. Overlay
    cmap = plt.get_cmap("jet")
    heat_rgba = cmap(attn_up)                      # (H, W, 4)
    alpha = 0.55
    overlay = (img_np / 255.0) * (1 - alpha) + heat_rgba[:, :, :3] * alpha
    overlay = np.clip(overlay, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    # bbox on overlay
    if has_bbox and bbox_cx is not None:
        bx = int(bbox_cx * W)
        by = int(bbox_cy * H)
        area_side = int(np.sqrt(bbox_area) * W)
        half = area_side // 2
        rect2 = mpatches.Rectangle(
            (bx - half, by - half), area_side, area_side,
            linewidth=2, edgecolor="lime", facecolor="none"
        )
        axes[2].add_patch(rect2)

    plt.tight_layout()
    return fig


def bbox_attention_ratio(attn_map, cx, cy, area, threshold_top=0.3):
    """
    bbox 영역 내 attention 비율 계산
    - attn_map: (16, 16)
    - cx, cy: 0~1 normalized center
    - area: 0~1 normalized area
    - threshold_top: 상위 몇 %의 attention patch를 "주목 영역"으로 볼 것인지
    Returns: fraction of top-attention in bbox region
    """
    side = int(np.sqrt(area) * PATCH_GRID)
    side = max(1, side)
    half = side // 2

    bx = int(cx * PATCH_GRID)
    by = int(cy * PATCH_GRID)
    x1, x2 = max(0, bx - half), min(PATCH_GRID, bx + half + 1)
    y1, y2 = max(0, by - half), min(PATCH_GRID, by + half + 1)

    flat = attn_map.flatten()
    k = max(1, int(len(flat) * threshold_top))
    top_k_idx = np.argsort(flat)[-k:]
    top_k_mask = np.zeros(PATCH_GRID * PATCH_GRID, dtype=bool)
    top_k_mask[top_k_idx] = True
    top_k_mask = top_k_mask.reshape(PATCH_GRID, PATCH_GRID)

    bbox_mask = np.zeros((PATCH_GRID, PATCH_GRID), dtype=bool)
    bbox_mask[y1:y2, x1:x2] = True

    bbox_top = (top_k_mask & bbox_mask).sum()
    return bbox_top / k


# ─────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    processor, vm, image_proj, anchor_feats = load_model(device)

    data = json.loads(DATA_PATH.read_text())
    ep_labels_all = [PATH_TO_DIR.get(ep["path_type"], "unk") for ep in data]
    valid_idx = [i for i, l in enumerate(ep_labels_all) if l != "unk"]
    data_valid = [data[i] for i in valid_idx]
    labels_valid = [ep_labels_all[i] for i in valid_idx]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(data_valid)), labels_valid))
    val_eps = [data_valid[i] for i in te_idx]

    # 방향별로 분류
    dir_eps = defaultdict(list)
    for ep in val_eps:
        d = PATH_TO_DIR.get(ep["path_type"])
        if d:
            dir_eps[d].append(ep)

    stats = []  # bbox attention 통계
    grid_imgs = {}  # (dir, seg) → PIL image

    print("\n각 방향 × 구간별 attention map 생성 중...\n")

    for direction in ["left", "center", "right"]:
        eps = dir_eps[direction]
        # 에피소드 1개에서 early/mid/late 프레임 1장씩 추출
        # 여러 에피소드에서 각 구간을 하나씩 가져옴 (총 3장)
        seg_samples = {"early": None, "mid": None, "late": None}

        for ep in eps:
            frames = ep["frames"]
            n = len(frames)
            if n < 3:
                continue
            e_cut = max(1, n // 3)
            l_cut = n - max(1, n // 3)
            candidates = {
                "early": frames[:e_cut],
                "mid":   frames[e_cut:l_cut],
                "late":  frames[l_cut:],
            }
            for seg, seg_frames in candidates.items():
                if seg_samples[seg] is None and seg_frames:
                    seg_samples[seg] = (ep, seg_frames[len(seg_frames) // 2])
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

            # bbox 정보
            has_bbox = fr.get("has_bbox", False)
            cx   = fr.get("cx", 0.5)
            cy   = fr.get("cy", 0.5)
            area = fr.get("area", 0.0)

            # bbox attention 비율
            bbox_ratio = None
            if has_bbox and area > 0:
                bbox_ratio = bbox_attention_ratio(attn_map, cx, cy, area)

            correct = (pred_dir == direction)
            print(
                f"  [{direction}/{seg:5s}] pred={pred_dir:<7} {'✅' if correct else '❌'}  "
                f"conf={conf:.3f}  "
                + (f"bbox_attn={bbox_ratio:.3f}" if bbox_ratio is not None else "no_bbox")
            )

            stats.append({
                "direction": direction, "seg": seg,
                "pred_dir": pred_dir, "correct": correct,
                "conf": float(conf),
                "has_bbox": has_bbox,
                "bbox_ratio": float(bbox_ratio) if bbox_ratio is not None else None,
                "cx": cx, "area": area,
            })

            # 이미지 저장
            title = f"{direction.upper()} / {seg}  →  pred: {pred_dir} {'✅' if correct else '❌'}"
            fig = make_overlay(img, attn_map, cx, cy, area, has_bbox)
            fig.suptitle(title, fontsize=13, fontweight="bold")
            out_path = OUT_DIR / f"{direction}_{seg}.png"
            fig.savefig(str(out_path), dpi=100, bbox_inches="tight")
            plt.close(fig)

            # 그리드용 overlay 저장
            grid_imgs[(direction, seg)] = str(out_path)

    # ─── 요약 그리드 (3×3) ──────────────────────────────
    seg_order = ["early", "mid", "late"]
    dir_order = ["left", "center", "right"]
    fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
    fig2.suptitle("Stage 1 Attention Maps: direction × episode position", fontsize=15, fontweight="bold")

    for r, seg in enumerate(seg_order):
        for c, direction in enumerate(dir_order):
            key = (direction, seg)
            ax = axes2[r][c]
            if key in grid_imgs:
                img_loaded = np.array(Image.open(grid_imgs[key]))
                ax.imshow(img_loaded)
            ax.set_title(f"{direction} / {seg}", fontsize=10)
            ax.axis("off")

    plt.tight_layout()
    grid_path = OUT_DIR / "grid_summary.png"
    fig2.savefig(str(grid_path), dpi=80, bbox_inches="tight")
    plt.close(fig2)
    print(f"\n[SAVED] {grid_path}")

    # ─── 통계 요약 ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  실험 B 요약")
    print(f"{'='*60}")
    print(f"\n  bbox 내 attention 비율 (상위 30% patch 기준):")
    print(f"  {'방향':<8} {'구간':<8} {'bbox_ratio':>12}  {'pred':>8}  {'정답?':>5}")
    print("  " + "-" * 50)
    has_bbox_stats = [s for s in stats if s["bbox_ratio"] is not None]
    for s in stats:
        ratio_str = f"{s['bbox_ratio']:.3f}" if s["bbox_ratio"] is not None else "  N/A"
        print(f"  {s['direction']:<8} {s['seg']:<8} {ratio_str:>12}  {s['pred_dir']:>8}  {'✅' if s['correct'] else '❌':>5}")

    if has_bbox_stats:
        avg_ratio = np.mean([s["bbox_ratio"] for s in has_bbox_stats])
        print(f"\n  bbox_ratio 평균: {avg_ratio:.3f}")
        # 랜덤 기대값: bbox가 차지하는 patch 비율
        avg_bbox_area = np.mean([s["area"] for s in has_bbox_stats if s["area"] > 0])
        random_baseline = avg_bbox_area  # 랜덤이면 area 비율만큼 attention이 bbox에 떨어짐
        print(f"  bbox 면적 평균(랜덤 기대값): {random_baseline:.3f}")
        ratio_vs_random = avg_ratio / (random_baseline + 1e-8)
        print(f"  주목 집중도 (ratio / random): {ratio_vs_random:.2f}x")
        if ratio_vs_random > 2.0:
            verdict = "basket 영역에 유의미하게 집중 ✅ (2× 이상)"
        elif ratio_vs_random > 1.3:
            verdict = "약한 집중 (1.3~2×)"
        else:
            verdict = "복도 전체에 분산 — basket 무관 ⚠️ (1× 이하)"
        print(f"  → 판정: {verdict}")

    json.dump(stats, open(str(OUT_DIR / "attention_stats.json"), "w"), indent=2)
    print(f"\n  이미지: {OUT_DIR}/")
    print(f"  통계:   {OUT_DIR}/attention_stats.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
