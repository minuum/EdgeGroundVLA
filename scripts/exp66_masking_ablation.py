#!/usr/bin/env python3
"""
Exp66 Masking Ablation — Stage2 v2 (SOTA)

basket 영역을 gray로 가리면 Exp66의 action prediction이 바뀌는가?
bbox history = zeros 로 고정 → image 경로만 격리하여 인과성 증명

Usage:
  .venv/bin/python3 scripts/exp66_masking_ablation.py
"""
import json, sys, warnings
from pathlib import Path
from collections import defaultdict
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────
VLM_PATH     = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT  = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT  = ROOT / "runs" / "v5_nav" / "mlp" / "exp66" / "action_mlp.pt"
DATA_PATH    = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"

OUT_DIR_VIZ  = ROOT / "docs" / "v5" / "exp66_masking_viz"
OUT_PNG_EXP  = ROOT / "docs" / "v5" / "exp54_viz" / "masking_comparison.png"
OUT_PNG_PORT = ROOT / "docs" / "v5" / "portfolio" / "masking_comparison.png"

OUT_DIR_VIZ.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────
VIS_DIM    = 1024
PROJ_DIM   = 256
BBOX_DIM   = 32   # 8 frames × 4 features
N_CLASSES  = 8
MASK_SCALE = 1.6   # bbox 크기 배수로 마스킹 (약간 넉넉하게)
MASK_COLOR = (128, 128, 128)
N_SAMPLE   = 8    # 방향별 샘플 수
MIN_AREA   = 0.04  # basket이 충분히 보여야 마스킹 의미 있음
MIN_AREA_RIGHT = 0.002  # right 경로는 basket이 작음 (낮은 임계값)
MAX_AREA   = 0.80  # 너무 가까우면 제외 (STOP 프레임)

ACTION_NAMES = ["STOP", "FWD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
ACTION_COLORS = [
    "#ef4444",  # STOP: red
    "#4ade80",  # FWD: green
    "#60a5fa",  # LEFT: blue
    "#f59e0b",  # RIGHT: orange
    "#a78bfa",  # FWD+L: violet
    "#fb923c",  # FWD+R: orange-light
    "#38bdf8",  # ROT_L: sky
    "#818cf8",  # ROT_R: indigo
]
DIR_LABELS = {"left": "←LEFT", "center": "CENTER", "right": "RIGHT→"}


# ── Model Loading ───────────────────────────────────────────────────────
def load_models(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("[Stage1] Loading Kosmos-2 + image_proj ...")
    s1 = torch.load(str(STAGE1_CKPT), map_location=device, weights_only=False)
    print(f"  Stage1 val_acc={s1['val_acc']:.4f}")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm = base.vision_model.to(device).eval()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM, bias=True).to(device)
    image_proj.load_state_dict(s1["image_proj"])
    image_proj.eval()

    print("[Stage2] Loading ActionMLP ...")
    s2 = torch.load(str(STAGE2_CKPT), map_location=device, weights_only=False)
    print(f"  Stage2 val_acc={s2['val_acc']:.4f}")

    class ActionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(PROJ_DIM + BBOX_DIM, 256), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, N_CLASSES),
            )
        def forward(self, x):
            return self.net(x)

    mlp = ActionMLP().to(device)
    mlp.load_state_dict(s2["mlp"])
    mlp.eval()

    return processor, vm, image_proj, mlp


@torch.no_grad()
def predict(vm, image_proj, mlp, processor, img_pil, device):
    """Image만 사용 (bbox_history = zeros)"""
    inputs = processor(images=[img_pil], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv)
    vis = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(image_proj(vis), dim=-1)  # (1, 256)

    bbox_zeros = torch.zeros(1, BBOX_DIM, device=device)
    feat = torch.cat([proj, bbox_zeros], dim=-1)  # (1, 288)
    logits = mlp(feat)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    pred = int(probs.argmax())
    return pred, probs


def mask_basket(img_pil, cx, cy, area, scale=MASK_SCALE):
    """bbox 중심·면적 기준으로 basket 영역을 gray로 마스킹"""
    W, H = img_pil.size
    side = (area ** 0.5) * scale
    x0 = int((cx - side / 2) * W)
    y0 = int((cy - side / 2) * H)
    x1 = int((cx + side / 2) * W)
    y1 = int((cy + side / 2) * H)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x1), min(H, y1)
    masked = img_pil.copy()
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x0, y0, x1, y1], fill=MASK_COLOR)
    return masked, (x0, y0, x1, y1)


def load_frame_image(ep_path, frame_idx):
    with h5py.File(ep_path, "r") as f:
        raw = f["observations"]["images"][frame_idx]
    if isinstance(raw, bytes):
        import io
        return Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(raw)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return Image.fromarray(arr).convert("RGB")


# ── Sampling ────────────────────────────────────────────────────────────
def collect_samples(data, n_per_dir=N_SAMPLE):
    random.seed(42)
    by_dir = defaultdict(list)
    for ep_info in data:
        direction = ep_info.get("direction")
        ep_path = ep_info["episode"]
        if not Path(ep_path).exists():
            continue
        min_a = MIN_AREA_RIGHT if direction == "right" else MIN_AREA
        for fr in ep_info["frames"]:
            if not fr.get("consistent"):
                continue
            area = fr.get("area_det", 0)
            if area < min_a or area > MAX_AREA:
                continue
            by_dir[direction].append({
                "ep_path": ep_path,
                "frame_idx": fr["frame_idx"],
                "cx": fr["cx_det"],
                "cy": fr["cy_det"],
                "area": area,
                "gt_class": fr["gt_class"],
                "direction": direction,
            })
    samples = []
    for d in ["left", "center", "right"]:
        pool = by_dir.get(d, [])
        # center: 큰 area 우선 (마스킹 효과 극대화)
        if d == "center":
            pool = sorted(pool, key=lambda x: x["area"], reverse=True)
        else:
            random.shuffle(pool)
        samples.extend(pool[:n_per_dir])
    return samples


# ── Visualization ────────────────────────────────────────────────────────
CELL_W, CELL_H = 224, 168
PAD = 10
LABEL_H = 36
BG = (15, 23, 42)
FG = (226, 232, 240)
GREEN = (74, 222, 128)
RED = (239, 68, 68)
YELLOW = (251, 191, 36)


def hex2rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_prob_bar(draw, x, y, w, h, probs, pred):
    bar_w = w // N_CLASSES
    for i, p in enumerate(probs):
        bh = int(p * h)
        col = hex2rgb(ACTION_COLORS[i])
        draw.rectangle([x + i*bar_w, y + h - bh, x + (i+1)*bar_w, y + h], fill=col)
    # top action label
    draw.rectangle([x + pred*bar_w, y + h - int(probs[pred]*h) - 4,
                    x + (pred+1)*bar_w, y + h - int(probs[pred]*h)], fill=YELLOW)


def draw_action_badge(draw, x, y, action_idx, font):
    col = hex2rgb(ACTION_COLORS[action_idx])
    name = ACTION_NAMES[action_idx]
    w = len(name) * 7 + 8
    draw.rectangle([x, y, x + w, y + 14], fill=col)
    draw.text((x + 4, y + 2), name, fill=(15, 23, 42), font=font)
    return x + w


def make_grid(samples_results):
    n = len(samples_results)
    CELL_PAD = 6
    COL_CELL_W = CELL_W * 2 + CELL_PAD + 18  # orig + arrow gap + mask
    cols = 3
    rows = (n + cols - 1) // cols
    STAT_H = 72
    W = cols * (COL_CELL_W + PAD) + PAD
    H = STAT_H + rows * (CELL_H + LABEL_H + PAD) + PAD
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    try:
        font_md = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
        font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
        font_xs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 9)
    except Exception:
        font_md = ImageFont.load_default()
        font_sm = font_md
        font_xs = font_md

    # ── Stats bar ──────────────────────────────────────────────────────
    flips = sum(1 for r in samples_results if r["changed"])
    total = len(samples_results)
    flip_pct = 100 * flips / max(total, 1)

    # center-only stats
    center_res = [r for r in samples_results if r["direction"] == "center"]
    c_flip = sum(1 for r in center_res if r["changed"])
    c_tot = len(center_res)

    draw.rectangle([0, 0, W, STAT_H], fill=(15, 23, 42))
    draw.rectangle([0, STAT_H - 1, W, STAT_H], fill=(30, 41, 59))

    draw.text((PAD, 8), "Exp66 Stage2 v2 — Basket Masking Ablation",
              fill=FG, font=font_sm)
    draw.text((PAD, 24),
              "bbox history = zeros (image-only path)  |  basket region masked gray (×1.6 bbox)",
              fill=(100, 116, 139), font=font_xs)

    # Flip rate badges
    bx = PAD
    draw.text((bx, 40), f"Action Flip: {flips}/{total} ({flip_pct:.0f}%)",
              fill=(74, 222, 128) if flip_pct >= 50 else YELLOW, font=font_md)
    bx2 = PAD + 240
    draw.text((bx2, 40), f"Center: {c_flip}/{c_tot} ({100*c_flip/max(c_tot,1):.0f}%)",
              fill=(74, 222, 128), font=font_md)
    bx3 = PAD + 420
    draw.text((bx3, 40), "val_acc 93.5%  CL 96.6%  FPE 0.102m",
              fill=(100, 116, 139), font=font_xs)

    # ── Cell grid ──────────────────────────────────────────────────────
    for idx, r in enumerate(samples_results):
        col_i = idx % cols
        row_i = idx // cols
        ox = PAD + col_i * (COL_CELL_W + PAD)
        oy = STAT_H + PAD + row_i * (CELL_H + LABEL_H + PAD)

        pred_o = r["pred_orig"]
        pred_m = r["pred_mask"]
        changed = r["changed"]
        border_col = RED if changed else (30, 41, 59)

        # Original image
        orig_resized = r["orig_img"].resize((CELL_W, CELL_H))
        canvas.paste(orig_resized, (ox, oy))
        draw.rectangle([ox, oy, ox + CELL_W, oy + CELL_H], outline=(51, 65, 85), width=1)

        # Arrow region
        ax = ox + CELL_W + 1
        draw.rectangle([ax, oy, ax + 16, oy + CELL_H], fill=(15, 23, 42))
        arrow_col = RED if changed else (74, 222, 128)
        mid_y = oy + CELL_H // 2
        draw.text((ax + 1, mid_y - 7), "→", fill=arrow_col, font=font_md)

        # Masked image
        mx = ax + 17
        mask_resized = r["mask_img"].resize((CELL_W, CELL_H))
        canvas.paste(mask_resized, (mx, oy))
        draw.rectangle([mx, oy, mx + CELL_W, oy + CELL_H], outline=border_col, width=2)

        # Label row
        ly = oy + CELL_H + 2
        dir_lbl = DIR_LABELS.get(r["direction"], r["direction"])
        draw.text((ox, ly), dir_lbl, fill=(148, 163, 184), font=font_xs)
        draw.text((ox, ly + 11), f"area={r['area']:.2f}", fill=(71, 85, 105), font=font_xs)

        # Action badges
        badge_x = ox + 60
        badge_x = draw_action_badge(draw, badge_x, ly, pred_o, font_xs)
        draw.text((badge_x + 2, ly), "→", fill=(100, 116, 139), font=font_xs)
        badge_x2 = badge_x + 14
        draw_action_badge(draw, badge_x2, ly, pred_m, font_xs)

        # FLIP tag
        if changed:
            fx = mx - 1
            fy = oy
            tag_w = 38
            draw.rectangle([fx, fy, fx + tag_w, fy + 16], fill=RED)
            draw.text((fx + 3, fy + 2), "FLIP", fill=(255, 255, 255), font=font_xs)

    return canvas


# ── Main ────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor, vm, image_proj, mlp = load_models(device)

    with open(DATA_PATH) as f:
        data = json.load(f)

    samples = collect_samples(data, N_SAMPLE)
    print(f"Collected {len(samples)} samples")

    results = []
    flip_count = 0
    for s in samples:
        try:
            orig_img = load_frame_image(s["ep_path"], s["frame_idx"])
        except Exception as e:
            print(f"  Skip {s['ep_path']} frame {s['frame_idx']}: {e}")
            continue

        pred_o, probs_o = predict(vm, image_proj, mlp, processor, orig_img, device)
        mask_img, bbox_px = mask_basket(orig_img, s["cx"], s["cy"], s["area"])
        pred_m, probs_m = predict(vm, image_proj, mlp, processor, mask_img, device)

        changed = pred_o != pred_m
        if changed:
            flip_count += 1

        results.append({
            "orig_img": orig_img,
            "mask_img": mask_img,
            "pred_orig": pred_o,
            "pred_mask": pred_m,
            "probs_orig": probs_o,
            "probs_mask": probs_m,
            "direction": s["direction"],
            "area": s["area"],
            "changed": changed,
        })
        status = "FLIP ✓" if changed else "same "
        print(f"  [{s['direction']:6s}] area={s['area']:.2f}  "
              f"{ACTION_NAMES[pred_o]:8s} → {ACTION_NAMES[pred_m]:8s}  {status}")

    print(f"\n── Results ──────────────────────────────")
    print(f"Total samples : {len(results)}")
    print(f"Flipped       : {flip_count} / {len(results)}  ({100*flip_count/max(len(results),1):.1f}%)")

    if not results:
        print("No results — exiting")
        return

    grid = make_grid(results)
    grid.save(str(OUT_DIR_VIZ / "masking_comparison_exp66.png"))
    grid.save(str(OUT_PNG_EXP))
    grid.save(str(OUT_PNG_PORT))
    print(f"\nSaved:")
    print(f"  {OUT_DIR_VIZ}/masking_comparison_exp66.png")
    print(f"  {OUT_PNG_EXP}")
    print(f"  {OUT_PNG_PORT}")


if __name__ == "__main__":
    main()
