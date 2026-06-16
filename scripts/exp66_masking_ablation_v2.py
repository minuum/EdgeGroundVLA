#!/usr/bin/env python3
"""
Exp66 Masking Ablation v2 — SOTA 파이프라인 그대로 재현

SOTA (Exp66) 실제 추론 순서:
  1. 프레임 → Kosmos-2 generate() "<grounding>The gray basket is at"
  2. has_bbox=True인 경우만 (basket/container entity 감지)
  3. Stage2 v2 ActionMLP 추론 (bbox history=zeros, 이미지 경로 격리)
  4. 감지된 bbox 영역 gray 마스킹
  5. 마스킹 이미지로 다시 ActionMLP 추론
  6. action 변화 측정

→ bbox_dataset_frame_level.json(사전 계산) 대신 live grounding 사용

Usage:
  .venv/bin/python3 scripts/exp66_masking_ablation_v2.py
"""
import io, json, sys, warnings, random
from pathlib import Path
from typing import Optional

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
VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp66" / "action_mlp.pt"
DATA_DIR    = ROOT / "ROS_action" / "mobile_vla_dataset_v5"

OUT_DIR     = ROOT / "docs" / "v5" / "exp66_masking_viz"
OUT_PNG_EXP = ROOT / "docs" / "v5" / "exp54_viz" / "masking_comparison.png"
OUT_PNG_PORT= ROOT / "docs" / "v5" / "portfolio" / "masking_comparison.png"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── SOTA grounding 설정 (inference_server.py와 동일) ────────────────────
GROUNDING_PROMPT     = "<grounding>The gray basket is at"
FULLSCREEN_THRESHOLD = 0.60   # inference_server 기본값
MIN_AREA             = 0.02   # basket이 너무 작으면 마스킹 효과 없음
MAX_AREA             = 0.70   # STOP 직전 도착 프레임 제외
MASK_SCALE           = 1.4
MASK_COLOR           = (128, 128, 128)

# ── Model constants ─────────────────────────────────────────────────────
VIS_DIM  = 1024
PROJ_DIM = 256
BBOX_DIM = 32
N_CLS    = 8
ACTION_NAMES  = ["STOP", "FWD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
ACTION_COLORS_HEX = [
    "#ef4444","#4ade80","#60a5fa","#f59e0b",
    "#a78bfa","#fb923c","#38bdf8","#818cf8",
]

# ── Episode sampling ────────────────────────────────────────────────────
PATH_TYPES = [
    "center_straight", "center_left", "center_right",
    "left_straight",   "left_right",
    "right_straight",  "right_left",
]
FRAMES_PER_EP = 4   # 에피소드당 테스트할 프레임 수 (초기~중기)
EPS_PER_TYPE  = 3   # path type당 에피소드 수


# ── Model loading ────────────────────────────────────────────────────────
def load_models(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("[Stage1] Kosmos-2 vision encoder + image_proj ...")
    s1 = torch.load(str(STAGE1_CKPT), map_location=device, weights_only=False)
    print(f"  val_acc={s1['val_acc']:.4f}")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base_full = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm = base_full.vision_model.to(device).eval()
    full_model = base_full.to(device).eval()  # generate() 위해 전체 모델 필요

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(s1["image_proj"])
    image_proj.eval()

    print("[Stage2] ActionMLP ...")
    s2 = torch.load(str(STAGE2_CKPT), map_location=device, weights_only=False)
    print(f"  val_acc={s2['val_acc']:.4f}")

    class ActionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(PROJ_DIM + BBOX_DIM, 256), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, N_CLS),
            )
        def forward(self, x): return self.net(x)

    mlp = ActionMLP().to(device)
    mlp.load_state_dict(s2["mlp"])
    mlp.eval()

    return processor, vm, full_model, image_proj, mlp


# ── SOTA grounding (inference_server와 동일 로직) ────────────────────────
@torch.no_grad()
def run_grounding(full_model, processor, img_pil: Image.Image, device) -> Optional[dict]:
    """Kosmos-2 generate() → basket bbox. None이면 감지 실패."""
    inputs = processor(text=GROUNDING_PROMPT, images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

    generated = full_model.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
        use_cache=True,
        max_new_tokens=64,
    )
    new_ids = generated[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
    caption, entities = processor.post_process_generation(raw)

    for entity_name, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            if max(x1, y1, x2, y2) > 1.5:
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            area = (x2 - x1) * (y2 - y1)
            if area > FULLSCREEN_THRESHOLD:
                continue
            if "basket" in entity_name.lower() or "container" in entity_name.lower():
                return {"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2, "entity": entity_name}
    return None


# ── Action inference (image-only, bbox=zeros) ────────────────────────────
@torch.no_grad()
def predict(vm, image_proj, mlp, processor, img_pil: Image.Image, device) -> tuple[int, np.ndarray]:
    inputs = processor(images=[img_pil], return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv)
    vis = out.last_hidden_state.mean(dim=1).float()
    proj = F.normalize(image_proj(vis), dim=-1)
    feat = torch.cat([proj, torch.zeros(1, BBOX_DIM, device=device)], dim=-1)
    logits = mlp(feat)
    probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    return int(probs.argmax()), probs


def mask_basket(img_pil: Image.Image, x1, y1, x2, y2, scale=MASK_SCALE) -> Image.Image:
    W, H = img_pil.size
    cx, cy = (x1+x2)/2, (y1+y2)/2
    hw = (x2-x1)*scale/2
    hh = (y2-y1)*scale/2
    px0 = max(0, int((cx-hw)*W)); py0 = max(0, int((cy-hh)*H))
    px1 = min(W, int((cx+hw)*W)); py1 = min(H, int((cy+hh)*H))
    masked = img_pil.copy()
    ImageDraw.Draw(masked).rectangle([px0, py0, px1, py1], fill=MASK_COLOR)
    return masked


def load_frame(ep_path: Path, idx: int) -> Image.Image:
    with h5py.File(str(ep_path), "r") as f:
        raw = f["observations"]["images"][idx]
    if isinstance(raw, (bytes, np.bytes_)):
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    arr = np.array(raw)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def get_ep_path_type(name: str) -> str:
    for pt in PATH_TYPES:
        if pt in name:
            return pt
    return "other"


# ── Visualization ─────────────────────────────────────────────────────────
BG = (15, 23, 42); FG = (226, 232, 240)
RED = (239, 68, 68); GREEN = (74, 222, 128); YELLOW = (251, 191, 36)
BLUE = (96, 165, 250)

def hex2rgb(h): h=h.lstrip("#"); return tuple(int(h[i:i+2],16) for i in (0,2,4))

CELL_W, CELL_H = 280, 180

def make_grid(results: list) -> Image.Image:
    cols = 3
    rows = (len(results) + cols - 1) // cols
    STAT_H = 80
    W = cols * (CELL_W * 2 + 30) + 20
    H = STAT_H + rows * (CELL_H + 48) + 10
    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    try:
        fn_b = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        fn_r = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        f14 = ImageFont.truetype(fn_b, 14)
        f11 = ImageFont.truetype(fn_b, 11)
        f9  = ImageFont.truetype(fn_r, 9)
    except Exception:
        f14 = f11 = f9 = ImageFont.load_default()

    flips   = sum(1 for r in results if r["changed"])
    total   = len(results)
    flip_pc = 100 * flips / max(total, 1)

    draw.text((12, 10), "Exp66 Masking Ablation — Live Kosmos-2 Grounding (SOTA 파이프라인)", fill=FG, font=f11)
    draw.text((12, 26), "bbox history=zeros · 이미지 경로 격리 · has_bbox=True 프레임만 사용", fill=(100,116,139), font=f9)
    pct_col = GREEN if flip_pc >= 60 else YELLOW
    draw.text((12, 44), f"Action Flip: {flips}/{total}  ({flip_pc:.0f}%)", fill=pct_col, font=f14)
    draw.text((260, 48), "Stage2 v2 val_acc 93.5% · CL 96.6% · FPE 0.102m", fill=(100,116,139), font=f9)

    for idx, r in enumerate(results):
        ci = idx % cols; ri = idx // cols
        ox = 10 + ci * (CELL_W * 2 + 30)
        oy = STAT_H + ri * (CELL_H + 48)

        # ── original ──
        orig_r = r["orig_img"].resize((CELL_W, CELL_H))
        canvas.paste(orig_r, (ox, oy))

        # grounding bbox 빨간 박스 표시
        W_, H_ = r["orig_img"].size
        bx0 = int(r["x1"] * W_ / W_ * CELL_W)  # normalize to cell
        by0 = int(r["y1"] * H_ / H_ * CELL_H)
        bx1 = int(r["x2"] * W_ / W_ * CELL_W)
        by1 = int(r["y2"] * H_ / H_ * CELL_H)
        ov = ImageDraw.Draw(canvas)
        ov.rectangle([ox+bx0, oy+by0, ox+bx1, oy+by1], outline=(255,80,80), width=2)

        # ── arrow ──
        ax = ox + CELL_W + 1
        draw.rectangle([ax, oy, ax+26, oy+CELL_H], fill=BG)
        arrow_col = RED if r["changed"] else GREEN
        draw.text((ax+4, oy+CELL_H//2-10), "→", fill=arrow_col, font=f14)

        # ── masked ──
        mx = ax + 27
        mask_r = r["mask_img"].resize((CELL_W, CELL_H))
        canvas.paste(mask_r, (mx, oy))
        border = RED if r["changed"] else (30,41,59)
        draw.rectangle([mx, oy, mx+CELL_W, oy+CELL_H], outline=border, width=2)

        if r["changed"]:
            draw.rectangle([mx, oy, mx+46, oy+18], fill=RED)
            draw.text((mx+4, oy+3), "FLIP", fill=(255,255,255), font=f9)

        # ── labels ──
        ly = oy + CELL_H + 3
        po = r["pred_orig"]; pm = r["pred_mask"]
        draw.text((ox, ly), f"{r['path_type']}  area={r['area']:.2f}  entity:\"{r['entity']}\"",
                  fill=(100,116,139), font=f9)
        ac_o = hex2rgb(ACTION_COLORS_HEX[po]); ac_m = hex2rgb(ACTION_COLORS_HEX[pm])
        draw.rectangle([ox, ly+12, ox+55, ly+24], fill=ac_o)
        draw.text((ox+3, ly+13), ACTION_NAMES[po], fill=BG, font=f9)
        draw.text((ox+59, ly+13), "→", fill=(100,116,139), font=f9)
        draw.rectangle([ox+72, ly+12, ox+127, ly+24], fill=ac_m)
        draw.text((ox+75, ly+13), ACTION_NAMES[pm], fill=BG, font=f9)

    return canvas


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    processor, vm, full_model, image_proj, mlp = load_models(device)

    # v5 에피소드 수집
    all_eps = sorted(DATA_DIR.glob("*.h5"))
    by_type: dict[str, list] = {pt: [] for pt in PATH_TYPES}
    for ep in all_eps:
        pt = get_ep_path_type(ep.name)
        if pt in by_type:
            by_type[pt].append(ep)

    random.seed(0)
    results = []
    tested = 0
    found  = 0

    for pt, eps in by_type.items():
        random.shuffle(eps)
        ep_used = 0
        for ep_path in eps:
            if ep_used >= EPS_PER_TYPE:
                break
            try:
                with h5py.File(str(ep_path), "r") as f:
                    n_frames = len(f["observations"]["images"])
            except Exception:
                continue

            # 초기~중기 프레임 (앞 60%)
            frame_pool = list(range(max(1, int(n_frames * 0.6))))
            random.shuffle(frame_pool)
            frame_found = 0

            for fidx in frame_pool:
                if frame_found >= FRAMES_PER_EP:
                    break
                tested += 1
                try:
                    img = load_frame(ep_path, fidx)
                except Exception:
                    continue

                bbox = run_grounding(full_model, processor, img, device)
                if bbox is None:
                    print(f"  [{pt}] f{fidx:2d} NO BBOX")
                    continue
                if not (MIN_AREA <= bbox["area"] <= MAX_AREA):
                    print(f"  [{pt}] f{fidx:2d} area={bbox['area']:.3f} 범위 밖")
                    continue

                # 원본 예측
                pred_o, probs_o = predict(vm, image_proj, mlp, processor, img, device)
                # 마스킹 후 예측
                masked = mask_basket(img, bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
                pred_m, probs_m = predict(vm, image_proj, mlp, processor, masked, device)

                changed = pred_o != pred_m
                found  += 1
                frame_found += 1
                if changed:
                    status = "FLIP ✓"
                else:
                    status = "same "

                print(f"  [{pt}] f{fidx:2d} area={bbox['area']:.2f} cx={bbox['cx']:.2f}"
                      f"  {ACTION_NAMES[pred_o]:8s}→{ACTION_NAMES[pred_m]:8s}  {status}"
                      f"  entity=\"{bbox['entity']}\"")

                results.append({
                    "orig_img": img, "mask_img": masked,
                    "pred_orig": pred_o, "pred_mask": pred_m,
                    "probs_orig": probs_o, "probs_mask": probs_m,
                    "changed": changed,
                    "path_type": pt,
                    "area": bbox["area"],
                    "entity": bbox["entity"],
                    "cx": bbox["cx"], "cy": bbox["cy"],
                    "x1": bbox["x1"], "y1": bbox["y1"],
                    "x2": bbox["x2"], "y2": bbox["y2"],
                })

            if frame_found > 0:
                ep_used += 1

    print(f"\n── Summary ──────────────────────────────────────────────")
    print(f"  Frames tested : {tested}")
    print(f"  has_bbox=True : {found}")
    flips = sum(1 for r in results if r["changed"])
    print(f"  Action flip   : {flips}/{found}  ({100*flips/max(found,1):.1f}%)")

    if not results:
        print("No results.")
        return

    grid = make_grid(results)
    grid.save(str(OUT_DIR / "masking_comparison_exp66_v2.png"))
    grid.save(str(OUT_PNG_EXP))
    grid.save(str(OUT_PNG_PORT))
    print(f"\nSaved → {OUT_PNG_EXP}")
    print(f"         {OUT_PNG_PORT}")


if __name__ == "__main__":
    main()
