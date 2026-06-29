#!/usr/bin/env python3
"""CH53 이미지 생성: v5_2 대표 스트립 + 6/24 세션 프레임."""
import h5py
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/ch46_50_viz"
OUT.mkdir(exist_ok=True)


def box_from_cxcyarea(cx, cy, area, W, H):
    side = area ** 0.5
    return (max(0, int((cx - side / 2) * W)), max(0, int((cy - side / 2) * H)),
            min(W, int((cx + side / 2) * W)), min(H, int((cy + side / 2) * H)))


def save_web(img, path, max_w=1100):
    if img.width > max_w:
        r = max_w / img.width
        img = img.resize((max_w, int(img.height * r)), Image.LANCZOS)
    img.save(path, quality=85)


def save_frames(ep_path, prefix, label, idxs=None, max_w=800):
    with h5py.File(ep_path, "r") as f:
        imgs_ds = f["observations/images"]
        acts = f["actions"][:]
        n = len(imgs_ds)
        if idxs is None:
            idxs = [0, n // 2, n - 1]
        saved = []
        for t in idxs:
            if t >= n:
                continue
            raw = imgs_ds[t]
            if raw.ndim == 1:
                import io
                img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
            else:
                img = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
            if img.width > max_w:
                r = max_w / img.width
                img = img.resize((max_w, int(img.height * r)), Image.LANCZOS)
            W, H = img.size
            vx, vy, vr = acts[t]
            d = ImageDraw.Draw(img)
            d.rectangle([0, H - 24, W, H], fill=(0, 0, 0))
            d.text((4, H - 20), f"t={t}/{n-1}  vx={vx:.2f} vy={vy:.2f}", fill=(255, 255, 255))
            d.text((4, 4), label, fill=(251, 191, 36))
            out = OUT / f"{prefix}_{t}.jpg"
            img.save(out, quality=85)
            saved.append(out.name)
        return saved


# v5_2 개별 프레임 저장 (시작/중간/끝)
v5_2_root = ROOT / "ROS_action/mobile_vla_dataset_v5_2"
scenarios = {
    "free_center":            list(v5_2_root.glob("*free_center*.h5")),
    "free_left":              list(v5_2_root.glob("*free_left*.h5")),
    "free_right":             list(v5_2_root.glob("*free_right*.h5")),
    "target_center_straight": list(v5_2_root.glob("*target_center_straight*.h5")),
    "target_center_left":     list(v5_2_root.glob("*target_center_left*.h5")),
    "target_center_right":    list(v5_2_root.glob("*target_center_right*.h5")),
}
for sc_name, eps in scenarios.items():
    if not eps:
        print(f"[SKIP] {sc_name}")
        continue
    ep = eps[0]
    saved = save_frames(ep, f"v5_2_{sc_name}", sc_name)
    print(f"[OK] {sc_name}: {saved}")

# 6/24 세션 단일 프레임
for sp in sorted((ROOT / "docs/inference_sessions").glob("session_20260624*.h5")):
    with h5py.File(sp, "r") as f:
        imgs_ds = f["observations/images"]
        bbox = f["grounding/bbox"][:]
        cached = f["grounding/cached"][:]
        acts = f["actions"][:]
        n = len(imgs_ds)
        for t in range(min(n, 3)):
            raw = imgs_ds[t]
            if raw.ndim == 1:
                import io
                img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
            else:
                img = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
            W, H = img.size
            cx, cy, area, has = bbox[t]
            d = ImageDraw.Draw(img)
            if has > 0.5 and area > 0:
                x1, y1, x2, y2 = box_from_cxcyarea(cx, cy, area, W, H)
                clr = (74, 222, 128) if cached[t] == 0 else (251, 191, 36)
                d.rectangle([x1, y1, x2, y2], outline=clr, width=3)
            has_str = "OK" if has > 0.5 else "MISS"
            d.rectangle([0, H - 22, W, H], fill=(0, 0, 0))
            d.text((4, H - 18), f"t={t}  has_bbox={has_str}  area={area:.4f}", fill=(255, 255, 255))
            out = OUT / f"session_20260624_{sp.stem[-6:]}_{t}.jpg"
            save_web(img, out)
    print(f"[OK] 6/24 {sp.stem} → {min(n,3)}장")

print("완료")
