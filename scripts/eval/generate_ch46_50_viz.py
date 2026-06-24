#!/usr/bin/env python3
"""
research_story.html CH46~50 절을 위한 실제 이미지 시각화 생성.
PG2 grounding 결과(cx,cy,area)를 h5 원본 프레임 위에 bbox로 오버레이해서
"실제로 무엇을 보고 학습했는지"를 보여준다. 새 그라운딩을 다시 돌리지 않고
이미 저장된 cx/cy/area 값만 사용 — 그라운딩 모델 재로드 불필요.

생성 대상:
  ch46_kosmos_vs_pg2_*.png   : 같은 프레임, Kosmos2(구) vs PG2(신) bbox 비교 (빨강 vs 초록)
  ch50_zoom_before_after_*.png : area<0.05 프레임의 원본 bbox(빨강, 작음) vs 줌 재그라운딩 bbox(초록)
  ch49_skip3_cache_seq_*.png : 연속 3프레임에서 skip_n=3 캐시로 bbox가 고정되는 모습(시퀀스 합성)

Usage:
  .venv/bin/python3 scripts/eval/generate_ch46_50_viz.py
"""
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "docs/v5/ch46_50_viz"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OLD = json.loads((ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json").read_text())
PG2 = json.loads((ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json").read_text())
ZOOM = json.loads((ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2_zoomsmall.json").read_text())
SKIP3 = json.loads((ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2_skip3.json").read_text())


def load_frame(h5_path, idx):
    with h5py.File(h5_path, "r") as f:
        img = f["observations"]["images"][idx]
    return Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGB")


def box_from_cxcyarea(cx, cy, area, w, h):
    side = (area ** 0.5)
    x1 = (cx - side / 2) * w
    y1 = (cy - side / 2) * h
    x2 = (cx + side / 2) * w
    y2 = (cy + side / 2) * h
    return [x1, y1, x2, y2]


def draw_box(img, box, color, label=None):
    img = img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle(box, outline=color, width=3)
    if label:
        d.text((box[0] + 2, max(0, box[1] - 14)), label, fill=color)
    return img


def side_by_side(im1, im2, gap=8):
    w, h = im1.size
    canvas = Image.new("RGB", (w * 2 + gap, h), (20, 20, 20))
    canvas.paste(im1, (0, 0))
    canvas.paste(im2, (w + gap, 0))
    return canvas


WEB_MAX_W = 1000  # 웹 표시용 다운스케일 — 원본 해상도 불필요(파일당 ~1.3MB -> ~150KB)


def save_web(img, path):
    if img.width > WEB_MAX_W:
        ratio = WEB_MAX_W / img.width
        img = img.resize((WEB_MAX_W, int(img.height * ratio)), Image.LANCZOS)
    img.save(path, quality=85, optimize=True)


def find_existing(h5_episode):
    p = Path(h5_episode)
    return p if p.exists() else None


def gen_ch46_kosmos_vs_pg2(n=4):
    """46-4: 같은 프레임에서 구(Kosmos2) vs 신(PG2) bbox가 얼마나 다른지."""
    saved = 0
    for ep_o, ep_n in zip(OLD, PG2):
        if saved >= n:
            break
        h5 = find_existing(ep_o["episode"])
        if h5 is None:
            continue
        for fr_o, fr_n in zip(ep_o["frames"], ep_n["frames"]):
            if not (fr_o.get("has_bbox") and fr_n.get("has_bbox")):
                continue
            img = load_frame(h5, fr_o["frame_idx"])
            w, h = img.size
            box_o = box_from_cxcyarea(fr_o["cx"], fr_o["cy"], fr_o["area"], w, h)
            box_n = box_from_cxcyarea(fr_n["cx"], fr_n["cy"], fr_n["area"], w, h)
            im_o = draw_box(img, box_o, (248, 113, 113), f"Kosmos2(구) area={fr_o['area']:.3f}")
            im_n = draw_box(img, box_n, (74, 222, 128), f"PG2(신) area={fr_n['area']:.3f}")
            combo = side_by_side(im_o, im_n)
            out = OUT_DIR / f"ch46_kosmos_vs_pg2_{saved+1}.jpg"
            save_web(combo, out)
            print(f"[저장] {out}  (delta_area={fr_n['area']-fr_o['area']:+.3f})")
            saved += 1
            break
    return saved


def gen_ch50_zoom_before_after(n=4):
    """50-1: area<0.05 작은 객체의 원본 bbox vs 줌 재그라운딩 bbox."""
    saved = 0
    for ep_o, ep_n in zip(PG2, ZOOM):
        if saved >= n:
            break
        h5 = find_existing(ep_o["episode"])
        if h5 is None:
            continue
        for fr_o, fr_n in zip(ep_o["frames"], ep_n["frames"]):
            if not (fr_o.get("has_bbox") and fr_o["area"] < 0.05):
                continue
            img = load_frame(h5, fr_o["frame_idx"])
            w, h = img.size
            box_o = box_from_cxcyarea(fr_o["cx"], fr_o["cy"], fr_o["area"], w, h)
            box_n = box_from_cxcyarea(fr_n["cx"], fr_n["cy"], fr_n["area"], w, h)
            im_o = draw_box(img, box_o, (248, 113, 113), f"원본 area={fr_o['area']:.4f}")
            im_n = draw_box(img, box_n, (74, 222, 128), f"줌 재그라운딩 area={fr_n['area']:.4f}")
            combo = side_by_side(im_o, im_n)
            out = OUT_DIR / f"ch50_zoom_before_after_{saved+1}.jpg"
            save_web(combo, out)
            print(f"[저장] {out}")
            saved += 1
            break
    return saved


def gen_ch49_skip3_cache_seq(n=2):
    """49: skip_n=3 캐시로 3프레임 연속 bbox가 고정되는 모습 (1 fresh + 2 cached)."""
    saved = 0
    for ep in SKIP3:
        if saved >= n:
            break
        h5 = find_existing(ep["episode"])
        if h5 is None or len(ep["frames"]) < 6:
            continue
        frames = ep["frames"]
        # 캐시가 고정된 3프레임 구간 찾기 (t%3!=0에서 동일 area)
        for t in range(3, len(frames) - 2):
            f0, f1, f2 = frames[t], frames[t + 1], frames[t + 2]
            if f0["area"] == f1["area"] == f2["area"] and f0.get("has_bbox"):
                imgs = []
                w = h = None
                for fr in (f0, f1, f2):
                    im = load_frame(h5, fr["frame_idx"])
                    w, h = im.size
                    box = box_from_cxcyarea(fr["cx"], fr["cy"], fr["area"], w, h)
                    imgs.append(draw_box(im, box, (251, 191, 36), f"t={fr['frame_idx']} area={fr['area']:.4f}(캐시)"))
                gap = 6
                canvas = Image.new("RGB", (w * 3 + gap * 2, h), (20, 20, 20))
                for i, im in enumerate(imgs):
                    canvas.paste(im, (i * (w + gap), 0))
                out = OUT_DIR / f"ch49_skip3_cache_seq_{saved+1}.jpg"
                save_web(canvas, out)
                print(f"[저장] {out}")
                saved += 1
                break
    return saved


if __name__ == "__main__":
    n1 = gen_ch46_kosmos_vs_pg2(n=12)
    n2 = gen_ch50_zoom_before_after(n=12)
    n3 = gen_ch49_skip3_cache_seq(n=6)
    print(f"\n총 생성: ch46={n1}, ch50={n2}, ch49={n3}")
