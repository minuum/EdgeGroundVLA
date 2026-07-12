"""
CH61 §19 분석 리포트 생성: obj_right 세션들의 grounding(bbox) vs action(FORWARD 고착) 비교.
프레임 이미지를 로컬 파일로 저장하고, 분석 표+이미지 비교 HTML을 생성한다.
git-lfs 대상 아님(docs/v5/analysis_reports/frames_*/는 .gitignore에 추가해서 커밋 제외).
"""
import csv
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/minum/26CS/MoNaVLA")
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv/20260712")
OUT_DIR = ROOT / "docs/v5/analysis_reports"
FRAMES_DIR = OUT_DIR / "frames_20260712"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

SESSIONS = [
    "20260711_205228", "20260711_205354", "20260711_205621", "20260711_205726",
    "20260711_213650", "20260711_213749", "20260711_214155",
    "20260711_215709", "20260711_220142", "20260711_220233", "20260711_220439",
]

# episode_log.csv 결과 join
episode_by_sid = {}
with open(RECV / "episode_log.csv", newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) >= 14 and row[13] in SESSIONS:
            episode_by_sid[row[13]] = {
                "result": row[2], "steps": row[3], "note": row[11],
            }


def draw_overlay(im, bbox_row, action_row, label_text):
    """bbox=[cx,cy,area,has_bbox] (0~1 정규화), action=[vx,vyaw,?]"""
    W, H = im.size
    draw = ImageDraw.Draw(im, "RGBA")
    cx, cy, area, has_bbox = [float(x) for x in bbox_row]

    if has_bbox:
        # area = 정규화된 bbox 넓이(0~1) 가정 → 정사각 근사 한 변 비율
        side_frac = max(area, 1e-4) ** 0.5
        half_w, half_h = side_frac * W / 2, side_frac * H / 2
        x0, y0 = cx * W - half_w, cy * H - half_h
        x1, y1 = cx * W + half_w, cy * H + half_h
        draw.rectangle([x0, y0, x1, y1], outline=(52, 211, 153, 255), width=4)
        draw.ellipse([cx * W - 5, cy * H - 5, cx * W + 5, cy * H + 5], fill=(52, 211, 153, 255))
        bbox_txt = f"grounding: cx={cx:.2f} cy={cy:.2f} area={area:.2f} (탐지됨)"
        bbox_color = (52, 211, 153, 255)
    else:
        # fallback(center) — cx=0.5 지점에 X 표시
        draw.line([cx * W - 15, cy * H - 15, cx * W + 15, cy * H + 15], fill=(248, 113, 113, 255), width=4)
        draw.line([cx * W - 15, cy * H + 15, cx * W + 15, cy * H - 15], fill=(248, 113, 113, 255), width=4)
        bbox_txt = "grounding: 미탐지 (center fallback)"
        bbox_color = (248, 113, 113, 255)

    # cx 기준 세로 점선
    for y in range(0, H, 14):
        draw.line([cx * W, y, cx * W, y + 7], fill=bbox_color, width=2)

    # 액션 라벨 (회전 여부)
    vx, vyaw = float(action_row[0]), float(action_row[1])
    is_turn = abs(vyaw) > 1e-6
    action_txt = f"action: vx={vx:.2f} vyaw={vyaw:.2f}  {'[TURN]' if is_turn else '[FORWARD]'}"
    action_color = (251, 191, 36, 255) if is_turn else (156, 163, 175, 255)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    def text_with_bg(xy, text, color):
        tb = draw.textbbox(xy, text, font=font)
        pad = 3
        draw.rectangle([tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad], fill=(11, 18, 32, 210))
        draw.text(xy, text, fill=color, font=font)

    text_with_bg((8, 8), label_text, (147, 197, 253, 255))
    text_with_bg((8, 30), bbox_txt, bbox_color)
    text_with_bg((8, 52), action_txt, action_color)
    return im


def save_frame(sid, idx, images, bbox, actions, tag, label_text):
    img = images[idx]
    im = Image.fromarray(img).convert("RGB")
    im = draw_overlay(im, bbox[idx], actions[idx], label_text)
    im.thumbnail((560, 560))
    fn = f"{sid}_{tag}.jpg"
    im.save(FRAMES_DIR / fn, quality=88)
    return fn


rows = []
for sid in SESSIONS:
    h5p = RECV / f"session_{sid}.h5"
    with h5py.File(str(h5p), "r") as f:
        images = f["observations/images"][:]
        actions = f["actions"][:]
        bbox = f["grounding/bbox"][:]
        n = len(actions)

        first_detect = None
        for i in range(n):
            if bbox[i][3] == 1:
                first_detect = i
                break

        nonfwd_frames = [i for i, a in enumerate(actions) if abs(a[1]) > 1e-6]

        f0_fn = save_frame(sid, 0, images, bbox, actions, "frame0", f"{sid} · frame 0")
        det_fn = (
            save_frame(sid, first_detect, images, bbox, actions, "firstdetect", f"{sid} · 첫 탐지 (frame {first_detect})")
            if first_detect is not None else None
        )
        last_fn = save_frame(sid, n - 1, images, bbox, actions, "lastframe", f"{sid} · 마지막 프레임 (frame {n-1})")

        ep = episode_by_sid.get(sid, {})
        rows.append({
            "sid": sid,
            "result": ep.get("result", "?"),
            "note": ep.get("note", ""),
            "n_frames": n,
            "first_detect_frame": first_detect,
            "first_cx": round(float(bbox[first_detect][0]), 3) if first_detect is not None else None,
            "last_cx": round(float(bbox[n - 1][0]), 3),
            "last_has_bbox": bool(bbox[n - 1][3]),
            "nonfwd_frames": nonfwd_frames,
            "f0_fn": f0_fn,
            "det_fn": det_fn,
            "last_fn": last_fn,
        })

with open(OUT_DIR / "forward_lock_data_20260712.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)

print(f"{len(rows)}개 세션 처리 완료 → {OUT_DIR}")
for r in rows:
    print(r["sid"], r["result"], "first_detect=", r["first_detect_frame"], "nonfwd=", r["nonfwd_frames"])
