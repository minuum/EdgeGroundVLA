#!/usr/bin/env python3
"""방향 오차가 '이미지의 무엇을 어떤 각도로 바꾼 값인지'를 실제 프레임 위에 그린다.
오버레이: ① Florence-2 bbox 중심(+) 과 크기(원), ② 화면 정면 축(수직 점선, 로봇 전진 방향), ③ 우하단 '명령 나침반': 로봇 몸체 좌표에서 속도 명령 (lx,ly)의 방향(위=전진, 왼쪽=좌, 각도 θ=atan2(ly,lx)),
초록 = 사람 조작, 노랑 = 모델(EMA 0.5), 빨간 호 = 방향 오차. 입력: error_anatomy.json(fold0·LOPO 에피소드별 프레임 값), 캐시 bbox. 출력: docs/v5/egvla2_fm/anatomy/overlay_*.jpg
실행: .venv/bin/python3 -I scripts/make_anatomy_overlay.py"""
import json
import math
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"; OUT = ROOT / "docs/v5/egvla2_fm/anatomy"
d = json.load(open(ROOT / "docs/v5/egvla2_fm/error_anatomy.json")); cache = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
F = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 15); FS = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 12)
G, A, R = (74, 222, 128), (251, 191, 36), (248, 113, 113)
W, H = 640, 360


def vec_screen(v, cx, cy, s):
    m = math.hypot(v[0], v[1])
    if m < 1e-6:
        return cx, cy
    u = min(1.0, m)
    return cx - v[1] / m * s * u, cy - v[0] / m * s * u            # 위=전진(lx), 왼쪽=좌(ly>0)


def draw(frame_rgb, bbox, hum, pred, t, name):
    im = Image.fromarray(frame_rgb).resize((W, H)); dr = ImageDraw.Draw(im, "RGBA")
    # 정면 축
    for y in range(0, H, 14):
        dr.line([W // 2, y, W // 2, y + 7], fill=(255, 255, 255, 110), width=1)
    dr.text((W // 2 + 4, 4), "정면(전진 축)", font=FS, fill=(255, 255, 255, 200))
    cxp, cyp, area = bbox[0] * W, bbox[1] * H, bbox[2]
    if bbox[3] >= 0.5:
        r = max(8, math.sqrt(area * W * H) / 2)
        dr.ellipse([cxp - r, cyp - r, cxp + r, cyp + r], outline=(56, 189, 248, 255), width=3); dr.line([cxp - 8, cyp, cxp + 8, cyp], fill=(56, 189, 248, 255), width=3); dr.line([cxp, cyp - 8, cxp, cyp + 8], fill=(56, 189, 248, 255), width=3)
        dr.line([W // 2, cyp, cxp, cyp], fill=(56, 189, 248, 160), width=2)
        tx, ty = min(cxp + r + 6, W - 150), max(cyp - r - 22, 24); dr.rectangle([tx - 3, ty - 2, tx + 142, ty + 15], fill=(8, 12, 22, 200))
        dr.text((tx, ty), f"bbox 중심 x={bbox[0]:.2f}", font=FS, fill=(56, 189, 248, 255))
    # 명령 나침반
    cx0, cy0, S = W - 110, H - 22, 82
    dr.pieslice([cx0 - S - 8, cy0 - S - 8, cx0 + S + 8, cy0 + S + 8], 180, 360, fill=(8, 12, 22, 215))
    for rr in (S // 2, S):
        dr.arc([cx0 - rr, cy0 - rr, cx0 + rr, cy0 + rr], 180, 360, fill=(120, 130, 150, 255), width=1)
    dr.line([cx0, cy0, cx0, cy0 - S - 4], fill=(120, 130, 150, 255), width=1)
    dr.text((cx0 - 22, cy0 - S - 24), "전진 0°", font=FS, fill=(200, 210, 225, 255)); dr.text((cx0 - S - 8, cy0 - 16), "좌 +90°", font=FS, fill=(200, 210, 225, 255)); dr.text((cx0 + S - 34, cy0 - 16), "우 -90°", font=FS, fill=(200, 210, 225, 255))
    th = lambda v: math.degrees(math.atan2(v[1], v[0]))
    mh, mp = math.hypot(*hum[:2]) >= 0.2, math.hypot(*pred[:2]) >= 0.2
    for v, col, ok in ((hum, G, mh), (pred, A, mp)):
        if ok:
            x, y = vec_screen(v, cx0, cy0, S); dr.line([cx0, cy0, x, y], fill=col + (255,), width=4); dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=col + (255,))
    if mh and mp:
        a1, a2 = th(hum), th(pred); err = abs((a1 - a2 + 180) % 360 - 180)
        rr = S * 0.45; s0, s1 = sorted([90 - a1, 90 - a2]); dr.arc([cx0 - rr, cy0 - rr, cx0 + rr, cy0 + rr], 270 + min(-a1, -a2) - 0 if False else 270 - max(a1, a2), 270 - min(a1, a2), fill=R + (255,), width=4)
        dr.text((10, H - 62), f"사람 {a1:+.0f}°   모델 {a2:+.0f}°   방향 오차 {err:.0f}°", font=F, fill=(255, 255, 255, 255))
    else:
        dr.text((10, H - 62), "정지/회전 프레임(방향 오차 계산 제외)", font=F, fill=(255, 255, 255, 255))
    dr.text((10, 8), f"프레임 #{t}", font=F, fill=(255, 255, 255, 255)); dr.text((10, H - 40), "초록=사람 조작, 노랑=모델, 빨강 호=오차", font=FS, fill=(255, 255, 255, 220))
    dr.text((10, H - 22), "각도 = 로봇 몸체 좌표의 속도 명령 방향(이미지 픽셀 각도가 아님)", font=FS, fill=(200, 210, 225, 255))
    return im


meta = {}
for sp in ("fold0", "lopo_strong_right"):
    for role, nm in d["examples"][sp].items():
        it = next(x for x in d["splits"][sp] if x["name"] == nm); n = it["n"]; err = np.array([np.nan if v is None else v for v in it["err"]])
        mov = np.where(~np.isnan(err))[0]
        picks = [int(mov[int(len(mov) * q)]) for q in (0.1, 0.35, 0.6, 0.85)]
        if role == "worst":
            picks[2] = int(np.nanargmax(err))
        bb = cache[nm + "|bbox"]
        with h5py.File(DATA / nm, "r") as h:
            for k, t in enumerate(picks):
                fr = np.ascontiguousarray(h["images"][t][:, :, ::-1])
                draw(fr, bb[t], it["human"][t], it["pred"][t], t, nm).save(OUT / f"overlay_{sp}_{role}_{k}.jpg", quality=80)
        meta[f"{sp}|{role}"] = dict(name=nm, frames=picks)
json.dump(meta, open(ROOT / "docs/v5/egvla2_fm/anatomy/overlay_meta.json", "w"), ensure_ascii=False)
print("ok", len(meta))
