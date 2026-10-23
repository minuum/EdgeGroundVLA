#!/usr/bin/env python3
"""세션·학습 에피소드 필름스트립 프레임 생성: 프레임마다 bbox, 명령 화살표(노랑), 관측된 장면 이동 화살표(청록)를 덧그린다.
화살표 규약: 명령 이동 화살표 = 위가 전진, 화면 가로 = ly(ly<0이면 왼쪽으로 이동: 데이터·실기 공통 관측), 회전 = 위쪽 호 화살표(+az 우회전, −az 좌회전).
관측 화살표 = 다음 프레임과의 장면 수평 이동(위상 상관): 장면이 왼쪽으로 가면(= 로봇 우회전) 왼쪽 화살표.
출력: docs/v5/egvla2_fm/filmstrips/*.jpg, filmstrips.json. 실행: .venv_lerobot/bin/python3 scripts/make_filmstrips.py
"""
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "docs/v5/egvla2_fm/filmstrips"
FNT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 17)
W, H = 448, 252; SC = W / 256
WIN = cv2.createHanningWindow((320, 110), cv2.CV_32F)
REP = ROOT / "ROS_action/dryrun_closedloop_20261022"
EXCLUDE = {"20261022_193630", "20261022_194447", "20261022_201157"}   # 저장 오류(0프레임·정지 이미지·액션 0) — 사용자 지정 제외


def gray(a):
    g = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY)
    return np.float32(cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)[:110])


def arrow(d, p0, p1, col, w=4):
    d.line([p0, p1], fill=col, width=w)
    v = np.array(p1, float) - np.array(p0, float); n = np.linalg.norm(v)
    if n < 2:
        return
    u = v / n; l = np.array([-u[1], u[0]])
    for s in (1, -1):
        d.line([p1, tuple(np.array(p1) - u * 9 * SC + l * 5 * SC * s)], fill=col, width=w)


def frame_img(arr, bbox, act, dx, col_bbox, text):
    im = Image.fromarray(np.ascontiguousarray(arr)).resize((W, H), Image.LANCZOS); d = ImageDraw.Draw(im)
    if bbox is not None and bbox[3] > 0.5:
        cx, cy = bbox[0] * W, bbox[1] * H; r = max(6, int(np.sqrt(max(bbox[2], 1e-4) * W * H) / 2)); d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=col_bbox, width=3)
    lx, ly, az = act
    if abs(az) > 0.1:                                                      # 회전: 위쪽 가로 호 화살표
        s = 1 if az > 0 else -1
        arrow(d, (W / 2 - 30 * SC * s, 34 * SC), (W / 2 + 30 * SC * s, 34 * SC), (255, 220, 40), 6)
    elif np.hypot(lx, ly) > 0.05:                                          # 이동: 아래 중앙에서 (ly → 가로, lx → 위)
        v = np.array([ly, -lx]) / max(np.hypot(lx, ly), 1e-6) * 42 * SC
        arrow(d, (W / 2, H - 14 * SC), (W / 2 + v[0], H - 14 * SC + v[1]), (255, 220, 40), 6)
    if dx is not None and abs(dx) > 0.5:                                   # 관측: 장면 가로 이동(px@320)
        L = float(np.clip(dx * SC, -60 * SC, 60 * SC)); arrow(d, (W / 2, H / 2), (W / 2 + L, H / 2), (60, 220, 255), 5)
    d.rectangle([0, 0, W, 24], fill=(0, 0, 0)); d.text((5, 2), text, fill=(255, 255, 255), font=FNT)
    return im


def main():
    idx = {"live": [], "train": []}
    for rp in sorted((REP / "reports").glob("*.json")):
        d = json.load(open(rp)); hp = REP / f"h5/session_{d['session_id']}.h5"
        if not hp.exists() or d["session_id"] in EXCLUDE:
            continue
        with h5py.File(hp, "r") as f:
            im = f["observations/images"]; n = min(im.shape[0], len(d["history"]))
            if n < 3:
                continue
            frames = [np.asarray(im[t]) for t in range(n)]
        g = [gray(x) for x in frames]; dxs = [cv2.phaseCorrelate(g[t], g[t + 1], WIN)[0][0] for t in range(n - 1)] + [None]
        rc = d["runtime_config"]; rc = json.loads(rc) if isinstance(rc, str) else rc
        items = []
        for t in range(n):
            h = d["history"][t]; b = h["bbox"]; bb = np.array([b["cx"], b["cy"], b["area"], 1.0 if b["has_bbox"] else 0.0]) if b else None
            lab = h["predicted_label"].split(":")[-1].split("(")[0]
            name = f"{d['session_id']}_{t:02d}.jpg"
            frame_img(frames[t], bb, h["action"], dxs[t], (255, 80, 80), f"#{t} {lab}").convert("RGB").save(OUT / name, "JPEG", quality=82)
            items.append(dict(f=name, t=t, label=lab, cx=None if not (b and b["has_bbox"]) else round(b["cx"], 2), area=None if not (b and b["has_bbox"]) else round(b["area"], 3),
                              lx=round(h["action"][0], 2), ly=round(h["action"][1], 2), az=round(h["action"][2], 2), dx=None if dxs[t] is None else round(float(dxs[t]), 1)))
        idx["live"].append(dict(sid=d["session_id"], mode=d.get("instruction_mode"), model=d["model_name"][-30:], target=rc.get("target_position"), status=d.get("status"), items=items))
    eps = S1.load_main()
    for cell in ("weak_right|straight", "strong_left|left_curve", "center|straight", "weak_left|right_curve"):
        e = next(x for x in eps if x["cell"] == cell)
        with h5py.File(T.DATA / e["name"], "r") as f:
            im = f["images"]; n = im.shape[0]; frames = [np.ascontiguousarray(np.asarray(im[t])[:, :, ::-1]) for t in range(n)]
        g = [gray(x) for x in frames]; dxs = [cv2.phaseCorrelate(g[t], g[t + 1], WIN)[0][0] for t in range(n - 1)] + [None]
        items = []
        for t in range(n):
            a = e["chunk"][t, 0]; bb = e["bbox"][t][8:12]; name = f"train_{cell.replace('|', '_')}_{t:02d}.jpg"
            frame_img(frames[t], bb, a, dxs[t], (80, 220, 80), f"#{t} 사람").convert("RGB").save(OUT / name, "JPEG", quality=82)
            items.append(dict(f=name, t=t, label="사람", cx=round(float(bb[0]), 2), area=round(float(bb[2]), 3), lx=round(float(a[0]), 2), ly=round(float(a[1]), 2), az=round(float(a[2]), 2), dx=None if dxs[t] is None else round(float(dxs[t]), 1)))
        idx["train"].append(dict(cell=cell, name=e["name"][-40:], items=items))
    (ROOT / "docs/v5/egvla2_fm/filmstrips.json").write_text(json.dumps(idx, ensure_ascii=False))
    print("live", sum(len(x["items"]) for x in idx["live"]), "train", sum(len(x["items"]) for x in idx["train"]))


if __name__ == "__main__":
    main()
