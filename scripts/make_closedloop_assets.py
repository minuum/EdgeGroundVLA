#!/usr/bin/env python3
"""CH79 폐루프 점검 탭용 자산 생성: 실기 세션 스트립 사진(회전 중앙화 / 오버슈트 / 이탈)과 학습 시작 구간 스트립, 세션별 스텝 요약 JSON.
출력: docs/v5/egvla2_fm/closedloop_photos/*.jpg, closedloop_summary.json. 실행: .venv_lerobot/bin/python3 scripts/make_closedloop_assets.py
"""
import ast
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "docs/v5/egvla2_fm"; PH = OUT / "closedloop_photos"; PH.mkdir(exist_ok=True)
FONT = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf", 24)
W, H = 640, 360
REP = ROOT / "ROS_action/dryrun_closedloop_20261022"


def tile(arr, label, bbox=None, col=(255, 80, 80)):
    im = Image.fromarray(np.asarray(arr)).resize((W, H)); d = ImageDraw.Draw(im)
    if bbox is not None and bbox[3] > 0.5:
        cx, cy = bbox[0] * W, bbox[1] * H; r = max(8, int(np.sqrt(bbox[2] * W * H) / 2))
        d.rectangle([cx - r, cy - r, cx + r, cy + r], outline=col, width=3)
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0)); d.text((6, 3), label, fill=(255, 255, 255), font=FONT)
    return im


def strip(tiles, name):
    rows = (len(tiles) + 2) // 3; c = Image.new("RGB", (W * 3, H * rows), (10, 10, 10))
    for k, t in enumerate(tiles):
        c.paste(t, ((k % 3) * W, (k // 3) * H))
    c.convert("RGB").save(PH / name, "JPEG", quality=80); return f"egvla2_fm/closedloop_photos/{name}"


def live_strip(sid, steps, name):
    rep = json.load(open(REP / f"reports/session_{sid}.json"))["history"]
    tiles = []
    with h5py.File(REP / f"h5/session_{sid}.h5", "r") as f:
        im = f["observations/images"]
        for s in steps:
            if s >= im.shape[0] or s >= len(rep):
                continue
            r = rep[s]; b = r["bbox"]; a = r["action"]
            lab = f"스텝{s} {r['predicted_label'].split(':')[-1][:9]} cx{b['cx']:.2f} az{a[2]:+.2f} ly{a[1]:+.2f}" if b else f"스텝{s}"
            tiles.append(tile(im[s], lab, np.array([b["cx"], b["cy"], b["area"], 1.0 if b["has_bbox"] else 0.0]) if b else None))
    return strip(tiles, name)


def main():
    out = {}
    out["strip_center"] = live_strip("20261022_204813", [0, 1, 2, 3, 5, 7], "live_204813_center.jpg")          # 한 번 회전 → 중앙, 이후 cx 점진 이탈
    out["strip_overshoot"] = live_strip("20261022_205118", [3, 4, 5, 6, 7, 9], "live_205118_overshoot.jpg")      # 두 번 회전 → 오버슈트
    out["strip_drift"] = live_strip("20261022_210745", [0, 1, 4, 8, 14, 18], "live_210745_drift.jpg")             # 회전 → 직진 중 화면 밖
    eps = S1.load_main(); e = next(x for x in eps if x["cell"] == "weak_right|straight"); tiles = []
    with h5py.File(T.DATA / e["name"], "r") as f:
        for t in (0, 2, 4, 6, 10, 20):
            b = e["bbox"][t][8:12]; a = e["chunk"][t, 0]
            tiles.append(tile(np.ascontiguousarray(f["images"][t][:, :, ::-1]), f"학습 프레임{t} cx{b[0]:.2f} 사람 az{a[2]:+.1f} lx{a[0]:+.1f}", b, (80, 220, 80)))
    out["strip_train"] = strip(tiles, "train_weak_right_start.jpg")
    # 세션별 스텝 요약(차트용)
    sess = {}
    for rp in sorted((REP / "reports").glob("*.json")):
        d = json.load(open(rp))
        if "egvla2" not in d["model_name"] or d.get("instruction_mode") != "SYNC":
            continue
        rows = []
        for s in d["history"]:
            b = s["bbox"]; cf = s.get("chunk_first")
            cf = ast.literal_eval(cf) if isinstance(cf, str) else cf
            rows.append(dict(cx=b["cx"] if b and b["has_bbox"] else None, area=b["area"] if b and b["has_bbox"] else None, label=s["predicted_label"].split(":")[-1].split("(")[0],
                             az=s["action"][2], ly=s["action"][1], lx=s["action"][0], raw_angle=float(np.degrees(np.arctan2(cf[1], cf[0]))) if cf else None))
        sess[d["session_id"]] = rows
    out["sessions"] = sess
    # 학습 분포: cx vs 사람 방향각(이동 프레임) 샘플
    cx, ang = [], []
    for e in eps[::3]:
        a = e["chunk"][:, 0]; b = e["bbox"]
        for t in range(0, len(a), 3):
            if b[t][11] > 0.5 and np.linalg.norm(a[t, :2]) > 0.2 and abs(a[t, 2]) < 0.1:
                cx.append(float(b[t][8])); ang.append(float(np.degrees(np.arctan2(a[t, 1], a[t, 0]))))
    out["train_scatter"] = dict(cx=cx[:1500], ang=ang[:1500])
    (OUT / "closedloop_summary.json").write_text(json.dumps(out, ensure_ascii=False))
    print("ok", len(sess), len(cx))


if __name__ == "__main__":
    main()
