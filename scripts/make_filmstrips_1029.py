#!/usr/bin/env python3
"""10/29 위치 일반화 96회 중 대표 세션의 필름스트립(프레임별 bbox·명령 화살표·장면 이동). make_filmstrips.py의 그리기 함수를 재사용.
선택 규칙(결과를 보고 고르지 않음): 출발 위치(pos1~6)마다 프레임 수 중앙값에 가장 가까운 세션 1개(동률이면 시각이 이른 것) + 재판정 대상 200618.
출력: docs/v5/egvla2_fm/filmstrips1029/*.jpg, filmstrips1029.json. 실행: .venv/bin/python3 scripts/make_filmstrips_1029.py"""
import csv, json, sys
from pathlib import Path
import cv2, h5py, numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import make_filmstrips as MF  # noqa: E402

OUT = ROOT / "docs/v5/egvla2_fm/filmstrips1029"; OUT.mkdir(parents=True, exist_ok=True)
rows = json.load(open(ROOT / "docs/v5/egvla2_fm/gen_1029_summary.json"))
sel = []
for pos in ["pos1", "pos2", "pos3", "pos4", "pos5", "pos6"]:
    rs = sorted([r for r in rows if r["start"] == pos], key=lambda r: r["sid"]); med = np.median([r["n"] for r in rs])
    sel.append(min(rs, key=lambda r: (abs(r["n"] - med), r["sid"])))
if not any(r["sid"].endswith("200618") for r in sel):
    sel.append(next(r for r in rows if r["sid"].endswith("200618")))
idx = []
for r in sel:
    f = h5py.File(ROOT / f"docs/inference_sessions/session_{r['sid']}.h5"); im = f["observations/images"]; act = f["actions"][:]; bb = f["grounding/bbox"][:]; n = len(bb)
    frames = [np.ascontiguousarray(np.asarray(im[t])) for t in range(n)]
    g = [MF.gray(x) for x in frames]; dxs = [cv2.phaseCorrelate(g[t], g[t + 1], MF.WIN)[0][0] for t in range(n - 1)] + [None]
    items = []
    for t in range(n):
        name = f"{r['sid']}_{t:02d}.jpg"; has = bb[t, 3] > .5
        MF.frame_img(frames[t], bb[t], act[t], dxs[t], (255, 80, 80), f"#{t}").convert("RGB").save(OUT / name, "JPEG", quality=82)
        items.append(dict(f=name, t=t, cx=round(float(bb[t, 0]), 2) if has else None, area=round(float(bb[t, 2]), 3) if has else None,
                          lx=round(float(act[t, 0]), 2), ly=round(float(act[t, 1]), 2), az=round(float(act[t, 2]), 2), dx=None if dxs[t] is None else round(float(dxs[t]), 1)))
    idx.append(dict(sid=r["sid"], start=r["start"], target=r["target"], stage=r["stage"], items=items))
(ROOT / "docs/v5/egvla2_fm/filmstrips1029.json").write_text(json.dumps(idx, ensure_ascii=False))
print([(x["sid"], x["start"], x["target"], len(x["items"])) for x in idx], sum(len(x["items"]) for x in idx))
