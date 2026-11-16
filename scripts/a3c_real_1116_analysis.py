#!/usr/bin/env python3
"""11/16 a3-s2(A3C_D30_s2.pt) 실기 45세션 vs 11/13 a3 실기: 같은 컨트롤러(near_area 0.25)에서 종료 시점 박스 면적 비교.
입력: docs/v5/egvla2_fm/a3c/real_1116_*(로봇 서버에서 받은 CSV 슬라이스·면적 시계열), objtest_data.json(11/13 a3). 출력 real_1116_vs_1113.json
면적은 지시어로 찾은 박스 기준(물체마다 크기가 다름). 'n_ge025' = 면적≥0.25인 프레임 수(정지 규칙 near_area=0.25는 연속 2프레임 필요)."""
import csv, json, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; A = ROOT / "docs/v5/egvla2_fm/a3c"
rows = {r["session_id"]: r for r in csv.DictReader(open(A / "real_1116_episode_log_slice.csv"))}; S = json.load(open(A / "real_1116_sessions_bbox_area.json"))
OB = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}; PO = {"trackF_center": "가운데", "trackA_weak_right": "오른쪽", "trackA_weak_left": "왼쪽"}
def stats(v):
    v = np.array(v, float); return dict(n=len(v), 성공=int(v[:, 0].sum()), 평균프레임=round(float(v[:, 1].mean()), 1), 첫면적=round(float(v[:, 2].mean()), 3), 마지막면적=round(float(v[:, 3].mean()), 3), 마지막면적_범위=[round(float(v[:, 3].min()), 3), round(float(v[:, 3].max()), 3)], n_ge025=round(float(v[:, 4].mean()), 1))
G = collections.defaultdict(list)
for sid, s in S.items():
    r = rows[sid]; a = np.array(s["bbox_area_seq"]); G[(OB[s["instruction"]], PO[r["경로"]])].append((r["결과"] == "성공", s["n_frames"], a[0], a[-1], (a >= .25).sum()))
D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json")); H = collections.defaultdict(list)
for x in D:
    if x["pos"] == "기록 없음": continue
    a = np.array(x["bbox"])[:, 2]; H[({"작은 책상": "책상"}.get(x["obj"], x["obj"]), x["pos"][:2].replace("가운", "가운데").replace("오른", "오른쪽"))].append((x["res"] == "성공", x["n"], a[0], a[-1], (a >= .25).sum()))
R = {"a3-s2(11/16)": {f"{k[0]}·{k[1]}": stats(v) for k, v in sorted(G.items())}, "a3(11/13)": {f"{k[0]}·{k[1]}": stats(v) for k, v in sorted(H.items())}}
json.dump(R, open(A / "real_1116_vs_1113.json", "w"), ensure_ascii=False, indent=1)
for m, d in R.items():
    print(m)
    for k, v in d.items(): print("  ", k, v)
