#!/usr/bin/env python3
"""11/16 실기 프레임(a3-s2가 실제로 달린 경로) 위에서 a3·a3-s1·a3-s2·앙상블 헤드의 정지 판단을 박스 면적 구간별로 비교.
11/13 프레임은 a3가 멈춘 곳(면적 ≈0.23)에서 끝나는 검열이 있었지만, 11/16 프레임은 면적 0.3~0.55까지 있어 그 너머를 볼 수 있다.
'정지' = 헤드 첫 명령 크기 < 0.2. 단 a3의 판단은 a3-s2의 경로에 대한 반사실(a3가 이 프레임들에서 무엇을 냈을지)이다. 출력 docs/v5/egvla2_fm/a3c/replay_1116_stop_by_area.json"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent; A = ROOT / "docs/v5/egvla2_fm/a3c"
R = json.load(open(A / "replay_1116.json")); R.pop("_validation", None)
heads = ["a3", "a3-s1", "a3-s2", "a3-ens3"]; nm = lambda c: np.linalg.norm(np.array(c), axis=1)
OB = {"go to the gray basket": "바구니", "go to the brown wooden chair": "의자", "go to the small desk": "책상"}
bins = [0, .1, .15, .2, .25, .3, .4, 1.0]; lab = ["<0.10", "0.10~0.15", "0.15~0.20", "0.20~0.25", "0.25~0.30", "0.30~0.40", "≥0.40"]
out = {"bins": lab, "by_area": {}, "by_object_ge025": {}, "last3": {}, "move_frac": {}}
for h in heads:
    st = np.zeros(len(lab)); tot = np.zeros(len(lab))
    for v in R.values():
        a = np.array(v["area_re"]); n = nm(v["cmd"][h]); b = np.digitize(a, bins[1:-1])
        for k in range(len(lab)): m = b == k; tot[k] += m.sum(); st[k] += (n[m] < .2).sum()
    out["by_area"][h] = dict(n=tot.astype(int).tolist(), stop_rate=[round(float(s / t), 3) if t else None for s, t in zip(st, tot)])
    out["move_frac"][h] = round(float(np.mean([(nm(v["cmd"][h]) > .2).mean() for v in R.values()])), 3)
    for o in ("바구니", "의자", "책상"):
        t = s_ = 0; e = []
        for v in R.values():
            if OB[v["instruction"]] != o: continue
            a = np.array(v["area_re"]); n = nm(v["cmd"][h]); m = a >= .25; t += int(m.sum()); s_ += int((n[m] < .2).sum()); e.append(float((n[-3:] < .2).mean()))
        out["by_object_ge025"].setdefault(o, {})[h] = dict(frames=t, stop=s_, rate=round(s_ / t, 3) if t else None); out["last3"].setdefault(o, {})[h] = round(float(np.mean(e)), 3)
json.dump(out, open(A / "replay_1116_stop_by_area.json", "w"), ensure_ascii=False, indent=1)
print("면적 구간별 정지율(헤드: " + ", ".join(lab) + ")")
for h in heads: print(h, out["by_area"][h]["stop_rate"], "n", out["by_area"][h]["n"] if h == "a3" else "")
print(json.dumps(out["by_object_ge025"], ensure_ascii=False)); print(json.dumps(out["last3"], ensure_ascii=False))
